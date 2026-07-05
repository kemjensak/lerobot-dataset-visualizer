"""Trim endpoints for the LeRobot dataset visualizer backend.

Cuts the motionless head/tail off episodes and writes a *trimmed copy* of the
dataset (the source is never modified). Two endpoints, mounted by ``app.py``:

- ``POST /api/trim/detect`` — scan every episode's action (or state) data at
  full resolution and suggest a keep-range per episode. Mirrors the
  client-side algorithm in ``src/utils/trimDetection.ts`` — keep in sync.
- ``POST /api/trim/apply`` — apply per-episode keep-ranges (seconds,
  episode-relative):

  * data parquet rows outside the range are dropped; ``timestamp`` is rebased
    to 0, ``frame_index`` renumbered per episode, ``index`` re-sequenced
    globally, and the last kept row's ``next.done`` set to True.
  * **v3.0**: videos are untouched (lossless) — the episode's
    ``videos/{key}/from_timestamp``/``to_timestamp`` window in the episode
    metadata is narrowed instead. Episode metadata (lengths,
    ``dataset_from/to_index``, per-episode ``stats/*``) is rewritten in the
    original chunk/file layout.
  * **v2.0 / v2.1**: per-episode videos are re-cut with ffmpeg (re-encode to
    h264). ``meta/episodes.jsonl`` lengths and per-episode
    ``meta/episodes_stats.jsonl`` are rewritten.
  * ``meta/stats.json`` (when present) is recomputed for features stored in
    the data parquet; image/video stats are carried over unchanged.

Row selection is timestamp-based (±half a frame tolerance), so ranges coming
from the visualizer's (possibly downsampled) charts land on exact frames.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import APIRouter, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel

logger = logging.getLogger("lerobot-trim")

router = APIRouter()

TRIM_CACHE_ROOT = Path(
    os.environ.get("LEROBOT_TRIM_CACHE", "/tmp/lerobot_visualizer_trim_cache")
)
TRIM_EXPORT_ROOT = Path(
    os.environ.get("LEROBOT_TRIM_EXPORT", "/tmp/lerobot_visualizer_trim_exports")
)

SMOOTHING_WINDOW = 5
MOTION_FEATURE_CANDIDATES = ("action", "observation.state")
STAT_NAMES = ("min", "max", "mean", "std", "count")


# --- Requests -----------------------------------------------------------------


class TrimRangeModel(BaseModel):
    start: float
    end: float


class TrimDatasetRef(BaseModel):
    repo_id: str | None = None
    revision: str | None = None
    local_path: str | None = None


class TrimDetectRequest(TrimDatasetRef):
    sensitivity: float = 0.1
    padding_seconds: float = 0.25
    min_trim_seconds: float = 0.1


class TrimApplyRequest(TrimDatasetRef):
    trims: dict[str, TrimRangeModel]
    output_dir: str | None = None


# --- Dataset resolution ---------------------------------------------------------


def _resolve_root(req: TrimDatasetRef, allow_patterns: list[str]) -> Path:
    if req.local_path:
        root = Path(req.local_path).expanduser().resolve()
        if not root.exists():
            raise HTTPException(status_code=404, detail=f"Dataset path not found: {root}")
        return root
    if req.repo_id:
        TRIM_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        slug = req.repo_id.replace("/", "__") + (f"@{req.revision}" if req.revision else "")
        root = TRIM_CACHE_ROOT / slug
        root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            req.repo_id,
            repo_type="dataset",
            revision=req.revision,
            local_dir=root,
            allow_patterns=allow_patterns,
        )
        return root
    raise HTTPException(status_code=400, detail="need repo_id or local_path")


def _load_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing meta/info.json at {root}")
    return json.loads(info_path.read_text())


def _major_version(info: dict[str, Any]) -> int:
    version = str(info.get("codebase_version", ""))
    if version.startswith("v2"):
        return 2
    if version.startswith("v3"):
        return 3
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported codebase_version for trim: {version!r} (need v2.x or v3.0)",
    )


# --- Motion detection (mirror of src/utils/trimDetection.ts) --------------------


def _column_matrix(values: pd.Series) -> np.ndarray | None:
    """Stack a parquet column's cells into an (n, D) float matrix."""
    if len(values) == 0:
        return None
    first = values.iloc[0]
    try:
        if isinstance(first, (list, np.ndarray)):
            mat = np.stack([np.asarray(v, dtype=np.float64) for v in values])
            if mat.ndim == 1:
                mat = mat[:, None]
            # Flatten any higher-rank features (e.g. (n, 2, 3) → (n, 6)).
            return mat.reshape(mat.shape[0], -1)
        return np.asarray(values, dtype=np.float64)[:, None]
    except (TypeError, ValueError):
        return None


def _motion_signal(mat: np.ndarray) -> np.ndarray | None:
    """Per-frame mean of |Δ| across dims, each normalized by its value range."""
    if mat.shape[0] < 3:
        return None
    finite = np.where(np.isfinite(mat), mat, np.nan)
    mins = np.nanmin(finite, axis=0)
    maxs = np.nanmax(finite, axis=0)
    scales = maxs - mins
    usable = scales > 1e-9
    if not usable.any():
        return None
    deltas = np.abs(np.diff(finite[:, usable], axis=0)) / scales[usable]
    motion = np.zeros(mat.shape[0], dtype=np.float64)
    motion[1:] = np.nanmean(deltas, axis=1)
    motion = np.nan_to_num(motion)
    if SMOOTHING_WINDOW > 1:
        kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
        motion = np.convolve(motion, kernel, mode="same")
    return motion


def _detect_keep_range(
    timestamps: np.ndarray,
    motion: np.ndarray,
    sensitivity: float,
    padding_seconds: float,
) -> tuple[float, float] | None:
    threshold = sensitivity * float(np.percentile(motion, 95))
    if not threshold > 0:
        return None
    active = np.flatnonzero(motion > threshold)
    if active.size == 0:
        return None
    duration = float(timestamps[-1])
    start = max(0.0, float(timestamps[active[0]]) - padding_seconds)
    end = min(duration, float(timestamps[active[-1]]) + padding_seconds)
    return start, end


def _pick_motion_column(columns: list[str]) -> str | None:
    for candidate in MOTION_FEATURE_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


# --- Episode iteration ----------------------------------------------------------


def _iter_episode_meta_files_v3(root: Path) -> list[Path]:
    episodes_root = root / "meta" / "episodes"
    if not episodes_root.exists():
        raise HTTPException(status_code=404, detail="Missing meta/episodes/ directory")
    files = sorted(episodes_root.rglob("*.parquet"))
    if not files:
        raise HTTPException(status_code=404, detail="No episode metadata parquet files found")
    return files


def _v2_episode_data_relpath(info: dict[str, Any], episode_index: int) -> str:
    chunks_size = int(info.get("chunks_size", 1000))
    return str(info["data_path"]).format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def _v2_video_relpath(info: dict[str, Any], video_key: str, episode_index: int) -> str:
    chunks_size = int(info.get("chunks_size", 1000))
    return str(info["video_path"]).format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
        video_key=video_key,
    )


def _v2_video_keys(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feat in (info.get("features") or {}).items()
        if isinstance(feat, dict) and feat.get("dtype") == "video"
    ]


class EpisodeSlice:
    """One episode's frames: rebased timestamps + the motion source matrix."""

    def __init__(self, episode_index: int, timestamps: np.ndarray, motion_mat: np.ndarray | None):
        self.episode_index = episode_index
        self.timestamps = timestamps - (timestamps[0] if len(timestamps) else 0.0)
        self.motion_mat = motion_mat


def _iter_episode_slices(root: Path, info: dict[str, Any]) -> Iterator[EpisodeSlice]:
    """Yields per-episode timestamps + motion data for both dataset versions."""
    if _major_version(info) == 3:
        for meta_path in _iter_episode_meta_files_v3(root):
            meta_df = pd.read_parquet(meta_path)
            for _, row in meta_df.iterrows():
                rel = str(info["data_path"]).format(
                    chunk_index=int(row["data/chunk_index"]),
                    file_index=int(row["data/file_index"]),
                )
                data_path = root / rel
                if not data_path.exists():
                    raise HTTPException(status_code=404, detail=f"Missing data file: {rel}")
                ep = int(row["episode_index"])
                df = _read_data_file_cached(data_path)
                g = df[df["episode_index"] == ep]
                motion_col = _pick_motion_column(list(g.columns))
                yield EpisodeSlice(
                    ep,
                    np.asarray(g["timestamp"], dtype=np.float64),
                    _column_matrix(g[motion_col]) if motion_col else None,
                )
    else:
        total = int(info.get("total_episodes", 0))
        for ep in range(total):
            rel = _v2_episode_data_relpath(info, ep)
            data_path = root / rel
            if not data_path.exists():
                raise HTTPException(status_code=404, detail=f"Missing data file: {rel}")
            df = pd.read_parquet(data_path)
            motion_col = _pick_motion_column(list(df.columns))
            yield EpisodeSlice(
                ep,
                np.asarray(df["timestamp"], dtype=np.float64),
                _column_matrix(df[motion_col]) if motion_col else None,
            )


# Tiny cache so v3 detection doesn't re-read a shared data file per episode.
_data_file_cache: dict[str, pd.DataFrame] = {}


def _read_data_file_cached(path: Path) -> pd.DataFrame:
    key = str(path)
    if key not in _data_file_cache:
        _data_file_cache.clear()  # keep at most one file in memory
        columns = [
            c
            for c in pq.read_schema(path).names
            if c in ("episode_index", "timestamp", *MOTION_FEATURE_CANDIDATES)
        ]
        _data_file_cache[key] = pd.read_parquet(path, columns=columns)
    return _data_file_cache[key]


# --- Detect endpoint -------------------------------------------------------------


@router.post("/api/trim/detect")
def detect_trims(req: TrimDetectRequest) -> dict[str, Any]:
    root = _resolve_root(req, allow_patterns=["meta/**", "data/**"])
    info = _load_info(root)
    fps = float(info.get("fps", 30))

    episodes: list[dict[str, Any]] = []
    scanned = 0
    for ep_slice in _iter_episode_slices(root, info):
        scanned += 1
        if ep_slice.motion_mat is None or len(ep_slice.timestamps) < 3:
            continue
        motion = _motion_signal(ep_slice.motion_mat)
        if motion is None:
            continue
        keep = _detect_keep_range(
            ep_slice.timestamps, motion, req.sensitivity, req.padding_seconds
        )
        if keep is None:
            continue
        start, end = keep
        duration = float(ep_slice.timestamps[-1]) + 1.0 / fps
        cut_head = start
        cut_tail = max(0.0, duration - end)
        if cut_head < req.min_trim_seconds and cut_tail < req.min_trim_seconds:
            continue
        episodes.append(
            {
                "episode_index": ep_slice.episode_index,
                "start": round(start, 4),
                "end": round(end, 4),
                "duration": round(duration, 4),
                "cut_head": round(cut_head, 4),
                "cut_tail": round(cut_tail, 4),
            }
        )
    _data_file_cache.clear()
    return {"episodes": episodes, "scanned": scanned}


# --- Stats helpers ----------------------------------------------------------------


class _GlobalStatsAccumulator:
    """Streaming min/max/mean/std over per-episode value matrices."""

    def __init__(self) -> None:
        self._acc: dict[str, dict[str, Any]] = {}

    def add(self, feature: str, mat: np.ndarray) -> None:
        entry = self._acc.setdefault(
            feature,
            {"count": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None},
        )
        entry["count"] += mat.shape[0]
        entry["sum"] = entry["sum"] + mat.sum(axis=0)
        entry["sumsq"] = entry["sumsq"] + (mat.astype(np.float64) ** 2).sum(axis=0)
        mn = mat.min(axis=0)
        mx = mat.max(axis=0)
        entry["min"] = mn if entry["min"] is None else np.minimum(entry["min"], mn)
        entry["max"] = mx if entry["max"] is None else np.maximum(entry["max"], mx)

    def stats_for(self, feature: str) -> dict[str, np.ndarray] | None:
        entry = self._acc.get(feature)
        if not entry or entry["count"] == 0:
            return None
        n = entry["count"]
        mean = entry["sum"] / n
        var = np.maximum(entry["sumsq"] / n - mean**2, 0.0)
        return {
            "min": entry["min"],
            "max": entry["max"],
            "mean": mean,
            "std": np.sqrt(var),
            "count": np.array([n]),
        }


def _match_shape(values: np.ndarray, template: Any) -> Any:
    """Coerce recomputed stats to the shape/type of the original entry."""
    if isinstance(template, np.ndarray):
        return values.reshape(template.shape).astype(template.dtype, copy=False)
    if isinstance(template, list):
        return np.asarray(values).reshape(np.asarray(template).shape).tolist()
    if np.ndim(template) == 0:
        return float(np.asarray(values).reshape(-1)[0])
    return values


def _episode_stats(mat: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "min": mat.min(axis=0),
        "max": mat.max(axis=0),
        "mean": mat.mean(axis=0),
        "std": mat.std(axis=0),
        "count": np.array([mat.shape[0]]),
    }


def _rewrite_global_stats_json(
    src: Path,
    dst: Path,
    global_acc: _GlobalStatsAccumulator,
    warnings: list[str],
) -> None:
    stats = json.loads(src.read_text())
    untouched: list[str] = []
    for feature, feature_stats in stats.items():
        recomputed = global_acc.stats_for(feature)
        if recomputed is None:
            untouched.append(feature)
            continue
        for stat_name, template in list(feature_stats.items()):
            if stat_name in recomputed:
                feature_stats[stat_name] = _match_shape(recomputed[stat_name], template)
    dst.write_text(json.dumps(stats, indent=4))
    if untouched:
        warnings.append(
            "meta/stats.json: kept original stats for features not in the data "
            f"parquet (recompute needs frame decoding): {', '.join(sorted(untouched))}"
        )


# --- Apply: shared per-table transform ---------------------------------------------


def _parse_trims(req: TrimApplyRequest) -> dict[int, tuple[float, float]]:
    if not req.trims:
        raise HTTPException(status_code=400, detail="No trims provided")
    trims: dict[int, tuple[float, float]] = {}
    for key, rng in req.trims.items():
        try:
            ep = int(key)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Bad episode index: {key!r}") from e
        if not rng.end > rng.start:
            raise HTTPException(
                status_code=400, detail=f"Episode {ep}: end must be greater than start"
            )
        trims[ep] = (float(rng.start), float(rng.end))
    return trims


def _replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    idx = table.schema.get_field_index(name)
    field = table.schema.field(idx)
    return table.set_column(idx, field, pa.array(values).cast(field.type))


class TrimmedFileResult:
    def __init__(self) -> None:
        # episode_index → (new_length, head_cut_seconds, last_kept_rebased_ts)
        self.episode_info: dict[int, tuple[int, float, float]] = {}


def _trim_table(
    table: pa.Table,
    trims: dict[int, tuple[float, float]],
    fps: float,
    next_global_index: int,
    stats_sink: Callable[[int, str, np.ndarray], None] | None,
) -> tuple[pa.Table, int, TrimmedFileResult]:
    """Drops out-of-range rows and rebases timestamp/frame_index/index.

    Handles both a v2 single-episode file and a v3 multi-episode file. Rows
    are assumed grouped by episode in storage order (lerobot writes them
    that way). Returns the new table, the next unused global index, and
    per-episode trim info for metadata rewriting.
    """
    result = TrimmedFileResult()
    episode_col = np.asarray(table.column("episode_index").to_pandas(), dtype=np.int64)
    ts_col = np.asarray(table.column("timestamp").to_pandas(), dtype=np.float64)
    eps = 0.5 / fps

    keep_mask = np.ones(len(episode_col), dtype=bool)
    for ep in np.unique(episode_col):
        ep_rows = episode_col == ep
        ep_ts = ts_col[ep_rows]
        base = float(ep_ts[0]) if len(ep_ts) else 0.0
        rel_ts = ep_ts - base
        if int(ep) in trims:
            start, end = trims[int(ep)]
            ep_keep = (rel_ts >= start - eps) & (rel_ts <= end + eps)
            if not ep_keep.any():
                raise HTTPException(
                    status_code=400,
                    detail=f"Episode {int(ep)}: keep-range [{start}, {end}] matches no frames",
                )
            keep_mask[ep_rows] = ep_keep
            kept_rel = rel_ts[ep_keep]
            head_cut = float(kept_rel[0])
            last_kept = float(kept_rel[-1] - kept_rel[0])
        else:
            head_cut = 0.0
            last_kept = float(rel_ts[-1]) if len(rel_ts) else 0.0
        result.episode_info[int(ep)] = (0, head_cut, last_kept)  # length filled below

    new_table = table.take(np.flatnonzero(keep_mask))
    new_episode_col = episode_col[keep_mask]
    new_ts = ts_col[keep_mask]

    new_frame_index = np.zeros(len(new_episode_col), dtype=np.int64)
    for ep in np.unique(new_episode_col):
        ep_rows = new_episode_col == ep
        n = int(ep_rows.sum())
        length, head_cut, last_kept = result.episode_info[int(ep)]
        result.episode_info[int(ep)] = (n, head_cut, last_kept)
        new_frame_index[ep_rows] = np.arange(n)
        # Rebase timestamps to 0 for trimmed episodes only — untrimmed
        # episodes keep their original values bit-for-bit.
        if int(ep) in trims:
            new_ts[ep_rows] = new_ts[ep_rows] - new_ts[ep_rows][0]

    names = set(new_table.column_names)
    new_table = _replace_column(new_table, "timestamp", new_ts)
    if "frame_index" in names:
        new_table = _replace_column(new_table, "frame_index", new_frame_index)
    if "index" in names:
        new_index = np.arange(next_global_index, next_global_index + len(new_episode_col))
        new_table = _replace_column(new_table, "index", new_index)
    if "next.done" in names:
        # copy() — arrow-backed arrays can be read-only views
        done = np.asarray(new_table.column("next.done").to_pandas(), dtype=bool).copy()
        for ep in np.unique(new_episode_col):
            last_row = int(np.flatnonzero(new_episode_col == ep)[-1])
            done[last_row] = True
        new_table = _replace_column(new_table, "next.done", done)

    if stats_sink is not None:
        df = new_table.to_pandas()
        for column in new_table.column_names:
            mat = _column_matrix(df[column])
            if mat is None or not np.issubdtype(np.asarray(mat).dtype, np.number):
                continue
            for ep in np.unique(new_episode_col):
                stats_sink(int(ep), column, mat[new_episode_col == ep])

    return new_table, next_global_index + len(new_episode_col), result


# --- Apply: v3 ---------------------------------------------------------------------


def _apply_v3(
    root: Path,
    out_root: Path,
    info: dict[str, Any],
    trims: dict[int, tuple[float, float]],
    warnings: list[str],
) -> dict[str, Any]:
    fps = float(info.get("fps", 30))
    meta_files = _iter_episode_meta_files_v3(root)
    meta_dfs = [pd.read_parquet(p) for p in meta_files]
    all_meta = pd.concat(meta_dfs, ignore_index=True)
    known_eps = set(int(e) for e in all_meta["episode_index"])
    missing = sorted(set(trims) - known_eps)
    if missing:
        raise HTTPException(status_code=400, detail=f"Unknown episode indices: {missing}")

    # Data files in storage order.
    file_refs = sorted(
        {
            (int(r["data/chunk_index"]), int(r["data/file_index"]))
            for _, r in all_meta.iterrows()
        }
    )

    per_episode_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    global_acc = _GlobalStatsAccumulator()

    def stats_sink(ep: int, column: str, mat: np.ndarray) -> None:
        per_episode_stats.setdefault(ep, {})[column] = _episode_stats(mat)
        global_acc.add(column, mat)

    episode_info: dict[int, tuple[int, float, float]] = {}
    frames_before = 0
    frames_after = 0
    next_index = 0
    for chunk_index, file_index in file_refs:  # storage order
        rel = str(info["data_path"]).format(chunk_index=chunk_index, file_index=file_index)
        src = root / rel
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"Missing data file: {rel}")
        table = pq.read_table(src)
        frames_before += table.num_rows
        new_table, next_index, result = _trim_table(table, trims, fps, next_index, stats_sink)
        frames_after += new_table.num_rows
        episode_info.update(result.episode_info)
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_table, dst)

    # Rewrite episode metadata files in their original layout.
    video_keys = sorted(
        {
            c.split("/")[1]
            for c in all_meta.columns
            if c.startswith("videos/") and c.endswith("/from_timestamp")
        }
    )
    stats_columns = [c for c in all_meta.columns if c.startswith("stats/")]
    from_index = 0
    # dataset_from/to_index are cumulative over episode order — process rows
    # sorted by episode_index across all metadata files.
    new_from_to: dict[int, tuple[int, int]] = {}
    for ep in sorted(known_eps):
        length = episode_info.get(ep, (0, 0.0, 0.0))[0]
        if ep not in episode_info:
            raise HTTPException(
                status_code=500, detail=f"Episode {ep} missing from data files"
            )
        new_from_to[ep] = (from_index, from_index + length)
        from_index += length

    for meta_path, meta_df in zip(meta_files, meta_dfs):
        meta_df = meta_df.copy()
        for i, row in meta_df.iterrows():
            ep = int(row["episode_index"])
            length, head_cut, last_kept = episode_info[ep]
            meta_df.at[i, "length"] = length
            meta_df.at[i, "dataset_from_index"] = new_from_to[ep][0]
            meta_df.at[i, "dataset_to_index"] = new_from_to[ep][1]
            if ep in trims:
                for key in video_keys:
                    from_col = f"videos/{key}/from_timestamp"
                    to_col = f"videos/{key}/to_timestamp"
                    old_from = float(row[from_col])
                    meta_df.at[i, from_col] = old_from + head_cut
                    meta_df.at[i, to_col] = old_from + head_cut + last_kept + 1.0 / fps
        # Stats cells are arrays — replace whole columns instead of using
        # `.at`, which unwraps length-1 arrays into 0-d scalars.
        for col in stats_columns:
            # "stats/{feature}/{stat}" — feature names may contain "/",
            # so split from the right.
            feature, stat_name = col[len("stats/") :].rsplit("/", 1)
            if stat_name not in STAT_NAMES:
                continue
            new_cells = []
            replaced = False
            for _, row in meta_df.iterrows():
                ep_stats = per_episode_stats.get(int(row["episode_index"]), {})
                if feature in ep_stats:
                    new_cells.append(_match_shape(ep_stats[feature][stat_name], row[col]))
                    replaced = True
                else:
                    new_cells.append(row[col])
            if replaced:
                meta_df[col] = pd.Series(new_cells, index=meta_df.index, dtype=object)
        rel = meta_path.relative_to(root)
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        meta_df.to_parquet(dst, index=False)

    return {
        "frames_before": frames_before,
        "frames_after": frames_after,
        "videos_processed": 0,
        "total_frames": frames_after,
        "_global_acc": global_acc,
    }


# --- Apply: v2 ---------------------------------------------------------------------


def _cut_video_ffmpeg(src: Path, dst: Path, start: float, duration: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.6f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg failed for {src.name}: {proc.stderr[-500:]}",
        )


def _apply_v2(
    root: Path,
    out_root: Path,
    info: dict[str, Any],
    trims: dict[int, tuple[float, float]],
    warnings: list[str],
) -> dict[str, Any]:
    fps = float(info.get("fps", 30))
    total_episodes = int(info.get("total_episodes", 0))
    missing = sorted(ep for ep in trims if ep < 0 or ep >= total_episodes)
    if missing:
        raise HTTPException(status_code=400, detail=f"Unknown episode indices: {missing}")

    video_keys = _v2_video_keys(info)
    has_videos = bool(video_keys)
    if has_videos and trims and shutil.which("ffmpeg") is None:
        raise HTTPException(
            status_code=400,
            detail="ffmpeg is required to trim v2.x episode videos but was not "
            "found on the backend PATH",
        )

    per_episode_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    global_acc = _GlobalStatsAccumulator()

    def stats_sink(ep: int, column: str, mat: np.ndarray) -> None:
        per_episode_stats.setdefault(ep, {})[column] = _episode_stats(mat)
        global_acc.add(column, mat)

    episode_info: dict[int, tuple[int, float, float]] = {}
    frames_before = 0
    frames_after = 0
    videos_processed = 0
    next_index = 0
    for ep in range(total_episodes):
        rel = _v2_episode_data_relpath(info, ep)
        src = root / rel
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"Missing data file: {rel}")
        table = pq.read_table(src)
        frames_before += table.num_rows
        new_table, next_index, result = _trim_table(table, trims, fps, next_index, stats_sink)
        frames_after += new_table.num_rows
        episode_info.update(result.episode_info)
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_table, dst)

        # Videos: re-cut trimmed episodes, hardlink/copy the rest.
        _, head_cut, last_kept = episode_info[ep]
        for key in video_keys:
            video_rel = _v2_video_relpath(info, key, ep)
            video_src = root / video_rel
            if not video_src.exists():
                warnings.append(f"Missing video (skipped): {video_rel}")
                continue
            video_dst = out_root / video_rel
            video_dst.parent.mkdir(parents=True, exist_ok=True)
            if ep in trims:
                _cut_video_ffmpeg(video_src, video_dst, head_cut, last_kept + 1.0 / fps)
                videos_processed += 1
            else:
                _link_or_copy(video_src, video_dst)

    # meta/episodes.jsonl — update lengths.
    episodes_jsonl = root / "meta" / "episodes.jsonl"
    if episodes_jsonl.exists():
        lines = []
        for line in episodes_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            ep = int(entry["episode_index"])
            if ep in episode_info:
                entry["length"] = episode_info[ep][0]
            lines.append(json.dumps(entry))
        (out_root / "meta").mkdir(parents=True, exist_ok=True)
        (out_root / "meta" / "episodes.jsonl").write_text("\n".join(lines) + "\n")

    # meta/episodes_stats.jsonl (v2.1) — recompute stats for parquet-backed
    # features of trimmed episodes; image/video stats carry over.
    stats_jsonl = root / "meta" / "episodes_stats.jsonl"
    if stats_jsonl.exists():
        lines = []
        for line in stats_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            ep = int(entry["episode_index"])
            if ep in trims and ep in per_episode_stats:
                for feature, feature_stats in (entry.get("stats") or {}).items():
                    recomputed = per_episode_stats[ep].get(feature)
                    if recomputed is None:
                        continue
                    for stat_name, template in list(feature_stats.items()):
                        if stat_name in recomputed:
                            feature_stats[stat_name] = _match_shape(
                                recomputed[stat_name], template
                            )
            lines.append(json.dumps(entry))
        (out_root / "meta").mkdir(parents=True, exist_ok=True)
        (out_root / "meta" / "episodes_stats.jsonl").write_text("\n".join(lines) + "\n")

    if videos_processed:
        warnings.append(
            f"{videos_processed} video(s) re-encoded with libx264 (h264) — "
            "the original codec is not preserved for cut segments"
        )

    result = {
        "frames_before": frames_before,
        "frames_after": frames_after,
        "videos_processed": videos_processed,
        "total_frames": frames_after,
    }
    result["_global_acc"] = global_acc
    return result


# --- Apply endpoint ------------------------------------------------------------------


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _copy_tree_linked(src: Path, dst: Path) -> None:
    shutil.copytree(
        src,
        dst,
        copy_function=lambda s, d: _link_or_copy(Path(s), Path(d)),
        dirs_exist_ok=True,
    )


@router.post("/api/trim/apply")
def apply_trims(req: TrimApplyRequest) -> dict[str, Any]:
    trims = _parse_trims(req)
    root = _resolve_root(req, allow_patterns=["meta/**", "data/**", "videos/**"])
    info = _load_info(root)
    major = _major_version(info)

    if req.output_dir:
        out_root = Path(req.output_dir).expanduser().resolve()
    else:
        TRIM_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
        name = (req.repo_id or root.name or "dataset").replace("/", "__")
        out_root = TRIM_EXPORT_ROOT / f"{name}_trimmed"
    if out_root == root or root in out_root.parents:
        raise HTTPException(
            status_code=400,
            detail="output_dir must be outside the source dataset (trims are non-destructive)",
        )
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    if major == 3:
        stats = _apply_v3(root, out_root, info, trims, warnings)
        # Videos are referenced by narrowed timestamp windows — carry the
        # files over untouched (lossless).
        src_videos = root / "videos"
        if src_videos.exists():
            _copy_tree_linked(src_videos, out_root / "videos")
    else:
        stats = _apply_v2(root, out_root, info, trims, warnings)
    global_acc = stats.pop("_global_acc")

    # Copy remaining meta files (tasks, modality configs, …) that we didn't
    # rewrite, then patch info.json and recompute stats.json when present.
    # info.json is always freshly written (never hardlinked — a hardlink
    # would let the write-through mutate the source dataset).
    out_meta = out_root / "meta"
    out_meta.mkdir(parents=True, exist_ok=True)
    for src in sorted((root / "meta").rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(root)
        if rel == Path("meta/info.json"):
            continue
        dst = out_root / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.name == "stats.json":
            _rewrite_global_stats_json(src, dst, global_acc, warnings)
            continue
        _link_or_copy(src, dst)

    new_info = json.loads((root / "meta" / "info.json").read_text())
    new_info["total_frames"] = int(stats["total_frames"])
    (out_meta / "info.json").write_text(json.dumps(new_info, indent=4))

    return {
        "output_dir": str(out_root),
        "version": str(info.get("codebase_version")),
        "episodes_trimmed": len(trims),
        "frames_before": int(stats["frames_before"]),
        "frames_after": int(stats["frames_after"]),
        "videos_processed": int(stats["videos_processed"]),
        "warnings": warnings,
    }
