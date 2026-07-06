"""Numeric-edit endpoint for the LeRobot dataset visualizer backend.

``POST /api/edit/apply`` rewrites numeric feature values over a frame range
and writes a *modified copy* of the dataset (the source is never touched) —
e.g. "set observation.state[3] to 1.0 for the second half of every episode".

Per-episode stats (v3 ``stats/*`` episode-metadata columns, v2.1
``meta/episodes_stats.jsonl``) are recomputed for the touched episodes, and
``meta/stats.json`` (when present) is recomputed for every feature stored in
the data parquet. Row counts, timestamps, indices and videos are unchanged —
videos are carried over as hardlinks.

Frame-mask semantics mirror ``src/utils/numericEdit.ts`` (``editFrameMask``)
— keep the two in sync:
- fraction mode: frame ``i`` of ``n`` is edited when
  ``round(start*n) <= i < round(end*n)``
- seconds mode: episode-relative timestamp ``t`` is edited when
  ``start - eps <= t <= end + eps`` (eps = half a frame)
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from trim import (
    STAT_NAMES,
    TRIM_EXPORT_ROOT,
    TrimDatasetRef,
    _column_matrix,
    _copy_tree_linked,
    _episode_stats,
    _GlobalStatsAccumulator,
    _iter_episode_meta_files_v3,
    _link_or_copy,
    _load_info,
    _major_version,
    _match_shape,
    _resolve_root,
    _rewrite_global_stats_json,
    _v2_episode_data_relpath,
)

logger = logging.getLogger("lerobot-edit")

router = APIRouter()


# --- Requests -----------------------------------------------------------------


class EditRangeModel(BaseModel):
    mode: Literal["fraction", "seconds"]
    start: float
    end: float


class NumericEditModel(BaseModel):
    feature: str
    dim: int | None = None
    episode_index: int | None = None
    range: EditRangeModel | None = None
    op: Literal["set", "add", "scale"]
    value: float


class EditApplyRequest(TrimDatasetRef):
    edits: list[NumericEditModel]
    output_dir: str | None = None


# --- Core ----------------------------------------------------------------------


def _range_mask(
    ts: np.ndarray, fps: float, rng: EditRangeModel | None
) -> np.ndarray:
    n = len(ts)
    if rng is None:
        return np.ones(n, dtype=bool)
    if rng.mode == "fraction":
        lo = int(round(max(0.0, rng.start) * n))
        hi = int(round(min(1.0, rng.end) * n))
        mask = np.zeros(n, dtype=bool)
        mask[max(0, lo) : max(0, hi)] = True
        return mask
    eps = 0.5 / fps
    rel = ts - (ts[0] if n else 0.0)
    return (rel >= rng.start - eps) & (rel <= rng.end + eps)


def _rebuild_feature_column(table: pa.Table, name: str, mat: np.ndarray) -> pa.Table:
    """Puts an edited (n, D) matrix back into the table, preserving the
    column's original arrow type (fixed_size_list / list / scalar)."""
    idx = table.schema.get_field_index(name)
    field = table.schema.field(idx)
    t = field.type
    if pa.types.is_fixed_size_list(t):
        flat = pa.array(mat.reshape(-1)).cast(t.value_type)
        arr = pa.FixedSizeListArray.from_arrays(flat, t.list_size)
    elif pa.types.is_list(t) or pa.types.is_large_list(t):
        arr = pa.array([row for row in mat]).cast(t)
    else:
        arr = pa.array(mat.reshape(-1)).cast(t)
    return table.set_column(idx, field, arr)


class _EditStats:
    def __init__(self) -> None:
        self.values_changed = 0
        self.touched_eps: set[int] = set()
        self.per_episode_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {}
        self.global_acc = _GlobalStatsAccumulator()


def _apply_edits_to_table(
    table: pa.Table,
    edits: list[NumericEditModel],
    fps: float,
    acc: _EditStats,
) -> pa.Table:
    """Applies all matching edits to one data file, then feeds the (post-edit)
    per-episode stats into the accumulator. Assumes rows are grouped by
    episode in storage order (lerobot writes them that way)."""
    episode_col = np.asarray(table.column("episode_index").to_pandas(), dtype=np.int64)
    ts_col = np.asarray(table.column("timestamp").to_pandas(), dtype=np.float64)
    df = table.to_pandas()

    features = sorted({e.feature for e in edits})
    for feature in features:
        if feature not in table.column_names:
            raise HTTPException(
                status_code=400,
                detail=f"Feature {feature!r} not found in the data parquet "
                f"(available: {', '.join(sorted(table.column_names))})",
            )

    # Full-file matrix per edited feature; mutated in place, written back once.
    mats: dict[str, np.ndarray] = {}
    for feature in features:
        mat = _column_matrix(df[feature])
        if mat is None:
            raise HTTPException(
                status_code=400, detail=f"Feature {feature!r} is not numeric"
            )
        mats[feature] = mat if mat.ndim == 2 else mat[:, None]

    for ep in np.unique(episode_col):
        ep_rows = np.flatnonzero(episode_col == ep)
        ep_ts = ts_col[ep_rows]
        for edit in edits:
            if edit.episode_index is not None and edit.episode_index != int(ep):
                continue
            mat = mats[edit.feature]
            n_dims = mat.shape[1]
            if edit.dim is not None and not (0 <= edit.dim < n_dims):
                raise HTTPException(
                    status_code=400,
                    detail=f"{edit.feature}: dim {edit.dim} out of range (D={n_dims})",
                )
            mask = _range_mask(ep_ts, fps, edit.range)
            if not mask.any():
                continue
            rows = ep_rows[mask]
            dims = [edit.dim] if edit.dim is not None else list(range(n_dims))
            for d in dims:
                if edit.op == "set":
                    mat[rows, d] = edit.value
                elif edit.op == "add":
                    mat[rows, d] += edit.value
                else:  # scale
                    mat[rows, d] *= edit.value
            acc.values_changed += len(rows) * len(dims)
            acc.touched_eps.add(int(ep))

    for feature, mat in mats.items():
        table = _rebuild_feature_column(table, feature, mat)
        df[feature] = list(mat)  # keep the stats source in sync

    # Post-edit stats for every episode in the file (global stats need all
    # rows; per-episode entries for untouched episodes are simply unchanged).
    for column in table.column_names:
        col_mat = mats.get(column)
        if col_mat is None:
            col_mat = _column_matrix(df[column])
        if col_mat is None:
            continue
        if col_mat.ndim == 1:
            col_mat = col_mat[:, None]
        for ep in np.unique(episode_col):
            ep_mat = col_mat[episode_col == ep]
            acc.per_episode_stats.setdefault(int(ep), {})[column] = _episode_stats(ep_mat)
            acc.global_acc.add(column, ep_mat)

    return table


# --- Endpoint --------------------------------------------------------------------


@router.post("/api/edit/apply")
def apply_edits(req: EditApplyRequest) -> dict[str, Any]:
    if not req.edits:
        raise HTTPException(status_code=400, detail="No edits provided")
    for edit in req.edits:
        if not np.isfinite(edit.value):
            raise HTTPException(status_code=400, detail="Edit value must be finite")
        if edit.range is not None and not edit.range.end > edit.range.start:
            raise HTTPException(
                status_code=400, detail="Edit range end must be greater than start"
            )

    root = _resolve_root(req, allow_patterns=["meta/**", "data/**", "videos/**"])
    info = _load_info(root)
    major = _major_version(info)
    fps = float(info.get("fps", 30))

    if req.output_dir:
        out_root = Path(req.output_dir).expanduser().resolve()
    else:
        TRIM_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
        name = (req.repo_id or root.name or "dataset").replace("/", "__")
        out_root = TRIM_EXPORT_ROOT / f"{name}_edited"
    if out_root == root or root in out_root.parents:
        raise HTTPException(
            status_code=400,
            detail="output_dir must be outside the source dataset (edits are non-destructive)",
        )
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    acc = _EditStats()

    if major == 3:
        meta_files = _iter_episode_meta_files_v3(root)
        meta_dfs = [pd.read_parquet(p) for p in meta_files]
        all_meta = pd.concat(meta_dfs, ignore_index=True)
        known_eps = {int(e) for e in all_meta["episode_index"]}
        _validate_episodes(req.edits, known_eps)

        file_refs = sorted(
            {
                (int(r["data/chunk_index"]), int(r["data/file_index"]))
                for _, r in all_meta.iterrows()
            }
        )
        for chunk_index, file_index in file_refs:
            rel = str(info["data_path"]).format(
                chunk_index=chunk_index, file_index=file_index
            )
            src = root / rel
            if not src.exists():
                raise HTTPException(status_code=404, detail=f"Missing data file: {rel}")
            table = _apply_edits_to_table(pq.read_table(src), req.edits, fps, acc)
            dst = out_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dst)

        # Episode metadata: only the stats columns of touched episodes change.
        stats_columns = [c for c in all_meta.columns if c.startswith("stats/")]
        for meta_path, meta_df in zip(meta_files, meta_dfs):
            meta_df = meta_df.copy()
            for col in stats_columns:
                feature, stat_name = col[len("stats/") :].rsplit("/", 1)
                if stat_name not in STAT_NAMES:
                    continue
                new_cells = []
                replaced = False
                for _, row in meta_df.iterrows():
                    ep = int(row["episode_index"])
                    ep_stats = acc.per_episode_stats.get(ep, {})
                    if ep in acc.touched_eps and feature in ep_stats:
                        new_cells.append(
                            _match_shape(ep_stats[feature][stat_name], row[col])
                        )
                        replaced = True
                    else:
                        new_cells.append(row[col])
                if replaced:
                    meta_df[col] = pd.Series(new_cells, index=meta_df.index, dtype=object)
            rel = meta_path.relative_to(root)
            dst = out_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            meta_df.to_parquet(dst, index=False)
    else:
        total_episodes = int(info.get("total_episodes", 0))
        _validate_episodes(req.edits, set(range(total_episodes)))
        for ep in range(total_episodes):
            rel = _v2_episode_data_relpath(info, ep)
            src = root / rel
            if not src.exists():
                raise HTTPException(status_code=404, detail=f"Missing data file: {rel}")
            table = _apply_edits_to_table(pq.read_table(src), req.edits, fps, acc)
            dst = out_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dst)

        # meta/episodes_stats.jsonl (v2.1) — recompute touched episodes.
        stats_jsonl = root / "meta" / "episodes_stats.jsonl"
        if stats_jsonl.exists():
            lines = []
            for line in stats_jsonl.read_text().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                ep = int(entry["episode_index"])
                if ep in acc.touched_eps and ep in acc.per_episode_stats:
                    for feature, feature_stats in (entry.get("stats") or {}).items():
                        recomputed = acc.per_episode_stats[ep].get(feature)
                        if recomputed is None:
                            continue
                        for stat_name, template in list(feature_stats.items()):
                            if stat_name in recomputed:
                                feature_stats[stat_name] = _match_shape(
                                    recomputed[stat_name], template
                                )
                lines.append(json.dumps(entry))
            (out_root / "meta").mkdir(parents=True, exist_ok=True)
            (out_root / "meta" / "episodes_stats.jsonl").write_text(
                "\n".join(lines) + "\n"
            )

    # Videos are untouched by numeric edits — hardlink the whole tree.
    src_videos = root / "videos"
    if src_videos.exists():
        _copy_tree_linked(src_videos, out_root / "videos")

    # Remaining meta files: recompute stats.json, carry the rest over.
    # info.json is written fresh (never hardlinked) even though its content
    # is unchanged, to keep the copy independent of the source inode.
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
            _rewrite_global_stats_json(src, dst, acc.global_acc, warnings)
            continue
        _link_or_copy(src, dst)
    (out_meta / "info.json").write_text(
        json.dumps(json.loads((root / "meta" / "info.json").read_text()), indent=4)
    )

    return {
        "output_dir": str(out_root),
        "version": str(info.get("codebase_version")),
        "edits_applied": len(req.edits),
        "episodes_touched": len(acc.touched_eps),
        "values_changed": int(acc.values_changed),
        "warnings": warnings,
    }


def _validate_episodes(edits: list[NumericEditModel], known_eps: set[int]) -> None:
    missing = sorted(
        {
            e.episode_index
            for e in edits
            if e.episode_index is not None and e.episode_index not in known_eps
        }
    )
    if missing:
        raise HTTPException(status_code=400, detail=f"Unknown episode indices: {missing}")
