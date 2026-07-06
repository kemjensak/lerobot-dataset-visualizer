"""Image-feature stats recomputation by sampling video frames with ffmpeg.

LeRobot stores per-camera stats as per-channel min/max/mean/std with shape
(3, 1, 1), normalized to [0, 1], computed from a *sample* of frames (not
every frame). This module mirrors that: for each episode/camera it decodes
up to ``num_samples`` evenly-spaced frames (downscaled for speed, matching
lerobot's own downsampling approximation) through a single ffmpeg
invocation, and produces both per-episode stats and a dataset-level
aggregate for ``meta/stats.json``.

Used by ``trim.py`` and ``edits.py`` when a request sets
``recompute_image_stats=true``. Requires ffmpeg + ffprobe on PATH.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException

logger = logging.getLogger("lerobot-videostats")

SCALE_HEIGHT = 128  # decode height; width keeps aspect (even-rounded)


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise HTTPException(
            status_code=400,
            detail="recompute_image_stats requires ffmpeg + ffprobe on the backend PATH",
        )


def _probe_dims(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise HTTPException(
            status_code=500, detail=f"ffprobe failed for {path.name}: {out.stderr[-300:]}"
        )
    w, h = out.stdout.strip().split(",")[:2]
    return int(w), int(h)


class EpisodeImageStats:
    """Per-episode per-channel stats plus raw accumulators for aggregation."""

    def __init__(self, frames: np.ndarray):
        # frames: (n, h, w, 3) uint8 → normalized channel-first stats
        norm = frames.astype(np.float64) / 255.0
        pixels = norm.reshape(-1, 3)  # every sampled pixel, per channel
        self.count = frames.shape[0]
        self.min = pixels.min(axis=0)
        self.max = pixels.max(axis=0)
        self.sum = pixels.sum(axis=0)
        self.sumsq = (pixels**2).sum(axis=0)
        self.n_pixels = pixels.shape[0]

    def stats(self) -> dict[str, np.ndarray]:
        mean = self.sum / self.n_pixels
        var = np.maximum(self.sumsq / self.n_pixels - mean**2, 0.0)
        return {
            "min": self.min.reshape(3, 1, 1),
            "max": self.max.reshape(3, 1, 1),
            "mean": mean.reshape(3, 1, 1),
            "std": np.sqrt(var).reshape(3, 1, 1),
            "count": np.array([self.count]),
        }


def sample_episode_frames(
    video_path: Path,
    start: float | None,
    end: float | None,
    num_samples: int,
) -> EpisodeImageStats | None:
    """Decodes up to ``num_samples`` evenly-spaced frames from
    ``video_path`` (optionally restricted to the [start, end) window) in one
    ffmpeg call. Returns None when nothing could be decoded."""
    w, h = _probe_dims(video_path)
    sh = SCALE_HEIGHT
    sw = max(2, int(round(w * sh / h / 2)) * 2)

    cmd = ["ffmpeg", "-v", "error"]
    duration = None
    if start is not None:
        cmd += ["-ss", f"{start:.6f}"]
    if end is not None:
        duration = max(0.05, end - (start or 0.0))
        cmd += ["-t", f"{duration:.6f}"]
    cmd += ["-i", str(video_path)]
    if duration is not None:
        sample_fps = num_samples / duration
        vf = f"fps={sample_fps:.6f},scale={sw}:{sh}"
    else:
        # Whole file: probe duration for the fps filter.
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
            ],
            capture_output=True,
            text=True,
        )
        try:
            total = max(0.05, float(probe.stdout.strip()))
        except ValueError:
            total = 1.0
        vf = f"fps={num_samples / total:.6f},scale={sw}:{sh}"
    cmd += [
        "-vf", vf,
        "-frames:v", str(num_samples),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        logger.warning("ffmpeg sampling failed for %s: %s", video_path, proc.stderr[-300:])
        return None
    frame_bytes = sw * sh * 3
    n = len(proc.stdout) // frame_bytes
    if n == 0:
        return None
    frames = np.frombuffer(proc.stdout[: n * frame_bytes], dtype=np.uint8).reshape(
        n, sh, sw, 3
    )
    return EpisodeImageStats(frames)


class ImageStatsAggregator:
    """Combines per-episode accumulators into dataset-level stats."""

    def __init__(self) -> None:
        self._by_feature: dict[str, list[EpisodeImageStats]] = {}

    def add(self, feature: str, ep_stats: EpisodeImageStats) -> None:
        self._by_feature.setdefault(feature, []).append(ep_stats)

    def global_stats(self) -> dict[str, dict[str, np.ndarray]]:
        out: dict[str, dict[str, np.ndarray]] = {}
        for feature, entries in self._by_feature.items():
            n_pixels = sum(e.n_pixels for e in entries)
            if n_pixels == 0:
                continue
            total_sum = np.sum([e.sum for e in entries], axis=0)
            total_sumsq = np.sum([e.sumsq for e in entries], axis=0)
            mean = total_sum / n_pixels
            var = np.maximum(total_sumsq / n_pixels - mean**2, 0.0)
            out[feature] = {
                "min": np.min([e.min for e in entries], axis=0).reshape(3, 1, 1),
                "max": np.max([e.max for e in entries], axis=0).reshape(3, 1, 1),
                "mean": mean.reshape(3, 1, 1),
                "std": np.sqrt(var).reshape(3, 1, 1),
                "count": np.array([sum(e.count for e in entries)]),
            }
        return out


def _video_feature_keys(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feat in (info.get("features") or {}).items()
        if isinstance(feat, dict) and feat.get("dtype") == "video"
    ]


def recompute_image_stats(
    out_root: Path,
    info: dict[str, Any],
    major_version: int,
    num_samples: int,
    warnings: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Samples the (already-written) output dataset's videos, patches the
    per-episode stats entries in ``out_root``, and returns the dataset-level
    image stats for the caller to fold into ``meta/stats.json``.

    Must run after the output ``videos/`` tree and per-episode stats files
    exist. Stats are sample-based (like lerobot's own image stats) and
    computed on downscaled frames — a close approximation, not a bit-exact
    full-resolution pass.
    """
    ensure_ffmpeg_available()
    # Local import to avoid a circular module dependency (trim imports us).
    from trim import _iter_episode_meta_files_v3, _match_shape, _v2_video_relpath

    video_keys = _video_feature_keys(info)
    if not video_keys:
        return {}
    agg = ImageStatsAggregator()
    per_ep: dict[int, dict[str, dict[str, np.ndarray]]] = {}

    if major_version == 3:
        meta_files = _iter_episode_meta_files_v3(out_root)
        for meta_path in meta_files:
            meta_df = pd.read_parquet(meta_path)
            for _, row in meta_df.iterrows():
                ep = int(row["episode_index"])
                for key in video_keys:
                    try:
                        rel = str(info["video_path"]).format(
                            video_key=key,
                            chunk_index=int(row[f"videos/{key}/chunk_index"]),
                            file_index=int(row[f"videos/{key}/file_index"]),
                        )
                        start = float(row[f"videos/{key}/from_timestamp"])
                        end = float(row[f"videos/{key}/to_timestamp"])
                    except KeyError:
                        continue
                    video_path = out_root / rel
                    if not video_path.exists():
                        warnings.append(f"Missing video (image stats skipped): {rel}")
                        continue
                    ep_stats = sample_episode_frames(video_path, start, end, num_samples)
                    if ep_stats is None:
                        warnings.append(f"Could not decode {rel} for episode {ep}")
                        continue
                    per_ep.setdefault(ep, {})[key] = ep_stats.stats()
                    agg.add(key, ep_stats)

            # Patch this metadata file's image stats columns in place.
            changed = False
            for key in video_keys:
                for stat_name in ("min", "max", "mean", "std", "count"):
                    col = f"stats/{key}/{stat_name}"
                    if col not in meta_df.columns:
                        continue
                    new_cells = []
                    for _, row in meta_df.iterrows():
                        ep = int(row["episode_index"])
                        stats = per_ep.get(ep, {}).get(key)
                        if stats is not None:
                            new_cells.append(_match_shape(stats[stat_name], row[col]))
                        else:
                            new_cells.append(row[col])
                    meta_df[col] = pd.Series(new_cells, index=meta_df.index, dtype=object)
                    changed = True
            if changed:
                meta_df.to_parquet(meta_path, index=False)
    else:
        total_episodes = int(info.get("total_episodes", 0))
        for ep in range(total_episodes):
            for key in video_keys:
                rel = _v2_video_relpath(info, key, ep)
                video_path = out_root / rel
                if not video_path.exists():
                    warnings.append(f"Missing video (image stats skipped): {rel}")
                    continue
                # v2 output videos are already per-episode (and already cut
                # by trim), so sample the whole file.
                ep_stats = sample_episode_frames(video_path, None, None, num_samples)
                if ep_stats is None:
                    warnings.append(f"Could not decode {rel}")
                    continue
                per_ep.setdefault(ep, {})[key] = ep_stats.stats()
                agg.add(key, ep_stats)

        stats_jsonl = out_root / "meta" / "episodes_stats.jsonl"
        if stats_jsonl.exists():
            lines = []
            for line in stats_jsonl.read_text().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                ep = int(entry["episode_index"])
                for key in video_keys:
                    stats = per_ep.get(ep, {}).get(key)
                    feature_stats = (entry.get("stats") or {}).get(key)
                    if stats is None or feature_stats is None:
                        continue
                    for stat_name, template in list(feature_stats.items()):
                        if stat_name in stats:
                            feature_stats[stat_name] = _match_shape(
                                stats[stat_name], template
                            )
                lines.append(json.dumps(entry))
            stats_jsonl.write_text("\n".join(lines) + "\n")

    sampled = sum(len(v) for v in per_ep.values())
    warnings.append(
        f"Image stats recomputed from sampled frames ({num_samples}/episode, "
        f"downscaled) for {sampled} episode-camera pairs — a close "
        "approximation of a full-resolution pass"
    )
    return agg.global_stats()
