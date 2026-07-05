/**
 * Motion-based trim detection.
 *
 * Finds the "active" time range of an episode — the span between the first
 * and last frame where the robot actually moves — so the static head/tail
 * (operator walking to the robot, waiting for the recording to stop, …) can
 * be trimmed away.
 *
 * The motion signal is the per-frame mean of |Δvalue| across all action (or,
 * lacking those, state) dimensions, each normalized by that dimension's
 * value range within the episode. The activity threshold adapts to the
 * episode: `sensitivity × p95(motion)`, so sensor noise in the static parts
 * doesn't register as motion regardless of units.
 *
 * The same algorithm is implemented server-side in `backend/trim.py`
 * (full-resolution, all episodes) — keep the two in sync.
 */

/** Seconds to KEEP: frames with `start <= t <= end` survive the trim. */
export type TrimRange = { start: number; end: number };

export interface TrimDetectionOptions {
  /** Threshold as a fraction of the 95th-percentile motion (0–1). */
  sensitivity?: number;
  /** Seconds of context preserved before/after the detected activity. */
  paddingSeconds?: number;
  /** Moving-average window (frames) applied to the motion signal. */
  smoothingWindow?: number;
}

export const TRIM_DEFAULTS = {
  sensitivity: 0.1,
  paddingSeconds: 0.25,
  smoothingWindow: 5,
} as const;

/** Chart keys usable as a motion source: prefer actions, fall back to state. */
export function getMotionKeys(row: Record<string, number>): string[] {
  const keys = Object.keys(row).filter((k) => k !== "timestamp");
  const actionKeys = keys.filter((k) => k.startsWith("action"));
  if (actionKeys.length > 0) return actionKeys;
  return keys.filter((k) => k.includes("state"));
}

function percentile(sortedAscending: number[], p: number): number {
  if (sortedAscending.length === 0) return 0;
  const idx = Math.min(
    sortedAscending.length - 1,
    Math.max(0, Math.floor(p * (sortedAscending.length - 1))),
  );
  return sortedAscending[idx];
}

function movingAverage(values: number[], window: number): number[] {
  if (window <= 1) return values;
  const half = Math.floor(window / 2);
  const out = new Array<number>(values.length);
  for (let i = 0; i < values.length; i++) {
    let sum = 0;
    let n = 0;
    for (let j = i - half; j <= i + half; j++) {
      if (j >= 0 && j < values.length) {
        sum += values[j];
        n++;
      }
    }
    out[i] = n > 0 ? sum / n : 0;
  }
  return out;
}

export interface MotionProfile {
  /** Per-frame timestamps in seconds (episode-relative). */
  timestamps: number[];
  /** Smoothed, range-normalized per-frame motion (index-aligned). */
  motion: number[];
}

/**
 * Computes the normalized motion profile for an episode.
 * Returns null when there aren't enough frames or usable columns.
 */
export function computeMotionProfile(
  rows: Record<string, number>[],
  duration: number,
  smoothingWindow: number = TRIM_DEFAULTS.smoothingWindow,
): MotionProfile | null {
  if (rows.length < 3) return null;
  const keys = getMotionKeys(rows[0]);
  if (keys.length === 0) return null;

  // Per-dimension value range for normalization; skip near-constant dims.
  const scales: number[] = keys.map((key) => {
    let min = Infinity;
    let max = -Infinity;
    for (const row of rows) {
      const v = row[key];
      if (typeof v !== "number" || !Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    return max > min ? max - min : 0;
  });

  const motion = new Array<number>(rows.length).fill(0);
  for (let i = 1; i < rows.length; i++) {
    let sum = 0;
    let n = 0;
    for (let k = 0; k < keys.length; k++) {
      if (scales[k] <= 1e-9) continue;
      const a = rows[i - 1][keys[k]];
      const b = rows[i][keys[k]];
      if (
        typeof a !== "number" ||
        typeof b !== "number" ||
        !Number.isFinite(a) ||
        !Number.isFinite(b)
      ) {
        continue;
      }
      sum += Math.abs(b - a) / scales[k];
      n++;
    }
    motion[i] = n > 0 ? sum / n : 0;
  }

  const hasTimestamps = typeof rows[0].timestamp === "number";
  const step = rows.length > 1 ? duration / (rows.length - 1) : 0;
  const timestamps = rows.map((row, i) =>
    hasTimestamps ? row.timestamp : i * step,
  );

  return { timestamps, motion: movingAverage(motion, smoothingWindow) };
}

/**
 * Detects the active [start, end] range (seconds to keep) of an episode.
 * Returns null when no motion is detectable (all-static or degenerate data).
 */
export function detectActiveRange(
  rows: Record<string, number>[],
  duration: number,
  options: TrimDetectionOptions = {},
): TrimRange | null {
  const {
    sensitivity = TRIM_DEFAULTS.sensitivity,
    paddingSeconds = TRIM_DEFAULTS.paddingSeconds,
    smoothingWindow = TRIM_DEFAULTS.smoothingWindow,
  } = options;

  const profile = computeMotionProfile(rows, duration, smoothingWindow);
  if (!profile) return null;
  const { timestamps, motion } = profile;

  const sorted = [...motion].sort((a, b) => a - b);
  const p95 = percentile(sorted, 0.95);
  const threshold = sensitivity * p95;
  if (!(threshold > 0)) return null;

  let first = -1;
  let last = -1;
  for (let i = 0; i < motion.length; i++) {
    if (motion[i] > threshold) {
      if (first === -1) first = i;
      last = i;
    }
  }
  if (first === -1) return null;

  return {
    start: Math.max(0, timestamps[first] - paddingSeconds),
    end: Math.min(duration, timestamps[last] + paddingSeconds),
  };
}

/**
 * A trim is worth keeping when it actually removes a meaningful amount of
 * time from either end of the episode.
 */
export function isMeaningfulTrim(
  range: TrimRange,
  duration: number,
  minTrimSeconds = 0.1,
): boolean {
  return (
    range.start >= minTrimSeconds || range.end <= duration - minTrimSeconds
  );
}
