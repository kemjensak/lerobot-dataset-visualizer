/**
 * Numeric value edits.
 *
 * An edit rewrites one dimension (or all dimensions) of a numeric feature
 * over a frame range of an episode — e.g. "set observation.state[odom_x]
 * to 1.0 for the second half of the episode". Edits are previewed
 * client-side and applied server-side by `backend/edits.py`, which also
 * recomputes the affected per-episode and global stats.
 *
 * The frame-mask semantics here MUST match `backend/edits.py`
 * (`_range_mask`) — keep the two in sync:
 * - fraction mode: frame i (of n) is edited when
 *   `round(start·n) <= i < round(end·n)`
 * - seconds mode: episode-relative timestamp t is edited when
 *   `start − ε <= t <= end + ε` (ε = half a frame)
 */

import { CHART_CONFIG } from "@/utils/constants";

export type EditOpKind = "set" | "add" | "scale";

export type EditRange = {
  mode: "fraction" | "seconds";
  start: number;
  end: number;
};

export interface NumericEdit {
  id: string;
  /** Parquet column name, e.g. "observation.state". */
  feature: string;
  /** Dimension index within the feature; null = all dimensions. */
  dim: number | null;
  /** Display label for the dimension (e.g. "odom_x"); null = all dims. */
  dimLabel: string | null;
  /** Episode to edit; null = every episode in the dataset. */
  episodeIndex: number | null;
  /** Frame range to edit; null = the whole episode. */
  range: EditRange | null;
  op: EditOpKind;
  value: number;
}

export interface FeatureDim {
  /** Full chart key, e.g. "observation.state | odom_x". */
  key: string;
  /** Display label (chart-key suffix or the full key for scalars). */
  label: string;
  /** Dimension index within the feature (position in key order). */
  index: number;
}

export interface FeatureGroup {
  feature: string;
  dims: FeatureDim[];
}

/**
 * Groups a chart row's series keys by feature, preserving dimension order.
 * Chart keys are `"{feature} | {name-or-index}"` generated in `names[]`
 * order, so a key's position within its feature group IS its dim index.
 */
export function featureGroupsFromRow(
  row: Record<string, number>,
): FeatureGroup[] {
  const delim = CHART_CONFIG.SERIES_NAME_DELIMITER;
  const groups = new Map<string, FeatureDim[]>();
  for (const key of Object.keys(row)) {
    if (key === "timestamp") continue;
    const sep = key.indexOf(delim);
    const feature = sep === -1 ? key : key.slice(0, sep);
    const label = sep === -1 ? key : key.slice(sep + delim.length);
    const dims = groups.get(feature) ?? [];
    dims.push({ key, label, index: dims.length });
    groups.set(feature, dims);
  }
  return [...groups.entries()].map(([feature, dims]) => ({ feature, dims }));
}

/** Frame mask for an edit range. Mirrors `_range_mask` in backend/edits.py. */
export function editFrameMask(
  timestamps: number[],
  duration: number,
  range: EditRange | null,
): boolean[] {
  const n = timestamps.length;
  if (!range) return new Array<boolean>(n).fill(true);
  if (range.mode === "fraction") {
    const lo = Math.round(Math.max(0, range.start) * n);
    const hi = Math.round(Math.min(1, range.end) * n);
    return timestamps.map((_, i) => i >= lo && i < hi);
  }
  const eps = n > 1 ? duration / n / 2 : 0;
  const base = timestamps[0] ?? 0;
  return timestamps.map((t) => {
    const rel = t - base;
    return rel >= range.start - eps && rel <= range.end + eps;
  });
}

/**
 * Applies an edit to one dimension's series (client-side preview).
 * Returns a new array; the input is not mutated.
 */
export function applyEditToSeries(
  values: number[],
  timestamps: number[],
  duration: number,
  edit: Pick<NumericEdit, "op" | "value" | "range">,
): number[] {
  const mask = editFrameMask(timestamps, duration, edit.range);
  return values.map((v, i) => {
    if (!mask[i]) return v;
    switch (edit.op) {
      case "set":
        return edit.value;
      case "add":
        return v + edit.value;
      case "scale":
        return v * edit.value;
    }
  });
}

/** Human-readable one-line description of an edit (for the pending list). */
export function describeEdit(edit: NumericEdit): string {
  const target =
    edit.dim === null
      ? `${edit.feature} (all dims)`
      : `${edit.feature}[${edit.dimLabel ?? edit.dim}]`;
  const opText =
    edit.op === "set"
      ? `= ${edit.value}`
      : edit.op === "add"
        ? `+= ${edit.value}`
        : `×= ${edit.value}`;
  const rangeText = !edit.range
    ? "whole episode"
    : edit.range.mode === "fraction"
      ? `${Math.round(edit.range.start * 100)}%–${Math.round(edit.range.end * 100)}%`
      : `${edit.range.start.toFixed(2)}s–${edit.range.end.toFixed(2)}s`;
  const scope =
    edit.episodeIndex === null ? "all eps" : `ep ${edit.episodeIndex}`;
  return `${target} ${opText} · ${rangeText} · ${scope}`;
}
