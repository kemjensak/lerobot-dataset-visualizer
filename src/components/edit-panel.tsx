"use client";

import React, { useState, useMemo, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useNumericEdits } from "@/context/edit-context";
import {
  applyEditToSeries,
  describeEdit,
  editFrameMask,
  featureGroupsFromRow,
  type EditOpKind,
  type EditRange,
} from "@/utils/numericEdit";
import { isAnnotateBackendEnabled } from "@/utils/annotationsClient";
import {
  applyEditsBackend,
  editsToJson,
  type ApplyEditsResult,
} from "@/utils/editClient";

const RANGE_THUMB_CLASS =
  "absolute inset-0 w-full appearance-none bg-transparent pointer-events-none [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-cyan-400 [&::-webkit-slider-thumb]:cursor-pointer [&::-moz-range-thumb]:pointer-events-auto [&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:w-3.5 [&::-moz-range-thumb]:h-3.5 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-white [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-cyan-400 [&::-moz-range-thumb]:cursor-pointer";

// Shared with the Trim panel so the dataset path is entered once.
const LOCAL_PATH_STORAGE_KEY = "trim-local-path";
const OUTPUT_DIR_STORAGE_KEY = "edit-output-dir";

const SELECT_CLASS =
  "bg-[var(--bg)]/50 border border-white/10 rounded px-1.5 py-1 text-slate-200 text-xs";

// ─── Before/after preview sparkline ──────────────────────────────

function EditPreview({
  original,
  edited,
  mask,
}: {
  original: number[];
  edited: number[];
  mask: boolean[];
}) {
  const W = 600;
  const H = 64;
  const all = [...original, ...edited].filter((v) => Number.isFinite(v));
  let min = Math.min(...all);
  let max = Math.max(...all);
  if (!(max > min)) {
    min -= 1;
    max += 1;
  }
  const toPoints = (vals: number[]) =>
    vals
      .map((v, i) => {
        const x = (i / Math.max(1, vals.length - 1)) * W;
        const y = H - 4 - ((v - min) / (max - min)) * (H - 8);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  const first = mask.indexOf(true);
  const last = mask.lastIndexOf(true);
  const n = Math.max(1, mask.length - 1);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      className="w-full h-16 rounded bg-[var(--bg)]/60 border border-white/5"
    >
      {first !== -1 && (
        <rect
          x={(first / n) * W}
          y={0}
          width={((last - first) / n) * W}
          height={H}
          fill="rgba(56,189,248,0.08)"
        />
      )}
      <polyline
        points={toPoints(original)}
        fill="none"
        stroke="rgba(148,163,184,0.55)"
        strokeWidth={1}
        vectorEffect="non-scaling-stroke"
      />
      <polyline
        points={toPoints(edited)}
        fill="none"
        stroke="#38bdf8"
        strokeWidth={1.4}
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

// ─── Main panel ──────────────────────────────────────────────────

interface EditPanelProps {
  repoId: string;
  episodeId: number;
  duration: number;
  fps: number;
  flatChartData: Record<string, number>[];
  /** All episode indices in the dataset (for the "another episode" target). */
  episodes: number[];
}

function EditPanel({
  repoId,
  episodeId,
  duration,
  fps,
  flatChartData,
  episodes,
}: EditPanelProps) {
  const router = useRouter();
  const { edits, count, add, remove, clear } = useNumericEdits();
  const backendEnabled = isAnnotateBackendEnabled();

  const groups = useMemo(
    () => (flatChartData.length ? featureGroupsFromRow(flatChartData[0]) : []),
    [flatChartData],
  );

  // ── Edit builder state ──
  const [featureIdx, setFeatureIdx] = useState(0);
  const group = groups[Math.min(featureIdx, Math.max(0, groups.length - 1))];
  const [dimIdx, setDimIdx] = useState<number | "all">(0);
  useEffect(() => setDimIdx(0), [featureIdx]);
  const [scope, setScope] = useState<"episode" | "other" | "all">("episode");
  const [otherEpisode, setOtherEpisode] = useState<number>(episodeId);
  const otherEpisodeValid =
    scope !== "other" ||
    (Number.isInteger(otherEpisode) && episodes.includes(otherEpisode));
  const [useRange, setUseRange] = useState(true);
  const [rangeMode, setRangeMode] = useState<"fraction" | "seconds">(
    "fraction",
  );
  const [rangeStart, setRangeStart] = useState(0.5);
  const [rangeEnd, setRangeEnd] = useState(1.0);
  const [op, setOp] = useState<EditOpKind>("set");
  const [value, setValue] = useState(1.0);

  // Keep the range in bounds when toggling between % and seconds.
  const rangeMax = rangeMode === "fraction" ? 1 : duration;
  const switchMode = (mode: "fraction" | "seconds") => {
    if (mode === rangeMode) return;
    if (mode === "seconds") {
      setRangeStart(rangeStart * duration);
      setRangeEnd(rangeEnd * duration);
    } else {
      setRangeStart(duration > 0 ? rangeStart / duration : 0);
      setRangeEnd(duration > 0 ? rangeEnd / duration : 1);
    }
    setRangeMode(mode);
  };

  const range: EditRange | null = useRange
    ? { mode: rangeMode, start: rangeStart, end: rangeEnd }
    : null;

  // ── Preview (first selected dim of the current episode) ──
  const previewDim =
    group && group.dims.length > 0
      ? group.dims[
          dimIdx === "all" ? 0 : Math.min(dimIdx, group.dims.length - 1)
        ]
      : null;
  const preview = useMemo(() => {
    if (!previewDim) return null;
    const timestamps = flatChartData.map((r, i) =>
      typeof r.timestamp === "number"
        ? r.timestamp
        : (i / Math.max(1, flatChartData.length - 1)) * duration,
    );
    const original = flatChartData.map((r) => r[previewDim.key]);
    const edited = applyEditToSeries(original, timestamps, duration, {
      op,
      value,
      range,
    });
    const mask = editFrameMask(timestamps, duration, range);
    return { original, edited, mask };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    flatChartData,
    duration,
    previewDim?.key,
    op,
    value,
    useRange,
    rangeMode,
    rangeStart,
    rangeEnd,
  ]);

  const handleAdd = useCallback(() => {
    if (!group || !otherEpisodeValid) return;
    add({
      feature: group.feature,
      dim: dimIdx === "all" ? null : group.dims[dimIdx].index,
      dimLabel: dimIdx === "all" ? null : group.dims[dimIdx].label,
      episodeIndex:
        scope === "episode"
          ? episodeId
          : scope === "other"
            ? otherEpisode
            : null,
      range,
      op,
      value,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    group,
    dimIdx,
    scope,
    episodeId,
    otherEpisode,
    otherEpisodeValid,
    op,
    value,
    useRange,
    rangeMode,
    rangeStart,
    rangeEnd,
    add,
  ]);

  // ── Backend state ──
  const [localPath, setLocalPath] = useState("");
  const [outputDir, setOutputDir] = useState("");
  useEffect(() => {
    try {
      setLocalPath(localStorage.getItem(LOCAL_PATH_STORAGE_KEY) ?? "");
      setOutputDir(localStorage.getItem(OUTPUT_DIR_STORAGE_KEY) ?? "");
    } catch {
      /* ignore */
    }
  }, []);
  const persistInput = (key: string, val: string) => {
    try {
      localStorage.setItem(key, val);
    } catch {
      /* ignore */
    }
  };

  const [applying, setApplying] = useState(false);
  const [recomputeImageStats, setRecomputeImageStats] = useState(false);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [applyResult, setApplyResult] = useState<ApplyEditsResult | null>(null);
  const [copied, setCopied] = useState(false);

  const handleApply = useCallback(async () => {
    setApplying(true);
    setBackendError(null);
    setApplyResult(null);
    try {
      const result = await applyEditsBackend(
        { repoId: localPath ? null : repoId, localPath: localPath || null },
        edits,
        { outputDir: outputDir || null, recomputeImageStats },
      );
      setApplyResult(result);
    } catch (err) {
      setBackendError(err instanceof Error ? err.message : String(err));
    } finally {
      setApplying(false);
    }
  }, [repoId, localPath, outputDir, edits, recomputeImageStats]);

  const exportJson = useMemo(() => editsToJson(repoId, edits), [repoId, edits]);
  const handleCopyJson = useCallback(() => {
    navigator.clipboard.writeText(exportJson);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }, [exportJson]);

  const fmtRange = (v: number) =>
    rangeMode === "fraction" ? `${Math.round(v * 100)}%` : `${v.toFixed(2)}s`;

  return (
    <div className="max-w-5xl mx-auto py-6 space-y-8">
      <div>
        <h2 className="text-xl font-bold text-slate-100">Edit Values</h2>
        <p className="text-sm text-slate-400 mt-1">
          Rewrite numeric feature values over a frame range — e.g. fix an
          odometry dimension to 1.0 for the second half of an episode. Applying
          produces a modified copy of the dataset with per-episode and global
          stats recomputed automatically.
        </p>
      </div>

      {/* ── Builder ── */}
      <div className="bg-[var(--surface-1)]/60 rounded-lg p-5 border border-white/10 space-y-4">
        <h3 className="text-sm font-semibold text-slate-200">New edit</h3>

        {groups.length === 0 ? (
          <p className="text-xs text-slate-500 italic">
            No numeric feature data available for this episode.
          </p>
        ) : (
          <>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-slate-400">
              <label className="flex items-center gap-2">
                feature
                <select
                  value={featureIdx}
                  onChange={(e) => setFeatureIdx(Number(e.target.value))}
                  className={SELECT_CLASS}
                >
                  {groups.map((g, i) => (
                    <option key={g.feature} value={i}>
                      {g.feature}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex items-center gap-2">
                dimension
                <select
                  value={dimIdx}
                  onChange={(e) =>
                    setDimIdx(
                      e.target.value === "all" ? "all" : Number(e.target.value),
                    )
                  }
                  className={SELECT_CLASS}
                >
                  {(group?.dims.length ?? 0) > 1 && (
                    <option value="all">all dims</option>
                  )}
                  {group?.dims.map((d, i) => (
                    <option key={d.key} value={i}>
                      {d.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex items-center gap-2">
                operation
                <select
                  value={op}
                  onChange={(e) => setOp(e.target.value as EditOpKind)}
                  className={SELECT_CLASS}
                >
                  <option value="set">set to (=)</option>
                  <option value="add">offset (+=)</option>
                  <option value="scale">scale (×=)</option>
                </select>
              </label>
              <label className="flex items-center gap-2">
                value
                <input
                  type="number"
                  step="any"
                  value={value}
                  onChange={(e) => setValue(Number(e.target.value))}
                  className="w-24 bg-[var(--bg)]/50 border border-white/10 rounded px-1.5 py-1 text-slate-200 tabular-nums"
                />
              </label>
              <label className="flex items-center gap-2">
                apply to
                <select
                  value={scope}
                  onChange={(e) =>
                    setScope(e.target.value as "episode" | "other" | "all")
                  }
                  className={SELECT_CLASS}
                >
                  <option value="episode">this episode ({episodeId})</option>
                  <option value="other">another episode…</option>
                  <option value="all">all episodes</option>
                </select>
              </label>
              {scope === "other" && (
                <label className="flex items-center gap-2">
                  episode
                  <input
                    type="number"
                    min={episodes[0] ?? 0}
                    max={episodes[episodes.length - 1] ?? 0}
                    step={1}
                    value={otherEpisode}
                    onChange={(e) => setOtherEpisode(Number(e.target.value))}
                    onKeyDown={(e) => {
                      if (
                        e.key === "Enter" &&
                        otherEpisodeValid &&
                        otherEpisode !== episodeId
                      ) {
                        router.push(`./episode_${otherEpisode}`);
                      }
                    }}
                    className={`w-20 bg-[var(--bg)]/50 border rounded px-1.5 py-1 text-slate-200 tabular-nums ${otherEpisodeValid ? "border-white/10" : "border-red-400/60"}`}
                  />
                  <button
                    onClick={() => router.push(`./episode_${otherEpisode}`)}
                    disabled={!otherEpisodeValid || otherEpisode === episodeId}
                    title="Load this episode in the viewer — the Edit tab stays open and the preview switches to its data"
                    className="text-xs bg-cyan-400/15 text-cyan-300 border border-cyan-400/40 rounded px-2 py-1 hover:bg-cyan-400/20 transition-colors disabled:opacity-40"
                  >
                    switch to ep {otherEpisodeValid ? otherEpisode : "…"}
                  </button>
                  {!otherEpisodeValid && (
                    <span className="text-red-300">not in dataset</span>
                  )}
                </label>
              )}
            </div>

            {/* range */}
            <div className="space-y-2">
              <div className="flex items-center gap-4 text-xs text-slate-400">
                <label className="flex items-center gap-1.5">
                  <input
                    type="checkbox"
                    checked={useRange}
                    onChange={(e) => setUseRange(e.target.checked)}
                    className="accent-cyan-400"
                  />
                  limit to a frame range
                </label>
                {useRange && (
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => switchMode("fraction")}
                      className={`px-2 py-0.5 rounded border text-xs transition-colors ${rangeMode === "fraction" ? "border-cyan-400/60 text-cyan-300 bg-cyan-400/10" : "border-white/10 text-slate-400 hover:text-slate-200"}`}
                    >
                      % of episode
                    </button>
                    <button
                      onClick={() => switchMode("seconds")}
                      className={`px-2 py-0.5 rounded border text-xs transition-colors ${rangeMode === "seconds" ? "border-cyan-400/60 text-cyan-300 bg-cyan-400/10" : "border-white/10 text-slate-400 hover:text-slate-200"}`}
                    >
                      seconds
                    </button>
                  </div>
                )}
                {useRange && (
                  <span className="tabular-nums text-slate-300">
                    {fmtRange(rangeStart)} – {fmtRange(rangeEnd)}
                  </span>
                )}
              </div>
              {useRange && (
                <div className="relative h-5">
                  <div className="absolute top-1/2 -translate-y-1/2 left-0 right-0 h-1 rounded bg-white/5" />
                  <div
                    className="absolute top-1/2 -translate-y-1/2 h-1 rounded bg-cyan-500"
                    style={{
                      left: `${(rangeStart / (rangeMax || 1)) * 100}%`,
                      right: `${100 - (rangeEnd / (rangeMax || 1)) * 100}%`,
                    }}
                  />
                  <input
                    type="range"
                    min={0}
                    max={rangeMax}
                    step={
                      rangeMode === "fraction" ? 0.01 : 1 / Math.max(fps, 1)
                    }
                    value={rangeStart}
                    onChange={(e) =>
                      setRangeStart(Math.min(Number(e.target.value), rangeEnd))
                    }
                    className={RANGE_THUMB_CLASS}
                    aria-label="Edit range start"
                  />
                  <input
                    type="range"
                    min={0}
                    max={rangeMax}
                    step={
                      rangeMode === "fraction" ? 0.01 : 1 / Math.max(fps, 1)
                    }
                    value={rangeEnd}
                    onChange={(e) =>
                      setRangeEnd(Math.max(Number(e.target.value), rangeStart))
                    }
                    className={RANGE_THUMB_CLASS}
                    aria-label="Edit range end"
                  />
                </div>
              )}
            </div>

            {/* preview */}
            {preview && previewDim && (
              <div className="space-y-1">
                <p className="text-xs text-slate-500">
                  Preview — {previewDim.key}
                  {dimIdx === "all" && " (first of all dims)"} · gray = before,
                  cyan = after
                  {scope === "other" &&
                    ` · preview shows episode ${episodeId}'s data — for a different target episode, "% of episode" ranges are recommended`}
                </p>
                <EditPreview
                  original={preview.original}
                  edited={preview.edited}
                  mask={preview.mask}
                />
              </div>
            )}

            <div className="flex justify-end">
              <button
                onClick={handleAdd}
                disabled={!Number.isFinite(value) || !otherEpisodeValid}
                className="text-xs bg-cyan-400/15 text-cyan-300 border border-cyan-400/40 rounded px-3 py-1.5 hover:bg-cyan-400/20 transition-colors disabled:opacity-40"
              >
                Add edit
              </button>
            </div>
          </>
        )}
      </div>

      {/* ── Pending edits ── */}
      <div className="bg-[var(--surface-1)]/60 rounded-lg p-5 border border-white/10 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-200">
            Pending edits
            <span className="text-xs text-slate-500 ml-2 font-normal">
              ({count})
            </span>
          </h3>
          {count > 0 && (
            <button
              onClick={clear}
              className="text-xs text-slate-500 hover:text-red-400 transition-colors"
            >
              Clear
            </button>
          )}
        </div>
        {count === 0 ? (
          <p className="text-xs text-slate-500">
            No edits queued. Build one above — edits are applied in order.
          </p>
        ) : (
          <ul className="space-y-1.5">
            {edits.map((e, i) => (
              <li
                key={e.id}
                className="flex items-center gap-3 text-xs text-slate-300 bg-[var(--surface-0)]/50 rounded px-3 py-1.5"
              >
                <span className="text-slate-600 tabular-nums shrink-0">
                  {i + 1}.
                </span>
                <span className="font-mono truncate">{describeEdit(e)}</span>
                <button
                  onClick={() => remove(e.id)}
                  className="ml-auto text-slate-500 hover:text-red-400 transition-colors shrink-0"
                >
                  remove
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* ── Apply ── */}
      <div className="bg-[var(--surface-1)]/60 rounded-lg p-5 border border-cyan-400/30 space-y-3">
        <h3 className="text-sm font-semibold text-cyan-300">Apply edits</h3>

        {backendEnabled ? (
          <>
            <p className="text-xs text-slate-400">
              Writes a <span className="text-slate-200">modified copy</span> of
              the dataset (original untouched). Per-episode stats and{" "}
              <code className="text-slate-300">meta/stats.json</code> are
              recomputed for the affected features; videos are carried over
              unchanged.
            </p>
            <div className="grid gap-2 sm:grid-cols-2">
              <label className="text-xs text-slate-400 space-y-1">
                <span>
                  Dataset path on the backend machine (blank = fetch{" "}
                  <span className="text-slate-300">{repoId}</span> from the Hub)
                </span>
                <input
                  type="text"
                  value={localPath}
                  onChange={(e) => {
                    setLocalPath(e.target.value);
                    persistInput(LOCAL_PATH_STORAGE_KEY, e.target.value);
                  }}
                  placeholder="/data/lerobot-datasets/org/dataset"
                  className="w-full bg-[var(--bg)]/50 border border-white/10 rounded px-2 py-1 text-slate-200 font-mono"
                />
              </label>
              <label className="text-xs text-slate-400 space-y-1">
                <span>Output directory (blank = auto)</span>
                <input
                  type="text"
                  value={outputDir}
                  onChange={(e) => {
                    setOutputDir(e.target.value);
                    persistInput(OUTPUT_DIR_STORAGE_KEY, e.target.value);
                  }}
                  placeholder="/data/lerobot-datasets/org/dataset_edited"
                  className="w-full bg-[var(--bg)]/50 border border-white/10 rounded px-2 py-1 text-slate-200 font-mono"
                />
              </label>
            </div>
            <label
              className="flex items-center gap-1.5 text-xs text-slate-400"
              title="Decodes sampled video frames with ffmpeg to refresh image-feature stats (min/max/mean/std per channel). Numeric edits don't change pixels, so use this to fix stats that were already stale."
            >
              <input
                type="checkbox"
                checked={recomputeImageStats}
                onChange={(e) => setRecomputeImageStats(e.target.checked)}
                className="accent-cyan-400"
              />
              also recompute image (camera) stats from video frames — slower
            </label>
            <div className="flex items-center gap-3">
              <button
                onClick={handleApply}
                disabled={count === 0 || applying}
                className="text-xs bg-cyan-400/15 text-cyan-300 border border-cyan-400/40 rounded px-3 py-1.5 hover:bg-cyan-400/20 transition-colors disabled:opacity-40"
              >
                {applying
                  ? "Applying…"
                  : `Apply ${count} edit${count !== 1 ? "s" : ""} → new dataset`}
              </button>
              {applying && (
                <span className="text-xs text-slate-500">
                  Rewriting parquet + stats — this can take a while…
                </span>
              )}
            </div>
            {applyResult && (
              <div className="text-xs text-slate-300 bg-[var(--bg)]/50 rounded px-3 py-2 space-y-1">
                <p>
                  ✓ Edited dataset written to{" "}
                  <span className="font-mono text-cyan-300">
                    {applyResult.output_dir}
                  </span>
                </p>
                <p className="tabular-nums text-slate-400">
                  {applyResult.edits_applied} edits ·{" "}
                  {applyResult.episodes_touched} episodes touched ·{" "}
                  {applyResult.values_changed} values changed
                </p>
                {applyResult.warnings.map((w, i) => (
                  <p key={i} className="text-amber-300/90">
                    ⚠ {w}
                  </p>
                ))}
              </div>
            )}
          </>
        ) : (
          <>
            <p className="text-xs text-slate-400">
              The FastAPI backend isn&apos;t configured (set{" "}
              <code className="text-slate-300">
                NEXT_PUBLIC_ANNOTATE_BACKEND_URL
              </code>{" "}
              and run <code className="text-slate-300">backend/app.py</code>),
              so edits can&apos;t be applied from here. Export the edit list and
              apply it with the backend API instead.
            </p>
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-500">
                {count} edit{count !== 1 ? "s" : ""} to export
              </span>
              <button
                onClick={handleCopyJson}
                disabled={count === 0}
                className="text-xs text-slate-400 hover:text-slate-200 transition-colors disabled:opacity-40"
              >
                {copied ? "Copied ✓" : "Copy JSON"}
              </button>
            </div>
            {count > 0 && (
              <pre className="text-xs text-slate-300 bg-[var(--bg)]/50 rounded px-2 py-1.5 overflow-x-auto max-h-40 select-all">
                {exportJson}
              </pre>
            )}
          </>
        )}

        {backendError && (
          <p className="text-xs text-red-300 whitespace-pre-wrap">
            {backendError}
          </p>
        )}
      </div>
    </div>
  );
}

export default EditPanel;
