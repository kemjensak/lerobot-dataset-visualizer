"use client";

import React, { useState, useMemo, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useTrims } from "@/context/trim-context";
import { useTime } from "@/context/time-context";
import {
  computeMotionProfile,
  detectActiveRange,
  isMeaningfulTrim,
  TRIM_DEFAULTS,
  type TrimRange,
} from "@/utils/trimDetection";
import { isAnnotateBackendEnabled } from "@/utils/annotationsClient";
import {
  applyTrimsBackend,
  detectTrimsBackend,
  type ApplyTrimsResult,
} from "@/utils/trimClient";

const RANGE_THUMB_CLASS =
  "absolute inset-0 w-full appearance-none bg-transparent pointer-events-none [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-cyan-400 [&::-webkit-slider-thumb]:cursor-pointer [&::-moz-range-thumb]:pointer-events-auto [&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:w-3.5 [&::-moz-range-thumb]:h-3.5 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-white [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-cyan-400 [&::-moz-range-thumb]:cursor-pointer";

const LOCAL_PATH_STORAGE_KEY = "trim-local-path";
const OUTPUT_DIR_STORAGE_KEY = "trim-output-dir";

function fmtSec(t: number): string {
  return `${t.toFixed(2)}s`;
}

// ─── Motion profile sparkline ─────────────────────────────────────

function MotionSparkline({
  motion,
  timestamps,
  duration,
  range,
  threshold,
}: {
  motion: number[];
  timestamps: number[];
  duration: number;
  range: TrimRange;
  threshold: number;
}) {
  const W = 600;
  const H = 64;
  const maxMotion = Math.max(...motion, 1e-9);
  const points = motion
    .map((m, i) => {
      const x = (timestamps[i] / Math.max(duration, 1e-9)) * W;
      const y = H - (m / maxMotion) * (H - 4);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const keepX = (range.start / Math.max(duration, 1e-9)) * W;
  const keepW = ((range.end - range.start) / Math.max(duration, 1e-9)) * W;
  const thresholdY = H - (threshold / maxMotion) * (H - 4);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      className="w-full h-16 rounded bg-[var(--bg)]/60 border border-white/5"
    >
      {/* kept region */}
      <rect
        x={keepX}
        y={0}
        width={Math.max(0, keepW)}
        height={H}
        fill="rgba(56,189,248,0.10)"
      />
      {/* cut regions */}
      {keepX > 0 && (
        <rect
          x={0}
          y={0}
          width={keepX}
          height={H}
          fill="rgba(239,68,68,0.10)"
        />
      )}
      {keepX + keepW < W && (
        <rect
          x={keepX + keepW}
          y={0}
          width={W - keepX - keepW}
          height={H}
          fill="rgba(239,68,68,0.10)"
        />
      )}
      {threshold > 0 && (
        <line
          x1={0}
          y1={thresholdY}
          x2={W}
          y2={thresholdY}
          stroke="rgba(148,163,184,0.5)"
          strokeWidth={1}
          strokeDasharray="4 4"
          vectorEffect="non-scaling-stroke"
        />
      )}
      <polyline
        points={points}
        fill="none"
        stroke="#38bdf8"
        strokeWidth={1.2}
        vectorEffect="non-scaling-stroke"
      />
      <line
        x1={keepX}
        y1={0}
        x2={keepX}
        y2={H}
        stroke="#f87171"
        strokeWidth={1}
        vectorEffect="non-scaling-stroke"
      />
      <line
        x1={keepX + keepW}
        y1={0}
        x2={keepX + keepW}
        y2={H}
        stroke="#f87171"
        strokeWidth={1}
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

// ─── Main panel ──────────────────────────────────────────────────

interface TrimPanelProps {
  repoId: string;
  episodeId: number;
  duration: number;
  fps: number;
  flatChartData: Record<string, number>[];
}

function TrimPanel({
  repoId,
  episodeId,
  duration,
  fps,
  flatChartData,
}: TrimPanelProps) {
  const router = useRouter();
  const { trims, count, get, set, setMany, remove, clear } = useTrims();
  const { currentTime } = useTime();
  const backendEnabled = isAnnotateBackendEnabled();

  const saved = get(episodeId);
  const [draft, setDraft] = useState<TrimRange>(
    () => saved ?? { start: 0, end: duration },
  );
  const [sensitivityPct, setSensitivityPct] = useState(
    TRIM_DEFAULTS.sensitivity * 100,
  );
  const [padding, setPadding] = useState<number>(TRIM_DEFAULTS.paddingSeconds);
  const [detectMessage, setDetectMessage] = useState<string | null>(null);

  // Reset the draft when navigating to another episode.
  useEffect(() => {
    setDraft(get(episodeId) ?? { start: 0, end: duration });
    setDetectMessage(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [episodeId, duration]);

  const profile = useMemo(
    () => computeMotionProfile(flatChartData, duration),
    [flatChartData, duration],
  );
  const threshold = useMemo(() => {
    if (!profile) return 0;
    const sorted = [...profile.motion].sort((a, b) => a - b);
    const p95 =
      sorted[
        Math.min(sorted.length - 1, Math.floor(0.95 * (sorted.length - 1)))
      ];
    return (sensitivityPct / 100) * p95;
  }, [profile, sensitivityPct]);

  const handleAutoDetect = useCallback(() => {
    const range = detectActiveRange(flatChartData, duration, {
      sensitivity: sensitivityPct / 100,
      paddingSeconds: padding,
    });
    if (!range) {
      setDetectMessage(
        "No motion detected in this episode — nothing to anchor a trim to.",
      );
      return;
    }
    setDraft(range);
    setDetectMessage(
      isMeaningfulTrim(range, duration)
        ? null
        : "Motion spans the whole episode — nothing worth trimming.",
    );
  }, [flatChartData, duration, sensitivityPct, padding]);

  const draftChanged = draft.start > 0.001 || draft.end < duration - 0.001;
  const savedMatchesDraft =
    !!saved &&
    Math.abs(saved.start - draft.start) < 1e-6 &&
    Math.abs(saved.end - draft.end) < 1e-6;

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
  const persistInput = (key: string, value: string) => {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* ignore */
    }
  };
  const ident = useMemo(
    () => ({
      repoId: localPath ? null : repoId,
      localPath: localPath || null,
    }),
    [repoId, localPath],
  );

  const [detectingAll, setDetectingAll] = useState(false);
  const [applying, setApplying] = useState(false);
  const [recomputeImageStats, setRecomputeImageStats] = useState(false);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [applyResult, setApplyResult] = useState<ApplyTrimsResult | null>(null);
  const [copied, setCopied] = useState(false);

  const handleDetectAll = useCallback(async () => {
    setDetectingAll(true);
    setBackendError(null);
    try {
      const result = await detectTrimsBackend(ident, {
        sensitivity: sensitivityPct / 100,
        paddingSeconds: padding,
      });
      setMany(
        result.episodes.map((e) => [
          e.episode_index,
          { start: e.start, end: e.end },
        ]),
      );
      setDetectMessage(
        `Detected trims for ${result.episodes.length} of ${result.scanned} episodes.`,
      );
    } catch (err) {
      setBackendError(err instanceof Error ? err.message : String(err));
    } finally {
      setDetectingAll(false);
    }
  }, [ident, sensitivityPct, padding, setMany]);

  const handleApply = useCallback(async () => {
    setApplying(true);
    setBackendError(null);
    setApplyResult(null);
    try {
      const result = await applyTrimsBackend(ident, trims, {
        outputDir: outputDir || null,
        recomputeImageStats,
      });
      setApplyResult(result);
    } catch (err) {
      setBackendError(err instanceof Error ? err.message : String(err));
    } finally {
      setApplying(false);
    }
  }, [ident, trims, outputDir, recomputeImageStats]);

  const exportJson = useMemo(() => {
    const obj: Record<string, TrimRange> = {};
    for (const [ep, range] of [...trims.entries()].sort(
      (a, b) => a[0] - b[0],
    )) {
      obj[String(ep)] = {
        start: Number(range.start.toFixed(3)),
        end: Number(range.end.toFixed(3)),
      };
    }
    return JSON.stringify(
      { repo_id: repoId, unit: "seconds", trims: obj },
      null,
      2,
    );
  }, [trims, repoId]);

  const handleCopyJson = useCallback(() => {
    navigator.clipboard.writeText(exportJson);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }, [exportJson]);

  const sortedTrims = useMemo(
    () => [...trims.entries()].sort((a, b) => a[0] - b[0]),
    [trims],
  );

  return (
    <div className="max-w-5xl mx-auto py-6 space-y-8">
      <div>
        <h2 className="text-xl font-bold text-slate-100">Trim</h2>
        <p className="text-sm text-slate-400 mt-1">
          Cut away the motionless head and tail of episodes. Auto-detect finds
          the active range from action data; adjust it, save it per episode,
          then apply all trims at once to produce a trimmed copy of the dataset.
        </p>
      </div>

      {/* ── Current episode ── */}
      <div className="bg-[var(--surface-1)]/60 rounded-lg p-5 border border-white/10 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-200">
            Episode {episodeId}
            {saved && (
              <span className="ml-2 text-xs font-normal text-cyan-300">
                trim saved
              </span>
            )}
          </h3>
          <span className="text-xs text-slate-500 tabular-nums">
            {fmtSec(duration)} · ~{Math.round(duration * fps)} frames
          </span>
        </div>

        {profile ? (
          <MotionSparkline
            motion={profile.motion}
            timestamps={profile.timestamps}
            duration={duration}
            range={draft}
            threshold={threshold}
          />
        ) : (
          <p className="text-xs text-slate-500 italic">
            No numeric action/state data available for motion detection.
          </p>
        )}

        {/* dual range slider */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="tabular-nums">
              keep from {fmtSec(draft.start)} (frame{" "}
              {Math.round(draft.start * fps)})
            </span>
            <span className="tabular-nums">
              to {fmtSec(draft.end)} (frame {Math.round(draft.end * fps)})
            </span>
          </div>
          <div className="relative h-5">
            <div className="absolute top-1/2 -translate-y-1/2 left-0 right-0 h-1 rounded bg-red-500/25" />
            <div
              className="absolute top-1/2 -translate-y-1/2 h-1 rounded bg-cyan-500"
              style={{
                left: `${(draft.start / (duration || 1)) * 100}%`,
                right: `${100 - (draft.end / (duration || 1)) * 100}%`,
              }}
            />
            <input
              type="range"
              min={0}
              max={duration}
              step={1 / Math.max(fps, 1)}
              value={draft.start}
              onChange={(e) =>
                setDraft((d) => ({
                  ...d,
                  start: Math.min(Number(e.target.value), d.end),
                }))
              }
              className={RANGE_THUMB_CLASS}
              aria-label="Trim start"
            />
            <input
              type="range"
              min={0}
              max={duration}
              step={1 / Math.max(fps, 1)}
              value={draft.end}
              onChange={(e) =>
                setDraft((d) => ({
                  ...d,
                  end: Math.max(Number(e.target.value), d.start),
                }))
              }
              className={RANGE_THUMB_CLASS}
              aria-label="Trim end"
            />
          </div>
          <div className="flex items-center justify-between text-xs text-slate-500 tabular-nums">
            <span>cut head: {fmtSec(draft.start)}</span>
            <span>cut tail: {fmtSec(Math.max(0, duration - draft.end))}</span>
          </div>
        </div>

        {/* detection controls */}
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-slate-400">
          <label className="flex items-center gap-2">
            sensitivity
            <input
              type="range"
              min={1}
              max={50}
              step={1}
              value={sensitivityPct}
              onChange={(e) => setSensitivityPct(Number(e.target.value))}
              className="w-28 h-1 accent-cyan-400"
            />
            <span className="tabular-nums w-8">{sensitivityPct}%</span>
          </label>
          <label className="flex items-center gap-2">
            padding
            <input
              type="number"
              min={0}
              step={0.05}
              value={padding}
              onChange={(e) => setPadding(Math.max(0, Number(e.target.value)))}
              className="w-16 bg-[var(--bg)]/50 border border-white/10 rounded px-1.5 py-0.5 text-slate-200 tabular-nums"
            />
            s
          </label>
        </div>

        {/* actions */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={handleAutoDetect}
            disabled={!profile}
            className="text-xs bg-cyan-400/15 text-cyan-300 border border-cyan-400/40 rounded px-2.5 py-1 hover:bg-cyan-400/20 transition-colors disabled:opacity-40"
          >
            Auto-detect
          </button>
          <button
            onClick={() =>
              setDraft((d) => ({
                ...d,
                start: Math.min(currentTime, d.end),
              }))
            }
            className="text-xs text-slate-300 border border-white/10 rounded px-2.5 py-1 hover:bg-white/5 transition-colors"
            title="Use the playback position from the Episodes tab"
          >
            Start = playhead ({fmtSec(currentTime)})
          </button>
          <button
            onClick={() =>
              setDraft((d) => ({ ...d, end: Math.max(currentTime, d.start) }))
            }
            className="text-xs text-slate-300 border border-white/10 rounded px-2.5 py-1 hover:bg-white/5 transition-colors"
            title="Use the playback position from the Episodes tab"
          >
            End = playhead
          </button>
          <button
            onClick={() => setDraft({ start: 0, end: duration })}
            className="text-xs text-slate-400 border border-white/10 rounded px-2.5 py-1 hover:bg-white/5 transition-colors"
          >
            Reset
          </button>
          <div className="ml-auto flex items-center gap-2">
            {saved && (
              <button
                onClick={() => remove(episodeId)}
                className="text-xs text-slate-500 hover:text-red-400 transition-colors px-2 py-1"
              >
                Remove trim
              </button>
            )}
            <button
              onClick={() => set(episodeId, draft)}
              disabled={!draftChanged || savedMatchesDraft}
              className="text-xs bg-cyan-400/15 text-cyan-300 border border-cyan-400/40 rounded px-3 py-1 hover:bg-cyan-400/20 transition-colors disabled:opacity-40"
            >
              {savedMatchesDraft ? "Saved" : "Save trim for this episode"}
            </button>
          </div>
        </div>

        {detectMessage && (
          <p className="text-xs text-amber-300/90">{detectMessage}</p>
        )}
      </div>

      {/* ── Saved trims ── */}
      <div className="bg-[var(--surface-1)]/60 rounded-lg p-5 border border-white/10 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-200">
            Pending trims
            <span className="text-xs text-slate-500 ml-2 font-normal">
              ({count})
            </span>
          </h3>
          <div className="flex items-center gap-3">
            <button
              onClick={handleDetectAll}
              disabled={!backendEnabled || detectingAll}
              title={
                backendEnabled
                  ? "Scan every episode's action data on the backend"
                  : "Requires the FastAPI backend (NEXT_PUBLIC_ANNOTATE_BACKEND_URL)"
              }
              className="text-xs bg-cyan-400/15 text-cyan-300 border border-cyan-400/40 rounded px-2.5 py-1 hover:bg-cyan-400/20 transition-colors disabled:opacity-40"
            >
              {detectingAll ? "Detecting…" : "Auto-detect all episodes"}
            </button>
            {count > 0 && (
              <button
                onClick={clear}
                className="text-xs text-slate-500 hover:text-red-400 transition-colors"
              >
                Clear
              </button>
            )}
          </div>
        </div>

        {count === 0 ? (
          <p className="text-xs text-slate-500">
            No trims saved yet. Save one for the current episode above, or
            auto-detect across all episodes.
          </p>
        ) : (
          <div className="max-h-64 overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="text-slate-500 sticky top-0 bg-[var(--surface-1)]">
                <tr className="text-left">
                  <th className="py-1 pr-3 font-normal">episode</th>
                  <th className="py-1 pr-3 font-normal">keep</th>
                  <th className="py-1 pr-3 font-normal">cut head</th>
                  <th className="py-1 pr-3 font-normal">cut tail</th>
                  <th className="py-1 font-normal" />
                </tr>
              </thead>
              <tbody className="text-slate-300 tabular-nums">
                {sortedTrims.map(([ep, range]) => (
                  <tr key={ep} className="border-t border-white/5">
                    <td className="py-1 pr-3">
                      <button
                        onClick={() => router.push(`./episode_${ep}`)}
                        className="text-cyan-300 hover:underline"
                      >
                        ep {ep}
                      </button>
                    </td>
                    <td className="py-1 pr-3">
                      {fmtSec(range.start)} – {fmtSec(range.end)}
                    </td>
                    <td className="py-1 pr-3">{fmtSec(range.start)}</td>
                    <td className="py-1 pr-3">
                      {ep === episodeId
                        ? fmtSec(Math.max(0, duration - range.end))
                        : "—"}
                    </td>
                    <td className="py-1 text-right">
                      <button
                        onClick={() => remove(ep)}
                        className="text-slate-500 hover:text-red-400 transition-colors"
                      >
                        remove
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ── Apply ── */}
      <div className="bg-[var(--surface-1)]/60 rounded-lg p-5 border border-cyan-400/30 space-y-3">
        <h3 className="text-sm font-semibold text-cyan-300">Apply trims</h3>

        {backendEnabled ? (
          <>
            <p className="text-xs text-slate-400">
              Writes a <span className="text-slate-200">trimmed copy</span> of
              the dataset (original untouched): data parquet rows outside each
              range are dropped, timestamps and indices are rebased, and episode
              metadata is rewritten. v3.0 video segments are trimmed losslessly
              via timestamps; v2.x videos are cut with ffmpeg.
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
                  placeholder="/data/lerobot-datasets/org/dataset_trimmed"
                  className="w-full bg-[var(--bg)]/50 border border-white/10 rounded px-2 py-1 text-slate-200 font-mono"
                />
              </label>
            </div>
            <label
              className="flex items-center gap-1.5 text-xs text-slate-400"
              title="Decodes sampled video frames with ffmpeg to refresh image-feature stats (min/max/mean/std per channel) after the trim. Without this, camera stats keep their pre-trim values."
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
                  : `Apply ${count} trim${count !== 1 ? "s" : ""} → new dataset`}
              </button>
              {applying && (
                <span className="text-xs text-slate-500">
                  Rewriting parquet/videos — this can take a while…
                </span>
              )}
            </div>
            {applyResult && (
              <div className="text-xs text-slate-300 bg-[var(--bg)]/50 rounded px-3 py-2 space-y-1">
                <p>
                  ✓ Trimmed dataset written to{" "}
                  <span className="font-mono text-cyan-300">
                    {applyResult.output_dir}
                  </span>
                </p>
                <p className="tabular-nums text-slate-400">
                  {applyResult.episodes_trimmed} episodes trimmed ·{" "}
                  {applyResult.frames_before} → {applyResult.frames_after}{" "}
                  frames
                  {applyResult.videos_processed > 0 &&
                    ` · ${applyResult.videos_processed} videos re-cut`}
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
              so trims can&apos;t be applied from here. Export the trim list and
              apply it with the backend API instead.
            </p>
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-500">
                {count} trim{count !== 1 ? "s" : ""} to export
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

export default TrimPanel;
