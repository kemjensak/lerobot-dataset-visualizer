/**
 * Client for the trim endpoints of the FastAPI backend (`backend/trim.py`).
 *
 * Shares the `NEXT_PUBLIC_ANNOTATE_BACKEND_URL` configuration with the
 * annotations client — one backend serves both features. When unset, the
 * Trim tab still works for range selection but can only export the trim
 * list as JSON instead of applying it.
 */

import { getAnnotateBackendUrl } from "./annotationsClient";
import type { TrimRange } from "./trimDetection";

export interface TrimDatasetIdent {
  repoId?: string | null;
  localPath?: string | null;
  revision?: string | null;
}

export interface DetectedTrim {
  episode_index: number;
  /** Seconds to keep, episode-relative. */
  start: number;
  end: number;
  duration: number;
  cut_head: number;
  cut_tail: number;
}

export interface DetectTrimsResult {
  episodes: DetectedTrim[];
  scanned: number;
}

export interface ApplyTrimsResult {
  output_dir: string;
  version: string;
  episodes_trimmed: number;
  frames_before: number;
  frames_after: number;
  videos_processed: number;
  warnings: string[];
}

function identBody(ident: TrimDatasetIdent) {
  return {
    repo_id: ident.repoId || null,
    revision: ident.revision || null,
    local_path: ident.localPath || null,
  };
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const base = getAnnotateBackendUrl();
  if (!base) throw new Error("Backend not configured");
  const res = await fetch(new URL(path, base).toString(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `${path}: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function detectTrimsBackend(
  ident: TrimDatasetIdent,
  options: {
    sensitivity: number;
    paddingSeconds: number;
    minTrimSeconds?: number;
  },
): Promise<DetectTrimsResult> {
  return post<DetectTrimsResult>("/api/trim/detect", {
    ...identBody(ident),
    sensitivity: options.sensitivity,
    padding_seconds: options.paddingSeconds,
    min_trim_seconds: options.minTrimSeconds ?? 0.1,
  });
}

export async function applyTrimsBackend(
  ident: TrimDatasetIdent,
  trims: Map<number, TrimRange>,
  options: { outputDir?: string | null; recomputeImageStats?: boolean } = {},
): Promise<ApplyTrimsResult> {
  const trimsObj: Record<string, TrimRange> = {};
  for (const [ep, range] of trims) trimsObj[String(ep)] = range;
  return post<ApplyTrimsResult>("/api/trim/apply", {
    ...identBody(ident),
    trims: trimsObj,
    output_dir: options.outputDir || null,
    recompute_image_stats: !!options.recomputeImageStats,
  });
}
