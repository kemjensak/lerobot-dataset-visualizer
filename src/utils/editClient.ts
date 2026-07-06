/**
 * Client for the numeric-edit endpoint of the FastAPI backend
 * (`backend/edits.py`). Shares `NEXT_PUBLIC_ANNOTATE_BACKEND_URL` with the
 * annotations/trim clients — one backend serves all three features.
 */

import { getAnnotateBackendUrl } from "./annotationsClient";
import type { NumericEdit } from "./numericEdit";
import type { TrimDatasetIdent } from "./trimClient";

export interface ApplyEditsResult {
  output_dir: string;
  version: string;
  edits_applied: number;
  episodes_touched: number;
  values_changed: number;
  warnings: string[];
}

export async function applyEditsBackend(
  ident: TrimDatasetIdent,
  edits: NumericEdit[],
  options: { outputDir?: string | null } = {},
): Promise<ApplyEditsResult> {
  const base = getAnnotateBackendUrl();
  if (!base) throw new Error("Backend not configured");
  const res = await fetch(new URL("/api/edit/apply", base).toString(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      repo_id: ident.repoId || null,
      revision: ident.revision || null,
      local_path: ident.localPath || null,
      output_dir: options.outputDir || null,
      edits: edits.map((e) => ({
        feature: e.feature,
        dim: e.dim,
        episode_index: e.episodeIndex,
        range: e.range,
        op: e.op,
        value: e.value,
      })),
    }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `/api/edit/apply: ${res.status}`);
  }
  return res.json() as Promise<ApplyEditsResult>;
}

/** Serializable export of the pending edits (backend-less fallback). */
export function editsToJson(repoId: string, edits: NumericEdit[]): string {
  return JSON.stringify(
    {
      repo_id: repoId,
      edits: edits.map((e) => ({
        feature: e.feature,
        dim: e.dim,
        episode_index: e.episodeIndex,
        range: e.range,
        op: e.op,
        value: e.value,
      })),
    },
    null,
    2,
  );
}
