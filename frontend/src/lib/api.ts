export interface SheetInfo {
  name: string;
  rows: number;
  columns: number;
  header: string[];
  sample_rows: (string | null)[][];
}

export interface UploadResponse {
  id: string;
  filename: string;
  size: number;
  sheets: SheetInfo[];
}

export interface JobSummary {
  id: string;
  upload_id: string;
  sheet_name: string;
  logo_column: string;
  evidence_column: string;
  appno_column: string;
  threshold: number;
  status: "pending" | "running" | "paused" | "finished" | "failed" | "cancelled";
  total_rows: number;
  done_rows: number;
  cost_usd: number;
  prompt_version: string | null;
  model?: string | null;
  created_at?: number;
  /** Job-level opt-in: every evidence call is followed by a verify-loop
   *  call to confirm the crop actually contains the trademark shape.
   *  Doubles per-row cost. Server projects the SQLite INTEGER to a bool. */
  verify_loop?: boolean;
}

export interface ModelOption {
  id: string;
  provider: string;
  label: string;
  notes: string;
}

export interface ModelsResponse {
  whitelist: ModelOption[];
  default: string;
  allow_custom: boolean;
}

export interface JobRow {
  id: number;
  job_id: string;
  row_index: number;
  appno: string | null;
  logo_url: string | null;
  evidence_urls: string[];
  status: "pending" | "running" | "ok" | "bad" | "needs_review" | "failed";
  best_crop_path: string | null;
  human_status: string | null;
  notes: string | null;
  best_evidence_url?: string | null;
  best_confidence?: number | null;
  best_clarity?: number | null;
  best_completeness?: number | null;
  best_isolation?: number | null;
  best_reason?: string | null;
  best_fallback_model?: string | null;
  /** Pixel bbox [x1, y1, x2, y2] in the chosen evidence image, or null when
   *  no best was selected or the LLM didn't return a bbox. Used by the
   *  review-detail modal to overlay the LLM's region on the full-size image. */
  best_bbox?: [number, number, number, number] | null;
  /** Per-evidence meta projection. Keyed by evidence URL; each value carries
   *  enough information for the row-detail modal to explain WHY a sibling
   *  evidence was rejected (verified=false, fit=wrong, sanity_rejected=...,
   *  fallback_model=gpt-5.5, etc.). Empty object on rows that haven't been
   *  processed yet. */
  match_meta?: Record<string, EvidenceMeta>;
}

export interface EvidenceMeta {
  found?: boolean | null;
  confidence?: number | null;
  /** Verify-loop outcome: True = verify accepted the bbox, False = rejected,
   *  null/undefined = verify was disabled or the call hasn't happened yet. */
  verified?: boolean | null;
  /** One of {"tight", "loose", "too_tight", "wrong"} when verify ran. */
  fit?: string | null;
  /** Short text from the verify call explaining the fit verdict. */
  verify_reason?: string | null;
  /** Non-null short reason when the post-crop sanity check rejected the bbox
   *  (e.g. "crop_too_small (area_ratio=0.001)" or "crop_mostly_blank"). */
  sanity_rejected?: string | null;
  /** Set to "gpt-5.5" when the primary model rejected the tile (<28 px) and
   *  we silently retried. Also set to "gpt-5.5" by the Ark-overdue fallback. */
  fallback_model?: string | null;
  /** Per-evidence terminal error string ("download: ...", "llm: ..."). */
  error?: string | null;
}

async function json<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const r = await fetch(input, init);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} on ${typeof input === "string" ? input : ""}`);
  return r.json() as Promise<T>;
}

export interface UploadProgress {
  loaded: number;
  total: number;
}

/**
 * XHR-based upload that exposes real upload progress.
 * `fetch` doesn't expose ReadableStream upload progress in any current browser,
 * so for the 428 MB workbook we drop to XHR just here — keeps everywhere else
 * on fetch.
 */
function uploadXlsxXhr(
  file: File,
  onProgress?: (p: UploadProgress) => void,
): Promise<UploadResponse> {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    fd.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/uploads", true);
    xhr.responseType = "text";
    if (onProgress) {
      xhr.upload.onprogress = (e: ProgressEvent) => {
        // `lengthComputable` is true once the browser sees Content-Length on the
        // outgoing request, which is always the case for FormData.
        if (e.lengthComputable) onProgress({ loaded: e.loaded, total: e.total });
      };
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as UploadResponse);
        } catch (parseErr) {
          reject(new Error(`upload ok but response parse failed: ${parseErr}`));
        }
      } else {
        reject(new Error(`${xhr.status} ${xhr.statusText} on /api/uploads`));
      }
    };
    xhr.onerror = () => reject(new Error("network error on /api/uploads"));
    xhr.onabort = () => reject(new Error("upload aborted"));
    xhr.send(fd);
  });
}

export const api = {
  uploadXlsx: (
    file: File,
    onProgress?: (p: UploadProgress) => void,
  ): Promise<UploadResponse> => uploadXlsxXhr(file, onProgress),
  getUpload: (id: string) => json<UploadResponse>(`/api/uploads/${id}`),
  createJob: (body: {
    upload_id: string;
    sheet_name: string;
    appno_column: string;
    logo_column: string;
    evidence_column: string;
    sample_kind: "first_n" | "range" | "row_ids";
    sample_params: Record<string, unknown>;
    threshold: number;
    model?: string | null;
    verify_loop?: boolean;
  }) => json<JobSummary>("/api/jobs", { method: "POST", body: JSON.stringify(body), headers: { "Content-Type": "application/json" } }),
  listModels: () => json<ModelsResponse>("/api/models"),
  startJob: (id: string) => json<JobSummary>(`/api/jobs/${id}/start`, { method: "POST" }),
  /** Cancel an in-flight job. Kills the asyncio task, marks remaining
   *  pending/running rows as failed("cancelled by user"), flips job status
   *  to "cancelled". Idempotent. See POST /api/jobs/{id}/cancel. */
  cancelJob: (id: string) => json<JobSummary>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  getJob: (id: string) => json<JobSummary>(`/api/jobs/${id}`),
  listRows: (id: string, status?: string) =>
    json<JobRow[]>(`/api/jobs/${id}/rows${status ? `?status=${status}` : ""}`),
  setRowStatus: (jobId: string, rowId: number, status: string, notes?: string) =>
    json<JobRow>(`/api/jobs/${jobId}/rows/${rowId}/status`, {
      method: "POST",
      body: JSON.stringify({ human_status: status, notes }),
      headers: { "Content-Type": "application/json" },
    }),
  rerun: (jobId: string, rowIds?: number[]) =>
    json<JobSummary>(`/api/jobs/${jobId}/rerun`, {
      method: "POST",
      body: JSON.stringify({ row_ids: rowIds ?? null }),
      headers: { "Content-Type": "application/json" },
    }),
  /** Per-row rerun with opt-in verify-loop + model override. The overrides
   *  persist on the job (no transient state) — pick the model again next
   *  rerun if you want to switch back. See /api/jobs/{id}/rows/{rowId}/rerun. */
  rerunRow: (
    jobId: string,
    rowId: number,
    opts?: { verify?: boolean; model?: string | null },
  ) =>
    json<JobSummary>(`/api/jobs/${jobId}/rows/${rowId}/rerun`, {
      method: "POST",
      body: JSON.stringify({
        verify: opts?.verify ?? false,
        model: opts?.model ?? null,
      }),
      headers: { "Content-Type": "application/json" },
    }),
  evalRuns: () => json<{ id: number; prompt_version: string; model: string; metrics: Record<string, unknown>; created_at: number }[]>("/api/dev/eval-runs"),
  /** Recent jobs, newest first. Backs the friendly empty state on
   *  /run, /review, /download when the user lands without a jobId in the URL. */
  listJobs: (limit = 5) =>
    json<JobSummary[]>(`/api/jobs?limit=${encodeURIComponent(limit)}`),
};
