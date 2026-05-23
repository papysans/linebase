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
  status: "pending" | "running" | "paused" | "finished" | "failed";
  total_rows: number;
  done_rows: number;
  cost_usd: number;
  prompt_version: string | null;
  model?: string | null;
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
  }) => json<JobSummary>("/api/jobs", { method: "POST", body: JSON.stringify(body), headers: { "Content-Type": "application/json" } }),
  listModels: () => json<ModelsResponse>("/api/models"),
  startJob: (id: string) => json<JobSummary>(`/api/jobs/${id}/start`, { method: "POST" }),
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
  evalRuns: () => json<{ id: number; prompt_version: string; model: string; metrics: Record<string, unknown>; created_at: number }[]>("/api/dev/eval-runs"),
};
