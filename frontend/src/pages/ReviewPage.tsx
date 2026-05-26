import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, Filter, Maximize2, RotateCcw } from "lucide-react";
import { api, type EvidenceMeta, type JobRow, type ModelOption } from "@/lib/api";
import { setSession } from "@/lib/session";
import { GlassButton } from "@/components/GlassButton";
import { GlassCard } from "@/components/GlassCard";
import { GlassInput, GlassSelect } from "@/components/GlassInput";
import { GlassPill, type PillStatus } from "@/components/GlassPill";
import { GlassSpinner } from "@/components/GlassSpinner";
import { RowDetailModal } from "@/components/RowDetailModal";
import { RowRerunDialog } from "@/components/RowRerunDialog";
import { cn } from "@/lib/cn";

type Filter = "" | "ok" | "bad" | "needs_review" | "failed";

const FILTER_LABEL: Record<Filter, string> = {
  "": "全部",
  ok: "OK",
  bad: "BAD",
  needs_review: "需要确认",
  failed: "失败",
};

export function ReviewPage() {
  const { jobId = "" } = useParams();
  const qc = useQueryClient();
  const [filter, setFilter] = useState<Filter>("");
  // ID of the row whose detail modal is open. We track by id (not the row
  // object) so that when /api/jobs/{id}/rows is re-fetched the modal stays
  // bound to the latest server-side state of that row (e.g., notes saved by
  // the inline pill update inside the modal show up immediately on the card).
  const [detailRowId, setDetailRowId] = useState<number | null>(null);
  const [detailEvidenceUrl, setDetailEvidenceUrl] = useState<string | null>(null);
  // ID of the row whose per-row rerun dialog is open. Separate state from
  // detail modal so the user can open one without closing the other (the
  // modal also surfaces a 🔄 button that triggers the same dialog).
  const [rerunRowId, setRerunRowId] = useState<number | null>(null);

  // Sync URL jobId into session — covers deep-link / shared URL / refresh.
  useEffect(() => {
    if (jobId) setSession({ jobId });
  }, [jobId]);

  const { data: rows, isLoading } = useQuery({
    queryKey: ["rows", jobId, filter],
    queryFn: () => api.listRows(jobId, filter || undefined),
  });
  // The job summary is fetched once so we can surface `model` in the header.
  // Cached infinitely — the model doesn't change during review.
  const { data: job } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.getJob(jobId),
  });

  const setStatus = useMutation({
    mutationFn: ({
      rowId,
      status,
      notes,
    }: {
      rowId: number;
      status: string;
      notes?: string;
    }) => api.setRowStatus(jobId, rowId, status, notes),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["rows", jobId] }),
  });

  const rerun = useMutation({
    mutationFn: () => api.rerun(jobId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["rows", jobId] }),
  });

  const rerunRow = useMutation({
    mutationFn: ({
      rowId,
      verify,
      model,
    }: {
      rowId: number;
      verify?: boolean;
      model?: string | null;
    }) => api.rerunRow(jobId, rowId, { verify, model }),
    onSuccess: () => {
      // Refetch both rows and job so the model + verify_loop labels update.
      qc.invalidateQueries({ queryKey: ["rows", jobId] });
      qc.invalidateQueries({ queryKey: ["job", jobId] });
      setRerunRowId(null);
    },
  });

  // Models list — fetched once for the rerun dialog dropdown. Same endpoint
  // ConfigurePage uses, so the cache key is shared.
  const { data: modelsResp } = useQuery({
    queryKey: ["models"],
    queryFn: () => api.listModels(),
  });

  return (
    <div className="space-y-5">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">审查结果</h1>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-slate-600 dark:text-slate-400">
          <span>逐行确认匹配质量，可批量重跑非 OK 行。</span>
          {job?.model && (
            <span
              className="inline-flex items-center gap-1 rounded-full border border-white/40 bg-white/40 px-2 py-0.5 text-[11px] font-medium text-slate-700 backdrop-blur-md dark:border-white/10 dark:bg-white/5 dark:text-slate-200"
              title={job.model}
            >
              <span className="uppercase tracking-wider text-slate-500 dark:text-slate-400">
                model
              </span>
              <span className="max-w-[260px] truncate font-mono text-aurora-cyan">
                {job.model}
              </span>
            </span>
          )}
        </div>
      </header>

      {/* Floating glass toolbar */}
      <GlassCard className="sticky top-20 z-10 flex flex-wrap items-center gap-3 px-4 py-3">
        <div className="flex items-center gap-2">
          <Filter size={14} className="text-aurora-cyan" />
          <GlassSelect
            value={filter}
            onChange={(e) => setFilter(e.target.value as Filter)}
            className="w-36"
          >
            {(Object.keys(FILTER_LABEL) as Filter[]).map((k) => (
              <option key={k || "all"} value={k}>
                {FILTER_LABEL[k]}
              </option>
            ))}
          </GlassSelect>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <GlassButton
            variant="warn"
            size="sm"
            onClick={() => rerun.mutate()}
            disabled={rerun.isPending}
            leadingIcon={
              rerun.isPending ? <GlassSpinner size={14} /> : <RotateCcw size={14} />
            }
          >
            重跑非 OK 行
          </GlassButton>
          <Link to={`/download/${jobId}`}>
            <GlassButton variant="primary" size="sm" leadingIcon={<Download size={14} />}>
              下载
            </GlassButton>
          </Link>
        </div>
      </GlassCard>

      <div className="space-y-3">
        {isLoading && (
          <div className="flex items-center gap-2 text-slate-500">
            <GlassSpinner /> 加载中…
          </div>
        )}
        {rows?.map((r) => (
          <ReviewRow
            key={r.id}
            row={r}
            onSet={(s, n) =>
              setStatus.mutate({ rowId: r.id, status: s, notes: n })
            }
            onOpenDetail={(evidenceUrl) => {
              setDetailRowId(r.id);
              setDetailEvidenceUrl(evidenceUrl ?? null);
            }}
            onOpenRerun={() => setRerunRowId(r.id)}
          />
        ))}
        {rows && rows.length === 0 && (
          <GlassCard className="p-6 text-center text-sm text-slate-500">
            没有匹配的行。
          </GlassCard>
        )}
      </div>

      {detailRowId !== null && rows && (() => {
        // Resolve the latest version of the row from the freshly-fetched list
        // so the modal reflects in-place edits without us having to keep a
        // separate copy in state.
        const r = rows.find((x) => x.id === detailRowId);
        if (!r) return null;
        return (
          <RowDetailModal
            row={r}
            initialEvidenceUrl={detailEvidenceUrl}
            onClose={() => {
              setDetailRowId(null);
              setDetailEvidenceUrl(null);
            }}
            onSet={(s, n) =>
              setStatus.mutate({ rowId: r.id, status: s, notes: n })
            }
            onOpenRerun={() => setRerunRowId(r.id)}
          />
        );
      })()}

      {rerunRowId !== null && rows && (() => {
        const r = rows.find((x) => x.id === rerunRowId);
        if (!r) return null;
        return (
          <RowRerunDialog
            row={r}
            models={modelsResp?.whitelist as ModelOption[] | undefined}
            defaultModel={modelsResp?.default}
            pending={rerunRow.isPending}
            onClose={() => setRerunRowId(null)}
            onSubmit={(opts) =>
              rerunRow.mutate({ rowId: r.id, ...opts })
            }
          />
        );
      })()}
    </div>
  );
}

function statusTone(status: string): string {
  if (status === "ok") return "underglow-ok";
  if (status === "bad" || status === "failed") return "underglow-bad";
  if (status === "needs_review") return "underglow-review";
  return "";
}

function ReviewRow({
  row,
  onSet,
  onOpenDetail,
  onOpenRerun,
}: {
  row: JobRow;
  onSet: (status: PillStatus, notes?: string) => void;
  onOpenDetail: (evidenceUrl?: string | null) => void;
  onOpenRerun: () => void;
}) {
  const [notes, setNotes] = useState(row.notes ?? "");
  const current = (row.human_status ?? null) as PillStatus | null;

  return (
    <GlassCard
      className={cn(
        "grid grid-cols-1 gap-4 p-4 lg:grid-cols-[5rem_minmax(0,1fr)_minmax(18rem,24rem)]",
        statusTone(row.human_status ?? row.status),
      )}
    >
      {/* row id */}
      <div className="text-xs text-slate-500 dark:text-slate-400">
        <div className="text-[11px] uppercase tracking-wider">行</div>
        <div className="text-base font-semibold text-slate-900 dark:text-slate-100">
          {row.row_index}
        </div>
        <div className="mt-1 font-mono text-[11px] text-aurora-magenta">
          {row.appno ?? "-"}
        </div>
        <GlassButton
          variant="ghost"
          size="sm"
          onClick={() => onOpenDetail(row.best_evidence_url)}
          leadingIcon={<Maximize2 size={12} />}
          className="mt-2 !px-2 !py-1 text-[11px]"
        >
          查看详情
        </GlassButton>
        <GlassButton
          variant="ghost"
          size="sm"
          onClick={onOpenRerun}
          leadingIcon={<RotateCcw size={12} />}
          className="mt-1.5 !px-2 !py-1 text-[11px]"
          title="重跑此行（可选二次校验 / 模型覆盖）"
        >
          重跑
        </GlassButton>
      </div>

      <div className="min-w-0 space-y-2">
        <div className="flex items-center gap-2">
          {row.logo_url && (
            <Thumb
              src={`/api/img?u=${encodeURIComponent(row.logo_url)}`}
              label="LOGO"
            />
          )}
          <div className="min-w-0 text-xs text-slate-500 dark:text-slate-400">
            <div className="font-medium text-slate-700 dark:text-slate-200">
              使用证据 {row.evidence_urls.length} 张
            </div>
            <div className="truncate font-mono text-[11px] text-aurora-magenta">
              {row.best_evidence_url
                ? `BEST EV${Math.max(1, row.evidence_urls.indexOf(row.best_evidence_url) + 1)}`
                : "NO BEST"}
            </div>
          </div>
        </div>
        <EvidenceStrip row={row} onOpenDetail={onOpenDetail} />
      </div>

      {/* controls */}
      <div className="flex min-w-0 flex-col gap-2.5">
        <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
          自动状态{" "}
          <span className="font-mono text-aurora-cyan">{row.status}</span>
        </div>
        <MetricChips row={row} />
        <GlassPill value={current} onChange={(s) => onSet(s, notes)} />
        <GlassInput
          placeholder="备注…"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          onBlur={() => {
            if ((notes ?? "") !== (row.notes ?? "")) {
              onSet((current ?? "needs_review") as PillStatus, notes);
            }
          }}
        />
      </div>
    </GlassCard>
  );
}

function MetricChips({ row }: { row: JobRow }) {
  const conf = row.best_confidence;
  const clar = row.best_clarity;
  const comp = row.best_completeness;
  const iso = row.best_isolation;
  const fb = row.best_fallback_model;
  // hide the whole strip when no scalars are present (legacy rows / failed rows)
  if (conf == null && clar == null && comp == null && iso == null && !fb) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Chip label="conf" value={conf} accent="cyan" />
      <Chip label="clar" value={clar} />
      <Chip label="comp" value={comp} />
      <Chip label="iso" value={iso} />
      {fb && (
        <span
          className="inline-flex items-center gap-1 rounded-full border border-amber-300/50 bg-amber-300/10 px-2 py-0.5 text-[11px] font-medium text-amber-700 backdrop-blur-md dark:border-amber-300/30 dark:bg-amber-300/10 dark:text-amber-300"
          title={`primary 模型拒绝 (< 28 px)，回落到 ${fb}`}
        >
          <span className="uppercase tracking-wider text-amber-600 dark:text-amber-400">回落</span>
          <span className="font-mono">{fb}</span>
        </span>
      )}
      {row.best_reason && (
        <span
          className="max-w-full truncate rounded-full border border-white/40 bg-white/40 px-2 py-0.5 text-[11px] text-slate-700 backdrop-blur-md dark:border-white/10 dark:bg-white/5 dark:text-slate-300"
          title={row.best_reason}
        >
          {row.best_reason}
        </span>
      )}
    </div>
  );
}

function EvidenceStrip({
  row,
  onOpenDetail,
}: {
  row: JobRow;
  onOpenDetail: (evidenceUrl?: string | null) => void;
}) {
  if (row.evidence_urls.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-white/40 bg-white/20 px-3 py-4 text-center text-xs text-slate-500 dark:border-white/10 dark:bg-white/5">
        无使用证据
      </div>
    );
  }
  return (
    <div className="flex gap-2 overflow-x-auto pb-2 pr-1">
      {row.evidence_urls.map((url, i) => (
        <EvidenceCandidate
          key={`${url}-${i}`}
          row={row}
          url={url}
          index={i}
          meta={row.match_meta?.[url]}
          cropPath={row.all_crops?.[url] ?? null}
          isBest={url === row.best_evidence_url}
          onOpenDetail={() => onOpenDetail(url)}
        />
      ))}
    </div>
  );
}

function EvidenceCandidate({
  row,
  url,
  index,
  meta,
  cropPath,
  isBest,
  onOpenDetail,
}: {
  row: JobRow;
  url: string;
  index: number;
  meta: EvidenceMeta | undefined;
  cropPath: string | null;
  isBest: boolean;
  onOpenDetail: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onOpenDetail}
      title={url}
      className={cn(
        "group min-w-[148px] rounded-2xl border bg-white/35 p-2 text-left backdrop-blur-md transition hover:-translate-y-0.5 hover:bg-white/50 dark:bg-white/5 dark:hover:bg-white/10",
        isBest
          ? "border-aurora-cyan shadow-[0_0_0_1px_rgba(56,189,248,0.45)]"
          : "border-white/40 dark:border-white/10",
      )}
    >
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <span className="font-mono text-[11px] font-semibold text-slate-700 dark:text-slate-200">
          EV{index + 1}
        </span>
        <EvidenceBadge meta={meta} isBest={isBest} />
      </div>
      <div className="grid grid-cols-2 gap-1.5">
        <MiniImage src={`/api/img?u=${encodeURIComponent(url)}`} label="证据" />
        {cropPath ? (
          <MiniImage
            src={`/api/jobs/${row.job_id}/file?p=${encodeURIComponent(cropPath)}`}
            label="裁切"
          />
        ) : (
          <div className="flex h-20 items-center justify-center rounded-xl border border-dashed border-white/40 bg-white/20 text-[10px] text-slate-400 dark:border-white/10 dark:bg-white/5">
            无裁切
          </div>
        )}
      </div>
      <div className="mt-1.5 flex items-center gap-1 text-[10px] text-slate-500 dark:text-slate-400">
        {typeof meta?.confidence === "number" && (
          <span className="font-mono text-aurora-cyan">
            conf {meta.confidence.toFixed(2)}
          </span>
        )}
        {meta?.fit && (
          <span className="truncate rounded-full bg-white/45 px-1.5 py-0.5 dark:bg-white/10">
            {meta.fit}
          </span>
        )}
      </div>
    </button>
  );
}

function EvidenceBadge({
  meta,
  isBest,
}: {
  meta: EvidenceMeta | undefined;
  isBest: boolean;
}) {
  if (isBest) {
    return (
      <span className="rounded-full bg-aurora-cyan/85 px-1.5 py-0.5 text-[9px] font-semibold text-slate-950">
        BEST
      </span>
    );
  }
  if (meta?.verified === false) {
    return (
      <span
        className="rounded-full bg-amber-500/90 px-1.5 py-0.5 text-[9px] font-semibold text-white"
        title={meta.verify_reason ?? undefined}
      >
        拒绝
      </span>
    );
  }
  if (meta?.sanity_rejected || meta?.error) {
    return (
      <span
        className="rounded-full bg-rose-500/90 px-1.5 py-0.5 text-[9px] font-semibold text-white"
        title={meta.sanity_rejected ?? meta.error ?? undefined}
      >
        异常
      </span>
    );
  }
  if (meta?.found) {
    return (
      <span className="rounded-full bg-emerald-500/90 px-1.5 py-0.5 text-[9px] font-semibold text-white">
        命中
      </span>
    );
  }
  return (
    <span className="rounded-full bg-slate-500/70 px-1.5 py-0.5 text-[9px] font-semibold text-white">
      未中
    </span>
  );
}

function MiniImage({ src, label }: { src: string; label: string }) {
  return (
    <div className="relative h-20 overflow-hidden rounded-xl border border-white/40 bg-white/40 dark:border-white/10 dark:bg-white/5">
      <img
        src={src}
        alt={label}
        className="h-full w-full object-contain transition-transform duration-300 group-hover:scale-105"
        loading="lazy"
      />
      <span className="absolute bottom-1 left-1 rounded bg-black/55 px-1 py-0.5 text-[9px] font-semibold text-white">
        {label}
      </span>
    </div>
  );
}

function Chip({
  label,
  value,
  accent = "magenta",
}: {
  label: string;
  value: number | null | undefined;
  accent?: "magenta" | "cyan";
}) {
  if (value == null) return null;
  const accentText =
    accent === "cyan" ? "text-aurora-cyan" : "text-aurora-magenta";
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border border-white/40 bg-white/40 px-2 py-0.5 text-[11px] font-medium text-slate-700 backdrop-blur-md dark:border-white/10 dark:bg-white/5 dark:text-slate-200",
      )}
    >
      <span className="uppercase tracking-wider text-slate-500 dark:text-slate-400">
        {label}
      </span>
      <span className={cn("font-mono", accentText)}>{value.toFixed(2)}</span>
    </span>
  );
}

function Thumb({
  src,
  label,
  accent = "magenta",
}: {
  src: string;
  label: string;
  accent?: "magenta" | "cyan" | "violet";
}) {
  const accentRing =
    accent === "cyan"
      ? "shadow-[0_0_0_1px_rgba(56,189,248,0.4)]"
      : accent === "violet"
        ? "shadow-[0_0_0_1px_rgba(167,139,250,0.4)]"
        : "shadow-[0_0_0_1px_rgba(240,171,252,0.4)]";
  return (
    <div className="group relative">
      <div
        className={cn(
          "h-24 w-24 overflow-hidden rounded-2xl border border-white/40 bg-white/40 backdrop-blur-md dark:border-white/10 dark:bg-white/5",
          accentRing,
        )}
      >
        <img
          src={src}
          alt={label}
          className="h-full w-full object-contain transition-transform duration-300 group-hover:scale-105"
          loading="lazy"
        />
      </div>
      <span className="absolute bottom-1 left-1 rounded-md bg-black/55 px-1.5 py-0.5 text-[10px] font-semibold text-white">
        {label}
      </span>
    </div>
  );
}
