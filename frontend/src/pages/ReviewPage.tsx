import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, Filter, Maximize2, RotateCcw } from "lucide-react";
import { api, type JobRow } from "@/lib/api";
import { setSession } from "@/lib/session";
import { GlassButton } from "@/components/GlassButton";
import { GlassCard } from "@/components/GlassCard";
import { GlassInput, GlassSelect } from "@/components/GlassInput";
import { GlassPill, type PillStatus } from "@/components/GlassPill";
import { GlassSpinner } from "@/components/GlassSpinner";
import { RowDetailModal } from "@/components/RowDetailModal";
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
            onOpenDetail={() => setDetailRowId(r.id)}
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
            onClose={() => setDetailRowId(null)}
            onSet={(s, n) =>
              setStatus.mutate({ rowId: r.id, status: s, notes: n })
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
}: {
  row: JobRow;
  onSet: (status: PillStatus, notes?: string) => void;
  onOpenDetail: () => void;
}) {
  const [notes, setNotes] = useState(row.notes ?? "");
  const current = (row.human_status ?? null) as PillStatus | null;
  const evidencePreview = row.evidence_urls.slice(0, 3);

  return (
    <GlassCard
      className={cn("p-4 flex flex-wrap items-start gap-5", statusTone(row.human_status ?? row.status))}
    >
      {/* row id */}
      <div className="shrink-0 w-20 text-xs text-slate-500 dark:text-slate-400">
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
          onClick={onOpenDetail}
          leadingIcon={<Maximize2 size={12} />}
          className="mt-2 !px-2 !py-1 text-[11px]"
        >
          查看详情
        </GlassButton>
      </div>

      {/* thumbnails */}
      <div className="flex shrink-0 flex-wrap gap-2">
        {row.logo_url && (
          <Thumb
            src={`/api/img?u=${encodeURIComponent(row.logo_url)}`}
            label="LOGO"
          />
        )}
        {row.best_crop_path && (
          <Thumb
            src={`/api/jobs/${row.job_id}/file?p=${encodeURIComponent(row.best_crop_path)}`}
            label="CROP"
            accent="cyan"
          />
        )}
        {evidencePreview.map((u, i) => (
          <Thumb
            key={i}
            src={`/api/img?u=${encodeURIComponent(u)}`}
            label={`EV${i + 1}`}
            accent="violet"
          />
        ))}
      </div>

      {/* controls */}
      <div className="flex min-w-[260px] flex-1 flex-col gap-2.5">
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
