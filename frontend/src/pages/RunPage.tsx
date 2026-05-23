import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Activity, ArrowRight, Play } from "lucide-react";
import { api, type JobRow, type JobSummary } from "@/lib/api";
import { GlassButton } from "@/components/GlassButton";
import { GlassCard } from "@/components/GlassCard";
import { GlassSpinner } from "@/components/GlassSpinner";
import { cn } from "@/lib/cn";

interface SseEvent {
  type: "row_done" | "row_failed" | "progress" | "finished";
  row?: JobRow;
  job?: JobSummary;
  message?: string;
}

const STATUS_LABEL: Record<JobSummary["status"], string> = {
  pending: "待启动",
  running: "运行中",
  paused: "已暂停",
  finished: "已完成",
  failed: "失败",
};

export function RunPage() {
  const { jobId = "" } = useParams();
  const { data: job, refetch } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.getJob(jobId),
    refetchInterval: 2000,
  });
  const [events, setEvents] = useState<SseEvent[]>([]);

  useEffect(() => {
    if (!jobId) return;
    const es = new EventSource(`/api/jobs/${jobId}/events`);
    es.onmessage = (e) => {
      try {
        const ev = JSON.parse(e.data) as SseEvent;
        setEvents((prev) => [...prev.slice(-200), ev]);
        if (ev.type === "row_done" || ev.type === "finished") refetch();
      } catch {
        /* ignore */
      }
    };
    es.onerror = () => es.close();
    return () => es.close();
  }, [jobId, refetch]);

  const start = async () => {
    await api.startJob(jobId);
  };

  const progressPct = useMemo(() => {
    if (!job || !job.total_rows) return 0;
    return Math.round((job.done_rows / job.total_rows) * 100);
  }, [job]);

  if (!job)
    return (
      <div className="flex items-center gap-2 text-slate-500">
        <GlassSpinner /> 加载任务…
      </div>
    );

  return (
    <div className="space-y-5">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">运行任务</h1>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-slate-600 dark:text-slate-400">
          <span>
            任务 <span className="font-mono">{job.id}</span>
          </span>
          <span>·</span>
          <span>
            工作表 <span className="font-mono">{job.sheet_name}</span>
          </span>
          {job.model && (
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

      {/* Progress card */}
      <GlassCard className="p-6 space-y-4">
        <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1">
          <span className="text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            进度
          </span>
          <span className="text-3xl font-semibold tracking-tight tabular-nums">
            {job.done_rows}
            <span className="text-base text-slate-500 dark:text-slate-400">
              {" "}
              / {job.total_rows}
            </span>
          </span>
          <span className="text-sm text-slate-500 dark:text-slate-400">
            {progressPct}% · 状态 {STATUS_LABEL[job.status]} · 估计成本{" "}
            <span className="font-mono">${job.cost_usd.toFixed(3)}</span>
          </span>
          <div className="ml-auto flex items-center gap-2">
            {job.status === "pending" && (
              <GlassButton
                variant="primary"
                onClick={start}
                leadingIcon={<Play size={14} />}
              >
                开始处理
              </GlassButton>
            )}
            {job.status === "finished" && (
              <Link to={`/review/${jobId}`}>
                <GlassButton
                  variant="success"
                  leadingIcon={<ArrowRight size={14} />}
                >
                  前往审查
                </GlassButton>
              </Link>
            )}
          </div>
        </div>

        <div className="glass-progress" role="progressbar" aria-valuenow={progressPct}>
          <div
            className="glass-progress__fill"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </GlassCard>

      {/* Live event log */}
      <GlassCard className="p-0 overflow-hidden">
        <div className="flex items-center gap-2 px-5 py-3 border-b border-white/30 dark:border-white/10">
          <Activity size={14} className="text-aurora-cyan" />
          <span className="text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            事件流
          </span>
          <span className="ml-auto text-xs text-slate-500 dark:text-slate-400">
            {events.length} 条
          </span>
        </div>
        <div className="max-h-[480px] overflow-y-auto px-5 py-3 font-mono text-[12.5px] leading-relaxed">
          {events.length === 0 && (
            <div className="flex items-center gap-2 text-slate-400">
              <GlassSpinner size={14} /> 等待事件…
            </div>
          )}
          {events.map((e, i) => (
            <EventLine key={i} event={e} />
          ))}
        </div>
      </GlassCard>
    </div>
  );
}

function EventLine({ event }: { event: SseEvent }) {
  const tone =
    event.type === "row_failed"
      ? "underglow-bad"
      : event.type === "row_done"
        ? "underglow-ok"
        : event.type === "finished"
          ? "underglow-ok"
          : "";
  const tagColor =
    event.type === "row_failed"
      ? "text-rose-600 dark:text-rose-300"
      : event.type === "row_done"
        ? "text-emerald-600 dark:text-emerald-300"
        : event.type === "finished"
          ? "text-emerald-600 dark:text-emerald-300"
          : "text-slate-500 dark:text-slate-400";

  return (
    <div
      className={cn(
        "mb-1 rounded-xl px-3 py-1.5 transition-colors",
        tone,
      )}
    >
      <span className={cn("font-semibold", tagColor)}>[{event.type}]</span>{" "}
      {event.row ? (
        <span>
          row {event.row.row_index} · appno={" "}
          <span className="text-aurora-magenta">{event.row.appno ?? "-"}</span>{" "}
          · status=
          <span className="text-aurora-cyan">{event.row.status}</span>
        </span>
      ) : (
        <span>{event.message ?? ""}</span>
      )}
    </div>
  );
}
