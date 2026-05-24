// Friendly empty state for /run, /review, /download when the user lands
// without a jobId in the URL AND nothing in `session`. Replaces the old
// "尚未创建任务" wall.
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { ArrowRight, UploadCloud } from "lucide-react";
import { api, type JobSummary } from "@/lib/api";
import { GlassCard } from "@/components/GlassCard";
import { GlassButton } from "@/components/GlassButton";
import { GlassSpinner } from "@/components/GlassSpinner";

type Section = "run" | "review" | "download" | "configure";

const SECTION_LABEL: Record<Section, string> = {
  run: "运行",
  review: "审查",
  download: "下载",
  configure: "配置",
};

interface Props {
  /** Which deep-link page is rendering us. Decides which path we link to. */
  section: Section;
}

function fmtTime(epochSec?: number): string {
  if (!epochSec) return "";
  const d = new Date(epochSec * 1000);
  // YYYY-MM-DD HH:MM — short, readable in zh-CN context
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function statusBadge(s: JobSummary["status"]): string {
  switch (s) {
    case "running":
      return "text-aurora-cyan";
    case "finished":
      return "text-emerald-600 dark:text-emerald-300";
    case "failed":
      return "text-rose-600 dark:text-rose-300";
    default:
      return "text-slate-500 dark:text-slate-400";
  }
}

export function EmptyJobState({ section }: Props) {
  const { data: jobs, isLoading } = useQuery({
    queryKey: ["recent-jobs"],
    queryFn: () => api.listJobs(5),
    // Cheap GET; refresh when the user comes back to this page.
    staleTime: 10_000,
  });

  return (
    <GlassCard className="mx-auto max-w-xl p-6 space-y-4">
      <div className="space-y-1">
        <h2 className="text-lg font-semibold tracking-tight">
          还没有进行中的任务
        </h2>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          先去
          <Link
            to="/"
            className="mx-1 underline decoration-aurora-magenta underline-offset-2 hover:text-aurora-magenta"
          >
            上传
          </Link>
          新建一个，或者打开历史任务前往「{SECTION_LABEL[section]}」：
        </p>
      </div>

      {isLoading && (
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <GlassSpinner size={14} /> 读取最近任务…
        </div>
      )}

      {jobs && jobs.length === 0 && (
        <div className="space-y-3">
          <p className="text-sm text-slate-500 dark:text-slate-400">
            还没有任何任务。
          </p>
          <Link to="/">
            <GlassButton
              variant="primary"
              leadingIcon={<UploadCloud size={14} />}
            >
              去上传
            </GlassButton>
          </Link>
        </div>
      )}

      {jobs && jobs.length > 0 && (
        <ul className="divide-y divide-white/30 dark:divide-white/10">
          {jobs.map((j) => {
            const target =
              section === "configure"
                ? `/configure/${j.upload_id}`
                : `/${section}/${j.id}`;
            return (
              <li key={j.id} className="py-2.5">
                <Link
                  to={target}
                  className="flex items-center gap-3 rounded-2xl px-2 py-1.5 transition-colors hover:bg-white/40 dark:hover:bg-white/5"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-baseline gap-2 text-sm">
                      <span className="font-mono text-aurora-magenta">
                        {j.id}
                      </span>
                      <span className={statusBadge(j.status)}>{j.status}</span>
                      <span className="font-mono tabular-nums text-slate-500 dark:text-slate-400">
                        {j.done_rows}/{j.total_rows}
                      </span>
                    </div>
                    <div className="truncate text-xs text-slate-500 dark:text-slate-400">
                      {j.sheet_name}
                      {j.model ? ` · ${j.model}` : ""}
                      {j.created_at ? ` · ${fmtTime(j.created_at)}` : ""}
                    </div>
                  </div>
                  <ArrowRight
                    size={14}
                    className="shrink-0 text-slate-400 group-hover:text-aurora-cyan"
                  />
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </GlassCard>
  );
}
