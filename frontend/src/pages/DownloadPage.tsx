import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { FileSpreadsheet, ImageDown } from "lucide-react";
import { api } from "@/lib/api";
import { setSession } from "@/lib/session";
import { GlassCard } from "@/components/GlassCard";
import { GlassButton } from "@/components/GlassButton";
import { GlassSpinner } from "@/components/GlassSpinner";

export function DownloadPage() {
  const { jobId = "" } = useParams();
  const { data: job, isLoading } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.getJob(jobId),
  });
  const [onlyOk, setOnlyOk] = useState(false);

  // Sync URL jobId into session — covers deep-link / shared URL / refresh.
  useEffect(() => {
    if (jobId) setSession({ jobId });
  }, [jobId]);

  if (isLoading || !job) {
    return (
      <div className="flex items-center gap-2 text-slate-500">
        <GlassSpinner /> 加载中…
      </div>
    );
  }

  const zipHref = onlyOk
    ? `/api/jobs/${jobId}/images.zip?status=OK`
    : `/api/jobs/${jobId}/images.zip`;

  return (
    <div className="mx-auto max-w-2xl space-y-5">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">下载结果</h1>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          导出 XLSX 与裁剪后的图片 ZIP。
        </p>
      </header>

      <GlassCard className="p-6 space-y-5">
        <dl className="grid grid-cols-2 gap-y-2 gap-x-4 text-sm">
          <dt className="text-slate-500 dark:text-slate-400">任务</dt>
          <dd className="font-mono text-aurora-magenta">{job.id}</dd>

          <dt className="text-slate-500 dark:text-slate-400">状态</dt>
          <dd className="font-mono">{job.status}</dd>

          <dt className="text-slate-500 dark:text-slate-400">完成</dt>
          <dd>
            <span className="font-mono tabular-nums">
              {job.done_rows} / {job.total_rows}
            </span>
          </dd>

          <dt className="text-slate-500 dark:text-slate-400">估计成本</dt>
          <dd className="font-mono">${job.cost_usd.toFixed(3)}</dd>
        </dl>

        <label className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
          <input
            type="checkbox"
            checked={onlyOk}
            onChange={(e) => setOnlyOk(e.target.checked)}
            className="h-4 w-4 accent-aurora-magenta"
          />
          ZIP 仅包含 OK 状态的图片
        </label>

        <div className="flex flex-wrap gap-3">
          <a href={`/api/jobs/${jobId}/xlsx`}>
            <GlassButton
              variant="primary"
              size="lg"
              leadingIcon={<FileSpreadsheet size={16} />}
            >
              下载结果 XLSX
            </GlassButton>
          </a>
          <a href={zipHref}>
            <GlassButton
              variant="success"
              size="lg"
              leadingIcon={<ImageDown size={16} />}
            >
              下载图片 ZIP
            </GlassButton>
          </a>
        </div>

        <div className="rounded-2xl border border-white/40 bg-white/30 p-3 text-xs text-slate-500 dark:border-white/10 dark:bg-white/5 dark:text-slate-400">
          图片以{" "}
          <span className="rounded bg-white/50 px-1.5 py-0.5 font-mono text-[11px] dark:bg-white/10">
            &lt;申请号&gt;_&lt;序号&gt;.&lt;ext&gt;
          </span>{" "}
          命名。
        </div>
      </GlassCard>
    </div>
  );
}
