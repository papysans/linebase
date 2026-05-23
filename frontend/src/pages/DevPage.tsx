import { useQuery } from "@tanstack/react-query";
import { ChevronDown, FlaskConical } from "lucide-react";
import { api } from "@/lib/api";
import { GlassCard } from "@/components/GlassCard";
import { GlassSpinner } from "@/components/GlassSpinner";

type Metrics = Record<string, unknown>;

interface MetricSpec {
  key: string;
  label: string;
  format?: (v: number) => string;
}

const HEADLINE_METRICS: MetricSpec[] = [
  { key: "selection_accuracy", label: "选择准确率", format: (v) => `${(v * 100).toFixed(1)}%` },
  { key: "mean_iou", label: "平均 IoU", format: (v) => v.toFixed(3) },
  { key: "mean_ssim", label: "平均 SSIM", format: (v) => v.toFixed(3) },
  { key: "pass_rate", label: "通过率", format: (v) => `${(v * 100).toFixed(1)}%` },
  { key: "cost_usd", label: "API 成本", format: (v) => `$${v.toFixed(3)}` },
  { key: "n_samples", label: "样本数", format: (v) => String(Math.round(v)) },
];

function pickNumber(m: Metrics, key: string): number | undefined {
  const v = m[key];
  return typeof v === "number" ? v : undefined;
}

export function DevPage() {
  const { data: runs, isLoading } = useQuery({
    queryKey: ["eval_runs"],
    queryFn: () => api.evalRuns(),
  });

  return (
    <div className="space-y-5">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">Dev · 评估</h1>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          每次自迭代轮的 prompt 版本、模型、metric 都记录在此。
        </p>
      </header>

      {isLoading && (
        <div className="flex items-center gap-2 text-slate-500">
          <GlassSpinner /> 加载中…
        </div>
      )}

      <div className="space-y-3">
        {runs?.map((r) => {
          const m = (r.metrics ?? {}) as Metrics;
          return (
            <GlassCard key={r.id} className="p-5 space-y-4">
              <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
                <div className="flex items-center gap-2">
                  <FlaskConical size={14} className="text-aurora-cyan" />
                  <span className="font-mono text-[13px]">#{r.id}</span>
                </div>
                <span className="text-sm font-semibold">
                  prompt{" "}
                  <span className="text-aurora-magenta">{r.prompt_version}</span>
                </span>
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  模型 <span className="font-mono">{r.model}</span>
                </span>
                <span className="ml-auto text-xs text-slate-500 dark:text-slate-400">
                  {new Date(r.created_at * 1000).toLocaleString()}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
                {HEADLINE_METRICS.map((spec) => {
                  const v = pickNumber(m, spec.key);
                  return (
                    <div
                      key={spec.key}
                      className="rounded-2xl border border-white/40 bg-white/30 px-3 py-2.5 dark:border-white/10 dark:bg-white/5"
                    >
                      <div className="text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                        {spec.label}
                      </div>
                      <div className="mt-0.5 font-mono text-base tabular-nums">
                        {v === undefined ? (
                          <span className="text-slate-400">—</span>
                        ) : (
                          (spec.format ?? String)(v)
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              <details className="group">
                <summary className="flex cursor-pointer items-center gap-1 text-xs font-medium text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200">
                  <ChevronDown
                    size={14}
                    className="transition-transform group-open:rotate-180"
                  />
                  原始 metrics JSON
                </summary>
                <pre className="mt-2 max-h-72 overflow-auto rounded-2xl border border-white/40 bg-white/30 p-3 font-mono text-[11px] leading-relaxed text-slate-700 dark:border-white/10 dark:bg-white/5 dark:text-slate-300">
                  {JSON.stringify(m, null, 2)}
                </pre>
              </details>
            </GlassCard>
          );
        })}
        {runs && runs.length === 0 && (
          <GlassCard className="p-6 text-center text-sm text-slate-500">
            还没有评估记录。
          </GlassCard>
        )}
      </div>
    </div>
  );
}
