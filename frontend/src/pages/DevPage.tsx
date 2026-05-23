import { useQuery } from "@tanstack/react-query";
import { ChevronDown, FlaskConical, Trophy } from "lucide-react";
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

// ---- Leaderboard helpers ---------------------------------------------------

interface LeaderRow {
  key: string;
  model: string;
  prompt_version: string;
  selection_accuracy: number | null;
  mean_iou: number | null;
  mean_ssim: number | null;
  cost_usd: number | null;
  created_at: number;
  run_id: number;
}

function buildLeaderboard(
  runs: { id: number; prompt_version: string; model: string; metrics: Metrics; created_at: number }[],
): LeaderRow[] {
  // One row per (model, prompt_version) tuple — keep the most recent eval-run
  // for that tuple since later runs reflect later code/prompt fixes.
  const byKey = new Map<string, LeaderRow>();
  for (const r of runs) {
    const key = `${r.model}::${r.prompt_version}`;
    const sel = pickNumber(r.metrics, "selection_accuracy");
    const iou = pickNumber(r.metrics, "mean_iou");
    const ssim = pickNumber(r.metrics, "mean_ssim");
    const cost = pickNumber(r.metrics, "cost_usd");
    const incoming: LeaderRow = {
      key,
      model: r.model,
      prompt_version: r.prompt_version,
      selection_accuracy: sel ?? null,
      mean_iou: iou ?? null,
      mean_ssim: ssim ?? null,
      cost_usd: cost ?? null,
      created_at: r.created_at,
      run_id: r.id,
    };
    const prev = byKey.get(key);
    if (!prev || incoming.created_at > prev.created_at) byKey.set(key, incoming);
  }
  return Array.from(byKey.values()).sort((a, b) => {
    // Primary: selection_accuracy desc. Treat null as -Infinity so unmeasured
    // tuples sink to the bottom instead of accidentally winning.
    const sa = a.selection_accuracy ?? -Infinity;
    const sb = b.selection_accuracy ?? -Infinity;
    if (sb !== sa) return sb - sa;
    // Tie-break: higher mean_ssim, then most recent.
    const sma = a.mean_ssim ?? -Infinity;
    const smb = b.mean_ssim ?? -Infinity;
    if (smb !== sma) return smb - sma;
    return b.created_at - a.created_at;
  });
}

function fmtPct(v: number | null): string {
  return v === null ? "—" : `${(v * 100).toFixed(1)}%`;
}
function fmt3(v: number | null): string {
  return v === null ? "—" : v.toFixed(3);
}
function fmtCost(v: number | null): string {
  return v === null ? "—" : `$${v.toFixed(3)}`;
}

export function DevPage() {
  const { data: runs, isLoading } = useQuery({
    queryKey: ["eval_runs"],
    queryFn: () => api.evalRuns(),
  });

  const leaderboard = runs ? buildLeaderboard(runs) : [];

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

      {leaderboard.length > 0 && (
        <GlassCard className="space-y-3 p-5">
          <div className="flex items-center gap-2">
            <Trophy size={16} className="text-aurora-magenta" />
            <h2 className="text-base font-semibold">模型 × Prompt 排行榜</h2>
            <span className="text-xs text-slate-500 dark:text-slate-400">
              按选择准确率排序 · 同一 (model, prompt) 取最近一次评测
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-sm">
              <thead className="text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                <tr className="border-b border-white/40 dark:border-white/10">
                  <th className="py-2 pr-3 text-left font-medium">#</th>
                  <th className="py-2 pr-3 text-left font-medium">Model</th>
                  <th className="py-2 pr-3 text-left font-medium">Prompt</th>
                  <th className="py-2 pr-3 text-right font-medium">Sel-acc</th>
                  <th className="py-2 pr-3 text-right font-medium">Mean IoU</th>
                  <th className="py-2 pr-3 text-right font-medium">Mean SSIM</th>
                  <th className="py-2 pr-3 text-right font-medium">Cost (est)</th>
                  <th className="py-2 pl-2 text-right font-medium">Created</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((row, idx) => {
                  const isTop = idx === 0 && row.selection_accuracy !== null;
                  return (
                    <tr
                      key={row.key}
                      className={
                        "border-b border-white/30 last:border-b-0 dark:border-white/5 " +
                        (isTop
                          ? "bg-gradient-to-r from-aurora-magenta/15 via-aurora-cyan/10 to-transparent"
                          : "")
                      }
                      style={
                        isTop
                          ? { boxShadow: "inset 0 1px 0 rgba(255,255,255,0.55)" }
                          : undefined
                      }
                    >
                      <td className="py-2 pr-3 align-middle font-mono text-xs tabular-nums">
                        {isTop ? "🏆" : idx + 1}
                      </td>
                      <td className="py-2 pr-3 align-middle">
                        <span className="font-mono text-[12px]">{row.model}</span>
                      </td>
                      <td className="py-2 pr-3 align-middle">
                        <span className="text-aurora-magenta">{row.prompt_version}</span>
                      </td>
                      <td className="py-2 pr-3 text-right align-middle font-mono tabular-nums">
                        {fmtPct(row.selection_accuracy)}
                      </td>
                      <td className="py-2 pr-3 text-right align-middle font-mono tabular-nums">
                        {fmt3(row.mean_iou)}
                      </td>
                      <td className="py-2 pr-3 text-right align-middle font-mono tabular-nums">
                        {fmt3(row.mean_ssim)}
                      </td>
                      <td className="py-2 pr-3 text-right align-middle font-mono tabular-nums">
                        {fmtCost(row.cost_usd)}
                      </td>
                      <td className="py-2 pl-2 text-right align-middle font-mono text-[11px] text-slate-500 dark:text-slate-400">
                        {new Date(row.created_at * 1000).toLocaleDateString()}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-[11px] text-slate-500 dark:text-slate-400">
            Cost 是 `cost_estimate()` 估算值（已按 provider 校准 × 0.02），仅作横向参考，不等于真实账单。
          </p>
        </GlassCard>
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
