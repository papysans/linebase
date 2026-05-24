/**
 * Per-row rerun dialog.
 *
 * Surfaces the two opt-in knobs that ship with POST /api/jobs/{id}/rows/{rowId}/rerun:
 *   - verify-loop (checkbox)  — runs the extra confirmation pass on every
 *     evidence, catching brand-recognition-shortcut failures at ~2x cost.
 *   - model override (dropdown) — swap to a different vision model just for
 *     this rerun. The override sticks on the job (no transient state) — pick
 *     again to switch back.
 *
 * Wire shape mirrors the existing GlassPill / GlassButton patterns. Uses a
 * portal so it can render on top of RowDetailModal without z-index fights.
 */
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { RotateCcw, X } from "lucide-react";
import type { JobRow, ModelOption } from "@/lib/api";
import { GlassButton } from "@/components/GlassButton";
import { GlassSelect } from "@/components/GlassInput";
import { GlassSpinner } from "@/components/GlassSpinner";

interface RowRerunDialogProps {
  row: JobRow;
  models?: ModelOption[];
  defaultModel?: string;
  pending: boolean;
  onClose: () => void;
  onSubmit: (opts: { verify?: boolean; model?: string | null }) => void;
}

// "" → keep current job model; otherwise the value is the model id to send.
const KEEP_MODEL = "";

export function RowRerunDialog({
  row,
  models,
  defaultModel,
  pending,
  onClose,
  onSubmit,
}: RowRerunDialogProps) {
  const [verify, setVerify] = useState(false);
  const [model, setModel] = useState<string>(KEEP_MODEL);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape" && !pending) onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, pending]);

  const handleSubmit = () => {
    onSubmit({
      verify,
      model: model === KEEP_MODEL ? null : model,
    });
  };

  return createPortal(
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-slate-950/60 p-4 backdrop-blur-sm"
      onClick={pending ? undefined : onClose}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="glass-pane relative flex w-full max-w-md flex-col gap-4 p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-3">
          <div className="space-y-0.5">
            <h2 className="text-lg font-semibold tracking-tight">
              重跑此行
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              行 {row.row_index} · 申请号{" "}
              <span className="font-mono text-aurora-magenta">
                {row.appno ?? "-"}
              </span>
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="关闭"
            disabled={pending}
            className="glass-button glass-button--ghost glass-button--sm"
          >
            <X size={14} />
          </button>
        </div>

        <label className="flex items-start gap-2 rounded-2xl border border-white/40 bg-white/30 p-3 text-sm dark:border-white/10 dark:bg-white/5">
          <input
            type="checkbox"
            className="mt-0.5 h-4 w-4 accent-aurora-magenta"
            checked={verify}
            onChange={(e) => setVerify(e.target.checked)}
          />
          <div className="space-y-0.5">
            <div className="font-medium text-slate-800 dark:text-slate-100">
              二次校验（更慢更准）
            </div>
            <div className="text-[11px] leading-snug text-slate-500 dark:text-slate-400">
              每个证据裁完后再调一次模型确认裁切真的包含该商标形状。成本翻倍，
              但能拦截"品牌识别捷径"的假阳性（例如把 Miami Heat 火球当成 NBA
              球员剪影）。
            </div>
          </div>
        </label>

        <div className="space-y-1.5">
          <label className="text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">
            覆盖模型（仅此次）
          </label>
          <GlassSelect value={model} onChange={(e) => setModel(e.target.value)}>
            <option value={KEEP_MODEL}>
              保持当前 {defaultModel ? `（默认 ${defaultModel}）` : ""}
            </option>
            {models?.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </GlassSelect>
          <p className="text-[11px] leading-snug text-slate-500 dark:text-slate-400">
            注意：模型覆盖会持久化到 job 上，下次重跑请重新选择以切回。
          </p>
        </div>

        <div className="flex items-center justify-end gap-2 pt-1">
          <GlassButton
            variant="ghost"
            size="sm"
            onClick={onClose}
            disabled={pending}
          >
            取消
          </GlassButton>
          <GlassButton
            variant="primary"
            size="sm"
            disabled={pending}
            onClick={handleSubmit}
            leadingIcon={
              pending ? <GlassSpinner size={14} /> : <RotateCcw size={14} />
            }
          >
            {pending ? "提交中…" : "开始重跑"}
          </GlassButton>
        </div>
      </div>
    </div>,
    document.body,
  );
}
