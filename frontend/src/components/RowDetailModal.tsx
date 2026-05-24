/**
 * Row detail modal — full-size side-by-side comparison for hand review.
 *
 * Layout: three columns (logo line-art, evidence with bbox overlay, crop result)
 * + a footer with the LLM reason/metrics, OK/BAD/NEEDS_REVIEW pill, and a notes
 * field so the reviewer can decide and save without closing.
 *
 * The bbox overlay draws an absolutely-positioned rectangle on top of the
 * evidence image. We measure the rendered image dimensions on load and scale
 * the LLM bbox (which is in original pixel coords) to display coords.
 *
 * Implementation notes
 * - Uses createPortal so the modal isn't trapped by the table's stacking
 *   context. Closes on Esc + outside-click.
 * - No external modal library — ~250 lines of pure React. Tradeoff: no focus
 *   trap and no scroll-lock helpers; acceptable for a single-user local tool.
 */
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";
import type { JobRow } from "@/lib/api";
import { GlassPill, type PillStatus } from "@/components/GlassPill";
import { GlassInput } from "@/components/GlassInput";
import { cn } from "@/lib/cn";

interface RowDetailModalProps {
  row: JobRow;
  onClose: () => void;
  onSet: (status: PillStatus, notes?: string) => void;
}

export function RowDetailModal({ row, onClose, onSet }: RowDetailModalProps) {
  // The center column starts on the LLM-chosen "best" evidence so the user
  // immediately sees what the model picked. Clicking thumbnails in the strip
  // swaps the center column.
  const [activeUrl, setActiveUrl] = useState<string>(
    row.best_evidence_url || row.evidence_urls[0] || "",
  );
  const [notes, setNotes] = useState(row.notes ?? "");
  const current = (row.human_status ?? null) as PillStatus | null;

  // The overlay rectangle is positioned over the SHOWN image, so we need the
  // image's display dimensions. We re-read them on every load + on viewport
  // resize, because the image is `max-h-[70vh]` and reflows with the modal.
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imgBox, setImgBox] = useState<{
    w: number;
    h: number;
    naturalW: number;
    naturalH: number;
  } | null>(null);

  // Per-evidence bbox lookup: only the chosen-best evidence has a bbox on
  // the wire (`best_bbox`). If the user swaps to another evidence in the
  // strip, we have no bbox for it — the overlay simply doesn't render.
  const activeBbox =
    activeUrl === row.best_evidence_url ? row.best_bbox || null : null;

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Measure on image load + on resize. We can't compute on mount because the
  // image might not be decoded yet.
  function measure() {
    const el = imgRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    setImgBox({
      w: rect.width,
      h: rect.height,
      naturalW: el.naturalWidth || 1,
      naturalH: el.naturalHeight || 1,
    });
  }
  useEffect(() => {
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);
  // Re-measure when the displayed image changes.
  useEffect(() => {
    setImgBox(null);
  }, [activeUrl]);

  const overlay = activeBbox && imgBox ? scaleBbox(activeBbox, imgBox) : null;

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/55 p-4 backdrop-blur-sm"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="glass-pane relative flex max-h-[92vh] w-full max-w-7xl flex-col gap-4 overflow-hidden p-5"
        onClick={(e) => e.stopPropagation()}
      >
        {/* header */}
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <h2 className="text-xl font-semibold tracking-tight">
              行 {row.row_index} · 申请号{" "}
              <span className="font-mono text-aurora-magenta">
                {row.appno ?? "-"}
              </span>
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              自动状态{" "}
              <span className="font-mono text-aurora-cyan">{row.status}</span>{" "}
              · LLM 选中{" "}
              <span className="font-mono">
                {row.best_evidence_url ? "evidence" : "无"}
              </span>
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="关闭"
            className="glass-button glass-button--ghost glass-button--sm"
          >
            <X size={14} />
          </button>
        </div>

        {/* three columns */}
        <div className="grid flex-1 min-h-0 grid-cols-1 gap-4 overflow-y-auto md:grid-cols-3">
          <Column title="LOGO 线稿">
            {row.logo_url ? (
              <PlainImage
                src={`/api/img?u=${encodeURIComponent(row.logo_url)}`}
                alt="logo"
              />
            ) : (
              <Empty label="无 logo URL" />
            )}
          </Column>

          <Column
            title="使用证据"
            subtitle={
              activeUrl === row.best_evidence_url
                ? "LLM 选中（含 bbox）"
                : "其他证据"
            }
          >
            {activeUrl ? (
              <div className="relative">
                <img
                  ref={imgRef}
                  src={`/api/img?u=${encodeURIComponent(activeUrl)}`}
                  alt="evidence"
                  onLoad={measure}
                  className="block max-h-[70vh] w-auto max-w-full rounded-2xl border border-white/40 bg-white/40 object-contain dark:border-white/10 dark:bg-white/5"
                />
                {overlay && (
                  <div
                    className="pointer-events-none absolute border-2 border-aurora-magenta/90 shadow-[0_0_0_1px_rgba(15,23,42,0.4)]"
                    style={{
                      left: overlay.left,
                      top: overlay.top,
                      width: overlay.width,
                      height: overlay.height,
                    }}
                  >
                    <span className="absolute -top-5 left-0 rounded-md bg-aurora-magenta/90 px-1.5 py-0.5 text-[10px] font-semibold text-white">
                      LLM bbox
                    </span>
                  </div>
                )}
              </div>
            ) : (
              <Empty label="无证据" />
            )}
          </Column>

          <Column title="裁切结果">
            {row.best_crop_path ? (
              <PlainImage
                src={`/api/jobs/${row.job_id}/file?p=${encodeURIComponent(
                  row.best_crop_path,
                )}`}
                alt="crop"
              />
            ) : (
              <Empty label="未生成裁切" />
            )}
          </Column>
        </div>

        {/* footer */}
        <div className="flex flex-col gap-3 border-t border-white/30 pt-3 dark:border-white/10">
          {row.best_reason && (
            <p className="text-sm text-slate-700 dark:text-slate-300">
              <span className="mr-2 text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                reason
              </span>
              {row.best_reason}
            </p>
          )}
          <Metrics row={row} />

          {/* other evidences strip */}
          {row.evidence_urls.length > 1 && (
            <div>
              <div className="mb-1.5 text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                其他证据 ({row.evidence_urls.length})
              </div>
              <div className="flex flex-wrap gap-2">
                {row.evidence_urls.map((u, i) => {
                  const isActive = u === activeUrl;
                  const isBest = u === row.best_evidence_url;
                  return (
                    <button
                      key={i}
                      type="button"
                      onClick={() => setActiveUrl(u)}
                      title={u}
                      className={cn(
                        "h-16 w-16 overflow-hidden rounded-xl border bg-white/40 backdrop-blur-md transition dark:bg-white/5",
                        isActive
                          ? "border-aurora-magenta shadow-[0_0_0_2px_rgba(240,171,252,0.6)]"
                          : isBest
                            ? "border-aurora-cyan/70"
                            : "border-white/40 dark:border-white/10",
                      )}
                    >
                      <img
                        src={`/api/img?u=${encodeURIComponent(u)}`}
                        alt={`ev${i + 1}`}
                        loading="lazy"
                        className="h-full w-full object-contain"
                      />
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          <div className="flex flex-wrap items-center gap-3">
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
              className="min-w-[260px] flex-1"
            />
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}

function scaleBbox(
  bbox: [number, number, number, number],
  img: { w: number; h: number; naturalW: number; naturalH: number },
) {
  // The image uses object-contain inside a flexible container, so its rendered
  // box matches naturalW × naturalH scaled by the same factor on both axes.
  // We scale bbox coords from natural → displayed pixels.
  const sx = img.w / img.naturalW;
  const sy = img.h / img.naturalH;
  const [x1, y1, x2, y2] = bbox;
  return {
    left: x1 * sx,
    top: y1 * sy,
    width: Math.max(2, (x2 - x1) * sx),
    height: Math.max(2, (y2 - y1) * sy),
  };
}

function Column({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
          {title}
        </h3>
        {subtitle && (
          <span className="text-[11px] text-slate-500 dark:text-slate-400">
            {subtitle}
          </span>
        )}
      </div>
      <div className="flex flex-1 items-center justify-center overflow-auto">
        {children}
      </div>
    </div>
  );
}

function PlainImage({ src, alt }: { src: string; alt: string }) {
  return (
    <img
      src={src}
      alt={alt}
      className="block max-h-[70vh] w-auto max-w-full rounded-2xl border border-white/40 bg-white/40 object-contain dark:border-white/10 dark:bg-white/5"
    />
  );
}

function Empty({ label }: { label: string }) {
  return (
    <div className="flex h-40 w-full items-center justify-center rounded-2xl border border-dashed border-white/40 bg-white/20 text-xs text-slate-500 dark:border-white/10 dark:bg-white/5">
      {label}
    </div>
  );
}

function Metrics({ row }: { row: JobRow }) {
  const items: { label: string; value: number | null | undefined }[] = [
    { label: "conf", value: row.best_confidence },
    { label: "clar", value: row.best_clarity },
    { label: "comp", value: row.best_completeness },
    { label: "iso", value: row.best_isolation },
  ];
  const present = items.filter((i) => i.value != null);
  if (present.length === 0 && !row.best_fallback_model) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {present.map((i) => (
        <span
          key={i.label}
          className="inline-flex items-center gap-1 rounded-full border border-white/40 bg-white/40 px-2 py-0.5 text-[11px] font-medium text-slate-700 backdrop-blur-md dark:border-white/10 dark:bg-white/5 dark:text-slate-200"
        >
          <span className="uppercase tracking-wider text-slate-500 dark:text-slate-400">
            {i.label}
          </span>
          <span className="font-mono text-aurora-magenta">
            {(i.value as number).toFixed(2)}
          </span>
        </span>
      ))}
      {row.best_fallback_model && (
        <span
          className="inline-flex items-center gap-1 rounded-full border border-amber-300/50 bg-amber-300/10 px-2 py-0.5 text-[11px] font-medium text-amber-700 backdrop-blur-md dark:border-amber-300/30 dark:bg-amber-300/10 dark:text-amber-300"
          title={`primary 模型拒绝 (< 28 px)，回落到 ${row.best_fallback_model}`}
        >
          <span className="uppercase tracking-wider text-amber-600 dark:text-amber-400">
            回落
          </span>
          <span className="font-mono">{row.best_fallback_model}</span>
        </span>
      )}
    </div>
  );
}
