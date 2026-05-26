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
 * Per-evidence inspector (2026-05-24)
 * - Every thumbnail in the evidence strip renders ITS OWN bbox overlay (scaled
 *   to the thumb size) + a status badge in the corner. Previously only the
 *   chosen-best evidence got an overlay, leaving the other 19 in a 20-evidence
 *   row visually unannotated even though the LLM had returned a bbox for each
 *   `found=true` evidence. The data was always there in `match_meta` — this is
 *   purely a render gap.
 * - Clicking a thumbnail swaps the center image AND the metric chips AND the
 *   "reason" line so the modal becomes a true per-evidence inspector instead
 *   of a chosen-best-only viewer.
 *
 * Implementation notes
 * - Uses createPortal so the modal isn't trapped by the table's stacking
 *   context. Closes on Esc + outside-click.
 * - No external modal library — ~350 lines of pure React. Tradeoff: no focus
 *   trap and no scroll-lock helpers; acceptable for a single-user local tool.
 */
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { RotateCcw, X } from "lucide-react";
import type { EvidenceMeta, JobRow } from "@/lib/api";
import { GlassButton } from "@/components/GlassButton";
import { GlassPill, type PillStatus } from "@/components/GlassPill";
import { GlassInput } from "@/components/GlassInput";
import { cn } from "@/lib/cn";

interface RowDetailModalProps {
  row: JobRow;
  onClose: () => void;
  onSet: (status: PillStatus, notes?: string) => void;
  initialEvidenceUrl?: string | null;
  /** Optional. When provided, the modal shows a 🔄 重跑 button in the header
   *  that delegates to the parent — the parent owns the rerun dialog state so
   *  the modal stays focused on hand-review. */
  onOpenRerun?: () => void;
}

export function RowDetailModal({
  row,
  onClose,
  onSet,
  initialEvidenceUrl,
  onOpenRerun,
}: RowDetailModalProps) {
  // The center column starts on the LLM-chosen "best" evidence so the user
  // immediately sees what the model picked. Clicking thumbnails in the strip
  // swaps the center column.
  const [activeUrl, setActiveUrl] = useState<string>(
    initialEvidenceUrl || row.best_evidence_url || row.evidence_urls[0] || "",
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

  // Per-evidence meta lookup. Whatever URL is currently centered, we pull its
  // bbox + metrics + reason from `match_meta` so the modal stays in sync.
  // Falls back to the top-level `best_*` fields when `match_meta` is missing
  // (older rows from before the projection was added).
  const meta: EvidenceMeta | undefined = row.match_meta?.[activeUrl];
  const isBest = activeUrl === row.best_evidence_url;
  const activeBbox: [number, number, number, number] | null =
    (meta?.bbox as [number, number, number, number] | undefined | null) ??
    (isBest ? row.best_bbox || null : null);
  const activeReason: string | null = (meta?.reason ?? null) || (isBest ? row.best_reason ?? null : null);
  const activeConfidence: number | null | undefined = meta?.confidence ?? (isBest ? row.best_confidence : null);
  const activeClarity: number | null | undefined = meta?.clarity ?? (isBest ? row.best_clarity : null);
  const activeCompleteness: number | null | undefined =
    meta?.completeness ?? (isBest ? row.best_completeness : null);
  const activeIsolation: number | null | undefined = meta?.isolation ?? (isBest ? row.best_isolation : null);
  const activeFallbackModel: string | null | undefined =
    meta?.fallback_model ?? (isBest ? row.best_fallback_model : null);
  const activeCropPath = row.all_crops?.[activeUrl] || (isBest ? row.best_crop_path : null);

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
        className="glass-pane relative flex h-[92vh] w-full max-w-7xl flex-col gap-3 overflow-hidden p-4"
        onClick={(e) => e.stopPropagation()}
      >
        {/* header */}
        <div className="flex shrink-0 items-center justify-between">
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
          <div className="flex items-center gap-2">
            {onOpenRerun && (
              <GlassButton
                variant="ghost"
                size="sm"
                onClick={onOpenRerun}
                leadingIcon={<RotateCcw size={14} />}
                title="重跑此行（可选二次校验 / 模型覆盖）"
              >
                重跑
              </GlassButton>
            )}
            <button
              type="button"
              onClick={onClose}
              aria-label="关闭"
              className="glass-button glass-button--ghost glass-button--sm"
            >
              <X size={14} />
            </button>
          </div>
        </div>

        {/* three columns */}
        <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 overflow-y-auto md:grid-cols-3 md:overflow-hidden">
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
              isBest
                ? "LLM 选中（含 bbox）"
                : meta?.found
                  ? "其他证据（含 bbox）"
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
                  className="block max-h-[40vh] w-auto max-w-full rounded-2xl border border-white/40 bg-white/40 object-contain dark:border-white/10 dark:bg-white/5"
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
            {activeCropPath ? (
              <PlainImage
                src={`/api/jobs/${row.job_id}/file?p=${encodeURIComponent(
                  activeCropPath,
                )}`}
                alt="crop"
              />
            ) : (
              <Empty label="未生成裁切" />
            )}
          </Column>
        </div>

        {/* footer */}
        <div className="flex shrink-0 flex-col gap-2 border-t border-white/30 pt-2 dark:border-white/10">
          {activeReason && (
            <p
              className="truncate text-sm text-slate-700 dark:text-slate-300"
              title={activeReason}
            >
              <span className="mr-2 text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                reason{isBest ? "" : "（当前证据）"}
              </span>
              {activeReason}
            </p>
          )}
          <ActiveMetrics
            confidence={activeConfidence}
            clarity={activeClarity}
            completeness={activeCompleteness}
            isolation={activeIsolation}
            fallbackModel={activeFallbackModel}
          />

          {/* other evidences strip — every thumb shows its own bbox + status badge */}
          {row.evidence_urls.length > 1 && (
            <div>
              <div className="mb-1.5 text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                所有证据 ({row.evidence_urls.length}) · 点击切换中间大图
              </div>
              <div className="flex max-h-24 gap-2 overflow-x-auto overflow-y-hidden pb-1.5 pr-1">
                {row.evidence_urls.map((u, i) => {
                  const isActive = u === activeUrl;
                  const isThisBest = u === row.best_evidence_url;
                  const thumbMeta: EvidenceMeta | undefined = row.match_meta?.[u];
                  return (
                    <EvidenceThumb
                      key={i}
                      url={u}
                      index={i}
                      meta={thumbMeta}
                      isActive={isActive}
                      isBest={isThisBest}
                      onClick={() => setActiveUrl(u)}
                    />
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

/**
 * One evidence thumbnail in the strip. Renders the image + its own bbox
 * overlay (scaled to thumb size) + a status badge in the top-right corner.
 *
 * The bbox overlay uses the same math as the center image (`scaleBbox`), but
 * we measure on a per-thumb basis since each evidence has different
 * naturalW/H. The badge encodes four cases:
 *   - sanity_rejected → red "空裁"
 *   - verified === false → amber "验证拒 <fit>"
 *   - found === true → green "✓ <conf>"
 *   - found === false → gray "✗"
 */
function EvidenceThumb({
  url,
  index,
  meta,
  isActive,
  isBest,
  onClick,
}: {
  url: string;
  index: number;
  meta: EvidenceMeta | undefined;
  isActive: boolean;
  isBest: boolean;
  onClick: () => void;
}) {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [box, setBox] = useState<{
    w: number;
    h: number;
    naturalW: number;
    naturalH: number;
  } | null>(null);

  function measure() {
    const el = imgRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    setBox({
      w: rect.width,
      h: rect.height,
      naturalW: el.naturalWidth || 1,
      naturalH: el.naturalHeight || 1,
    });
  }

  const bbox = (meta?.bbox as [number, number, number, number] | undefined | null) ?? null;
  const overlay = bbox && box ? scaleBbox(bbox, box) : null;

  // Status badge classification — first match wins.
  let badge: { text: string; cls: string; title?: string } | null = null;
  if (meta?.sanity_rejected) {
    badge = {
      text: "空裁",
      cls: "bg-red-500/90 text-white",
      title: `sanity_rejected: ${meta.sanity_rejected}`,
    };
  } else if (meta?.verified === false) {
    badge = {
      text: `验证拒${meta.fit ? " " + meta.fit : ""}`,
      cls: "bg-amber-500/90 text-white",
      title: meta.verify_reason ?? "verify rejected",
    };
  } else if (meta?.error) {
    badge = {
      text: "错",
      cls: "bg-red-500/90 text-white",
      title: meta.error,
    };
  } else if (meta?.found) {
    const conf = typeof meta.confidence === "number" ? meta.confidence : null;
    badge = {
      text: conf != null ? `✓ ${conf.toFixed(2)}` : "✓",
      cls: "bg-emerald-500/90 text-white",
    };
  } else if (meta?.found === false) {
    badge = { text: "✗", cls: "bg-slate-500/80 text-white" };
  }

  return (
    <button
      type="button"
      onClick={onClick}
      title={url}
      className={cn(
        "relative h-20 w-20 shrink-0 overflow-hidden rounded-xl border bg-white/40 backdrop-blur-md transition dark:bg-white/5",
        isActive
          ? "border-aurora-magenta shadow-[0_0_0_2px_rgba(240,171,252,0.6)]"
          : isBest
            ? "border-aurora-cyan/70"
            : "border-white/40 dark:border-white/10",
      )}
    >
      <img
        ref={imgRef}
        src={`/api/img?u=${encodeURIComponent(url)}`}
        alt={`ev${index + 1}`}
        loading="lazy"
        onLoad={measure}
        className="h-full w-full object-contain"
      />
      {overlay && (
        <div
          className="pointer-events-none absolute border border-aurora-magenta/95"
          style={{
            left: overlay.left,
            top: overlay.top,
            width: overlay.width,
            height: overlay.height,
          }}
        />
      )}
      {badge && (
        <span
          className={cn(
            "pointer-events-none absolute right-1 top-1 rounded-md px-1.5 py-0.5 text-[9px] font-semibold leading-none shadow-sm",
            badge.cls,
          )}
          title={badge.title}
        >
          {badge.text}
        </span>
      )}
    </button>
  );
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
    <div className="flex min-h-0 flex-col gap-2">
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
      <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto">
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
      className="block max-h-[40vh] w-auto max-w-full rounded-2xl border border-white/40 bg-white/40 object-contain dark:border-white/10 dark:bg-white/5"
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

/**
 * Metric chips for the currently-active evidence (not always the
 * chosen-best). Falls back to nothing when the evidence has no metrics yet
 * (e.g. pending row, or evidence the LLM returned `found=false` for without
 * scoring sub-metrics).
 */
function ActiveMetrics({
  confidence,
  clarity,
  completeness,
  isolation,
  fallbackModel,
}: {
  confidence: number | null | undefined;
  clarity: number | null | undefined;
  completeness: number | null | undefined;
  isolation: number | null | undefined;
  fallbackModel: string | null | undefined;
}) {
  const items: { label: string; value: number | null | undefined }[] = [
    { label: "conf", value: confidence },
    { label: "clar", value: clarity },
    { label: "comp", value: completeness },
    { label: "iso", value: isolation },
  ];
  const present = items.filter((i) => i.value != null);
  if (present.length === 0 && !fallbackModel) return null;
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
      {fallbackModel && (
        <span
          className="inline-flex items-center gap-1 rounded-full border border-amber-300/50 bg-amber-300/10 px-2 py-0.5 text-[11px] font-medium text-amber-700 backdrop-blur-md dark:border-amber-300/30 dark:bg-amber-300/10 dark:text-amber-300"
          title={`primary 模型拒绝 (< 28 px)，回落到 ${fallbackModel}`}
        >
          <span className="uppercase tracking-wider text-amber-600 dark:text-amber-400">
            回落
          </span>
          <span className="font-mono">{fallbackModel}</span>
        </span>
      )}
    </div>
  );
}
