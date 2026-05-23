import { useState, type DragEvent } from "react";
import { useNavigate } from "react-router-dom";
import { FileSpreadsheet, UploadCloud, X } from "lucide-react";
import { api } from "@/lib/api";
import { GlassButton } from "@/components/GlassButton";
import { GlassSpinner } from "@/components/GlassSpinner";
import { cn } from "@/lib/cn";

function fmtMB(bytes: number): string {
  return (bytes / 1024 / 1024).toFixed(1) + " MB";
}

export function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [progress, setProgress] = useState<{ loaded: number; total: number } | null>(null);
  const navigate = useNavigate();

  const submit = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    setProgress({ loaded: 0, total: file.size });
    try {
      const res = await api.uploadXlsx(file, (p) => setProgress(p));
      navigate(`/configure/${res.id}`);
    } catch (e) {
      setError(String(e));
      setProgress(null);
    } finally {
      setUploading(false);
    }
  };

  const onDrop = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setDragActive(false);
    const f = e.dataTransfer.files?.[0];
    if (f && /\.xlsx$/i.test(f.name)) setFile(f);
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">上传商标表格</h1>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          支持 .xlsx 工作簿；解析后可选择目标 sheet 与图片列。
        </p>
      </header>

      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={onDrop}
        className={cn(
          "glass-pane relative block cursor-pointer p-10 transition-all duration-300 ease-spring",
          dragActive && "scale-[1.01]",
        )}
        style={
          dragActive
            ? {
                boxShadow:
                  "0 0 0 2px rgba(192,132,252,0.5), 0 24px 48px -20px rgba(192,132,252,0.45), inset 0 1px 0 rgba(255,255,255,0.75)",
              }
            : undefined
        }
      >
        <input
          type="file"
          accept=".xlsx"
          className="hidden"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        {/* aurora glow seeping through */}
        <span
          aria-hidden
          className={cn(
            "pointer-events-none absolute inset-0 rounded-[28px] opacity-50 blur-2xl transition-opacity duration-500",
            dragActive ? "opacity-90" : "opacity-50",
          )}
          style={{
            background:
              "radial-gradient(60% 70% at 50% 50%, rgba(240,171,252,0.35) 0%, rgba(125,211,252,0.2) 50%, transparent 75%)",
          }}
        />

        <div className="relative flex flex-col items-center text-center">
          <div
            className="mb-4 flex h-16 w-16 items-center justify-center rounded-full"
            style={{
              background:
                "linear-gradient(135deg, rgba(240,171,252,0.5), rgba(125,211,252,0.5))",
              boxShadow:
                "inset 0 1px 0 rgba(255,255,255,0.7), 0 8px 24px -8px rgba(217,70,239,0.45)",
            }}
          >
            <UploadCloud className="text-white" size={28} />
          </div>
          {file ? (
            <div className="flex items-center gap-3 rounded-full bg-white/40 px-4 py-2 backdrop-blur-md dark:bg-white/10">
              <FileSpreadsheet size={16} className="text-aurora-magenta" />
              <span className="text-sm font-medium">{file.name}</span>
              <span className="text-xs text-slate-500 dark:text-slate-400">
                {(file.size / 1024 / 1024).toFixed(1)} MB
              </span>
              <button
                type="button"
                onClick={(e) => {
                  e.preventDefault();
                  setFile(null);
                }}
                aria-label="移除"
                className="rounded-full p-0.5 text-slate-500 hover:bg-white/40 hover:text-slate-900 dark:hover:bg-white/10 dark:hover:text-slate-100"
              >
                <X size={14} />
              </button>
            </div>
          ) : (
            <>
              <div className="text-base font-medium">
                点击选择 .xlsx 文件，或拖拽到此处
              </div>
              <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                文件保留在本机，不上传到外部服务
              </div>
            </>
          )}
        </div>
      </label>

      {error && (
        <div className="glass-card underglow-bad px-4 py-3 text-sm text-rose-700 dark:text-rose-300">
          {error}
        </div>
      )}

      {uploading && progress && progress.total > 0 && (
        <div className="glass-card space-y-2 px-4 py-3">
          <div className="flex items-baseline justify-between text-xs">
            <span className="font-medium text-slate-600 dark:text-slate-300">上传中</span>
            <span className="font-mono tabular-nums text-slate-500 dark:text-slate-400">
              {fmtMB(progress.loaded)} / {fmtMB(progress.total)}{" "}
              <span className="text-aurora-magenta">
                · {((progress.loaded / progress.total) * 100).toFixed(0)}%
              </span>
            </span>
          </div>
          <div
            className="relative h-2 overflow-hidden rounded-full bg-white/40 dark:bg-white/10"
            aria-label="upload progress"
            role="progressbar"
            aria-valuenow={Math.round((progress.loaded / progress.total) * 100)}
            aria-valuemin={0}
            aria-valuemax={100}
          >
            <div
              className="absolute inset-y-0 left-0 transition-[width] duration-150 ease-out"
              style={{
                width: `${Math.min(100, (progress.loaded / progress.total) * 100)}%`,
                background:
                  "linear-gradient(90deg, rgba(240,171,252,0.85), rgba(125,211,252,0.85))",
                boxShadow:
                  "0 0 12px rgba(192,132,252,0.55), inset 0 1px 0 rgba(255,255,255,0.7)",
              }}
            />
          </div>
          {progress.loaded >= progress.total && (
            <div className="text-[11px] text-slate-500 dark:text-slate-400">
              文件已传完，后端解析 sheet 中…
            </div>
          )}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <GlassButton
          variant="primary"
          size="lg"
          disabled={!file || uploading}
          onClick={submit}
          leadingIcon={uploading ? <GlassSpinner size={16} /> : undefined}
        >
          {uploading ? "上传中…" : "上传并解析"}
        </GlassButton>
        {file && !uploading && (
          <GlassButton variant="ghost" onClick={() => setFile(null)}>
            清除选择
          </GlassButton>
        )}
      </div>
    </div>
  );
}
