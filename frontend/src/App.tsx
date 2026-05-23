import { Route, Routes, useLocation } from "react-router-dom";
import { Sparkles } from "lucide-react";
import { UploadPage } from "@/pages/UploadPage";
import { ConfigurePage } from "@/pages/ConfigurePage";
import { RunPage } from "@/pages/RunPage";
import { ReviewPage } from "@/pages/ReviewPage";
import { DownloadPage } from "@/pages/DownloadPage";
import { DevPage } from "@/pages/DevPage";
import { NavPill } from "@/components/NavPill";
import { ThemeToggle } from "@/components/ThemeToggle";

const NAV = [
  { to: "/", label: "上传" },
  { to: "/configure", label: "配置" },
  { to: "/run", label: "运行" },
  { to: "/review", label: "审查" },
  { to: "/download", label: "下载" },
  { to: "/dev", label: "Dev" },
];

export function App() {
  const location = useLocation();
  return (
    <div className="relative min-h-screen flex flex-col">
      <header className="sticky top-0 z-20 px-4 pt-4 pb-2">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4">
          <div className="flex items-center gap-2 glass-nav px-4 py-1.5">
            <span
              className="inline-flex h-6 w-6 items-center justify-center rounded-full text-white"
              style={{
                background:
                  "linear-gradient(135deg, #f0abfc 0%, #7dd3fc 100%)",
              }}
              aria-hidden
            >
              <Sparkles size={14} />
            </span>
            <span className="text-[13px] font-semibold tracking-tight">linebase</span>
          </div>
          <NavPill items={NAV} />
          <ThemeToggle />
        </div>
      </header>

      <main className="relative z-10 mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6">
        <div key={location.pathname} className="page-fade">
          <Routes>
            <Route index element={<UploadPage />} />
            <Route path="configure" element={<RedirectHint label="请先在「上传」页选择文件" />} />
            <Route path="configure/:uploadId" element={<ConfigurePage />} />
            <Route path="run" element={<RedirectHint label="尚未创建任务" />} />
            <Route path="run/:jobId" element={<RunPage />} />
            <Route path="review" element={<RedirectHint label="尚未创建任务" />} />
            <Route path="review/:jobId" element={<ReviewPage />} />
            <Route path="download" element={<RedirectHint label="尚未创建任务" />} />
            <Route path="download/:jobId" element={<DownloadPage />} />
            <Route path="dev" element={<DevPage />} />
            <Route
              path="*"
              element={
                <div className="glass-card p-8 text-center text-slate-500">
                  页面不存在
                </div>
              }
            />
          </Routes>
        </div>
      </main>
    </div>
  );
}

function RedirectHint({ label }: { label: string }) {
  return (
    <div className="glass-card mx-auto max-w-md p-8 text-center">
      <p className="text-sm text-slate-600 dark:text-slate-300">{label}</p>
    </div>
  );
}
