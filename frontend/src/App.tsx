import { useMemo } from "react";
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
import { EmptyJobState } from "@/components/EmptyJobState";
import { useSession } from "@/lib/session";

const STATIC_NAV: { base: string; label: string }[] = [
  { base: "/", label: "上传" },
  { base: "/configure", label: "配置" },
  { base: "/run", label: "运行" },
  { base: "/review", label: "审查" },
  { base: "/download", label: "下载" },
  { base: "/dev", label: "Dev" },
];

export function App() {
  const location = useLocation();
  const session = useSession();

  // Top-nav links auto-fill the current id so clicking "审查" goes to
  // /review/<jobId> instead of bare /review (which used to render an
  // empty wall).  /configure pivots on uploadId because the configure
  // page is upload-scoped, not job-scoped.
  const navItems = useMemo(() => {
    return STATIC_NAV.map(({ base, label }) => {
      if (base === "/configure" && session.uploadId) {
        return { to: `/configure/${session.uploadId}`, label };
      }
      if (
        (base === "/run" || base === "/review" || base === "/download") &&
        session.jobId
      ) {
        return { to: `${base}/${session.jobId}`, label };
      }
      return { to: base, label };
    });
  }, [session.uploadId, session.jobId]);

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
          <NavPill items={navItems} />
          <ThemeToggle />
        </div>
      </header>

      <main className="relative z-10 mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6">
        <div key={location.pathname} className="page-fade">
          <Routes>
            <Route index element={<UploadPage />} />
            <Route path="configure" element={<EmptyJobState section="configure" />} />
            <Route path="configure/:uploadId" element={<ConfigurePage />} />
            <Route path="run" element={<EmptyJobState section="run" />} />
            <Route path="run/:jobId" element={<RunPage />} />
            <Route path="review" element={<EmptyJobState section="review" />} />
            <Route path="review/:jobId" element={<ReviewPage />} />
            <Route path="download" element={<EmptyJobState section="download" />} />
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
