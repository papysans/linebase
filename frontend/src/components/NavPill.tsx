import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { cn } from "@/lib/cn";

interface NavItem {
  to: string;
  label: string;
}

interface NavPillProps {
  items: NavItem[];
}

/**
 * Floating top-center nav. A sliding gradient indicator tracks the active
 * route — measured from the active <a> element's offset within the container.
 */
export function NavPill({ items }: NavPillProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const linkRefs = useRef<Record<string, HTMLAnchorElement | null>>({});
  const location = useLocation();
  const [indicator, setIndicator] = useState<{ left: number; width: number } | null>(null);

  const measure = () => {
    const container = containerRef.current;
    if (!container) return;
    // Find the active <a> by checking aria-current.
    const active = container.querySelector<HTMLAnchorElement>('a[aria-current="page"]');
    if (!active) {
      setIndicator(null);
      return;
    }
    const cRect = container.getBoundingClientRect();
    const aRect = active.getBoundingClientRect();
    setIndicator({
      left: aRect.left - cRect.left + container.scrollLeft,
      width: aRect.width,
    });
  };

  useLayoutEffect(() => {
    measure();
  }, [location.pathname]);

  useEffect(() => {
    const onResize = () => measure();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  return (
    <div
      ref={containerRef}
      className="glass-nav relative flex items-center gap-0.5 px-1.5 py-1.5"
    >
      {indicator && (
        <span
          aria-hidden
          className="pointer-events-none absolute top-1.5 bottom-1.5 rounded-full transition-all duration-300 ease-spring"
          style={{
            left: indicator.left,
            width: indicator.width,
            background:
              "linear-gradient(135deg, rgba(240,171,252,0.32) 0%, rgba(125,211,252,0.32) 100%)",
            boxShadow:
              "inset 0 1px 0 rgba(255,255,255,0.55), 0 4px 14px -6px rgba(217,70,239,0.35)",
            border: "1px solid rgba(255,255,255,0.4)",
          }}
        />
      )}
      {items.map((it) => (
        <NavLink
          key={it.to}
          to={it.to}
          end={it.to === "/"}
          ref={(el) => {
            linkRefs.current[it.to] = el;
          }}
          className={({ isActive }) =>
            cn(
              "relative z-10 px-3.5 py-1.5 text-[13px] font-medium rounded-full transition-colors duration-200",
              isActive
                ? "text-slate-900 dark:text-slate-50"
                : "text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100",
            )
          }
        >
          {it.label}
        </NavLink>
      ))}
    </div>
  );
}
