import type { HTMLAttributes } from "react";
import { cn } from "@/lib/cn";

interface GlassCardProps extends HTMLAttributes<HTMLDivElement> {
  variant?: "card" | "pane";
  interactive?: boolean;
}

export function GlassCard({
  variant = "card",
  interactive,
  className,
  children,
  ...rest
}: GlassCardProps) {
  return (
    <div
      className={cn(
        variant === "pane" ? "glass-pane" : "glass-card",
        interactive && "is-interactive cursor-pointer",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}
