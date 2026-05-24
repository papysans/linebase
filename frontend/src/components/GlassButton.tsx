import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";
import { cn } from "@/lib/cn";

type Variant = "default" | "primary" | "ghost" | "success" | "warn" | "danger";
type Size = "sm" | "md" | "lg";

interface GlassButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  leadingIcon?: ReactNode;
}

export const GlassButton = forwardRef<HTMLButtonElement, GlassButtonProps>(
  function GlassButton(
    { variant = "default", size = "md", leadingIcon, className, children, ...rest },
    ref,
  ) {
    return (
      <button
        ref={ref}
        className={cn(
          "glass-button",
          variant === "primary" && "glass-button--primary",
          variant === "ghost" && "glass-button--ghost",
          variant === "success" && "glass-button--success",
          variant === "warn" && "glass-button--warn",
          variant === "danger" && "glass-button--danger",
          size === "sm" && "glass-button--sm",
          size === "lg" && "glass-button--lg",
          className,
        )}
        {...rest}
      >
        {leadingIcon && <span className="inline-flex shrink-0">{leadingIcon}</span>}
        <span>{children}</span>
      </button>
    );
  },
);
