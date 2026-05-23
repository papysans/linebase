import { forwardRef, type InputHTMLAttributes, type SelectHTMLAttributes } from "react";
import { cn } from "@/lib/cn";

export const GlassInput = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  function GlassInput({ className, ...rest }, ref) {
    return <input ref={ref} className={cn("glass-input", className)} {...rest} />;
  },
);

export const GlassSelect = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(
  function GlassSelect({ className, children, ...rest }, ref) {
    return (
      <select ref={ref} className={cn("glass-select", className)} {...rest}>
        {children}
      </select>
    );
  },
);

export function GlassField({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="text-[13px] font-semibold text-slate-700 dark:text-slate-300">
          {label}
        </span>
        {hint && (
          <span className="text-[11px] text-slate-500 dark:text-slate-400">{hint}</span>
        )}
      </div>
      {children}
    </label>
  );
}
