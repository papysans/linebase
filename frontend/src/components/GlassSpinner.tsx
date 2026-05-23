import { cn } from "@/lib/cn";

interface GlassSpinnerProps {
  size?: number;
  className?: string;
}

/**
 * Concentric translucent rings spinning at slightly different speeds.
 * Pure SVG, no lucide spinner — the default would clash with the glass aesthetic.
 */
export function GlassSpinner({ size = 18, className }: GlassSpinnerProps) {
  const stroke = Math.max(1.5, size / 12);
  return (
    <span
      className={cn("inline-flex shrink-0", className)}
      style={{ width: size, height: size }}
      role="status"
      aria-label="loading"
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 32 32"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <circle
          cx="16"
          cy="16"
          r="13"
          stroke="currentColor"
          strokeOpacity="0.18"
          strokeWidth={stroke}
        />
        <path
          d="M16 3 a13 13 0 0 1 13 13"
          stroke="currentColor"
          strokeOpacity="0.9"
          strokeWidth={stroke}
          strokeLinecap="round"
          style={{
            transformOrigin: "center",
            animation: "spin 0.9s linear infinite",
          }}
        />
        <circle
          cx="16"
          cy="16"
          r="7"
          stroke="currentColor"
          strokeOpacity="0.5"
          strokeWidth={stroke * 0.7}
          strokeLinecap="round"
          strokeDasharray="6 14"
          style={{
            transformOrigin: "center",
            animation: "spin 1.6s linear infinite reverse",
          }}
        />
      </svg>
    </span>
  );
}
