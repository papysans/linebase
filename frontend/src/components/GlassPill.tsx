import { cn } from "@/lib/cn";

export type PillStatus = "ok" | "bad" | "needs_review";

interface GlassPillProps {
  value: PillStatus | null;
  onChange: (next: PillStatus) => void;
  options?: PillStatus[];
}

const LABEL: Record<PillStatus, string> = {
  ok: "OK",
  bad: "BAD",
  needs_review: "NEEDS REVIEW",
};

const VARIANT_CLASS: Record<PillStatus, string> = {
  ok: "glass-pill__option--ok",
  bad: "glass-pill__option--bad",
  needs_review: "glass-pill__option--review",
};

export function GlassPill({
  value,
  onChange,
  options = ["ok", "bad", "needs_review"],
}: GlassPillProps) {
  return (
    <div className="glass-pill" role="radiogroup">
      {options.map((opt) => {
        const active = value === opt;
        return (
          <button
            key={opt}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(opt)}
            className={cn(
              "glass-pill__option",
              VARIANT_CLASS[opt],
              active && "glass-pill__option--active",
            )}
          >
            {LABEL[opt]}
          </button>
        );
      })}
    </div>
  );
}
