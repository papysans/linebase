import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Saturated aurora hues used for background blobs and accent gradients.
        aurora: {
          cyan: "#7dd3fc",
          magenta: "#f0abfc",
          gold: "#fde68a",
          violet: "#c4b5fd",
        },
        // Glass tints — alpha-tinted surfaces. Use with backdrop-filter (see index.css).
        glass: {
          tint: {
            50: "rgba(255,255,255,0.45)",
            100: "rgba(255,255,255,0.55)",
            200: "rgba(255,255,255,0.65)",
            300: "rgba(255,255,255,0.75)",
            400: "rgba(255,255,255,0.85)",
          },
          stroke: "rgba(255,255,255,0.45)",
          strokeDark: "rgba(255,255,255,0.12)",
          fillDark: {
            50: "rgba(20,22,28,0.45)",
            100: "rgba(20,22,28,0.55)",
            200: "rgba(20,22,28,0.65)",
            300: "rgba(20,22,28,0.75)",
            400: "rgba(20,22,28,0.85)",
          },
        },
        // Keep brand for any third-party that referenced it; alias to the magenta-cyan accent.
        brand: {
          50: "#fdf2ff",
          100: "#fae8ff",
          500: "#d946ef",
          600: "#c026d3",
          700: "#a21caf",
          900: "#581c87",
        },
      },
      backdropBlur: {
        xs: "4px",
        glass: "28px",
        deep: "56px",
      },
      boxShadow: {
        "glass-rim":
          "inset 0 1px 0 rgba(255,255,255,0.75), inset 0 -1px 0 rgba(15,23,42,0.06)",
        "glass-rim-dark":
          "inset 0 1px 0 rgba(255,255,255,0.12), inset 0 -1px 0 rgba(0,0,0,0.35)",
        "glass-lift":
          "0 1px 1px rgba(15,23,42,0.04), 0 8px 24px rgba(15,23,42,0.10), 0 24px 48px -16px rgba(15,23,42,0.18)",
        "glass-press":
          "0 1px 2px rgba(15,23,42,0.08), 0 2px 6px rgba(15,23,42,0.10)",
        "inner-highlight": "inset 0 1px 0 rgba(255,255,255,0.7)",
      },
      borderRadius: {
        glass: "20px",
        "glass-lg": "28px",
      },
      fontFamily: {
        display: [
          '"SF Pro Display"',
          '"SF Pro Text"',
          "-apple-system",
          "BlinkMacSystemFont",
          '"Segoe UI"',
          '"PingFang SC"',
          '"Noto Sans CJK SC"',
          "system-ui",
          "sans-serif",
        ],
      },
      keyframes: {
        "aurora-drift": {
          "0%, 100%": { transform: "translate3d(0,0,0) scale(1)" },
          "33%": { transform: "translate3d(8%, -6%, 0) scale(1.08)" },
          "66%": { transform: "translate3d(-6%, 8%, 0) scale(0.96)" },
        },
        "aurora-drift-2": {
          "0%, 100%": { transform: "translate3d(0,0,0) scale(1)" },
          "50%": { transform: "translate3d(-10%, 10%, 0) scale(1.1)" },
        },
        shimmer: {
          "0%": { transform: "translateX(-120%) skewX(-12deg)" },
          "100%": { transform: "translateX(220%) skewX(-12deg)" },
        },
        "fade-in": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "progress-sheen": {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(100%)" },
        },
      },
      animation: {
        "aurora-drift": "aurora-drift 64s ease-in-out infinite",
        "aurora-drift-slow": "aurora-drift-2 80s ease-in-out infinite",
        shimmer: "shimmer 900ms ease-out",
        "fade-in": "fade-in 380ms cubic-bezier(0.22,1,0.36,1) both",
        "progress-sheen": "progress-sheen 2.4s ease-in-out infinite",
      },
      transitionTimingFunction: {
        spring: "cubic-bezier(0.22, 1, 0.36, 1)",
        fluid: "cubic-bezier(0.65, 0, 0.35, 1)",
      },
    },
  },
  plugins: [],
} satisfies Config;
