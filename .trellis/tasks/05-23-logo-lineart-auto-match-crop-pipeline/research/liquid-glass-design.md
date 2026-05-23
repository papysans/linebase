# Liquid-Glass Design Notes (linebase frontend)

## Why "liquid glass", not "glassmorphism"
- Glassmorphism (2020-era): flat, uniform `backdrop-filter: blur` over a static photo background. Surfaces look like a window pane.
- Liquid Glass (Apple WWDC24): the translucent surface still uses backdrop blur + saturation boost, but the background **behind** the glass is alive — slow-moving aurora gradients that the glass picks up and refracts. The glass itself has multi-layer shadows that suggest physical thickness (inner highlight on top edge + outer drop shadow). Surfaces respond to interaction (sheen sweep on hover, scale-press on tap).
- Practical translation in CSS:
  - Animated `radial-gradient` aurora behind everything (40–80s drift loop, paused under `prefers-reduced-motion`).
  - Real `backdrop-filter: blur(28px) saturate(180%)` on every surface, NOT `bg-white/80`.
  - Multi-layer `box-shadow`: outer drop (`0 8px 32px rgba(15,23,42,.12)`) + inner top-edge highlight (`inset 0 1px 0 rgba(255,255,255,.7)`) + inner bottom shade.
  - `border: 1px solid rgba(255,255,255,.45)` to mimic the bevel.
  - Hover `::before` overlay with a 45-degree linear-gradient sheen that translates across (~900ms).

## Color system
- Base: near-white (`#f6f7fb`) light / near-black (`#0b0d12`) dark. The aurora blobs sit ON this base.
- Aurora hues (the saturated underglow):
  - cyan: `#7dd3fc`
  - magenta: `#f0abfc`
  - gold: `#fde68a`
  - In dark mode the same hues but at lower lightness so they read as a faint glow rather than pastel sherbet.
- Accent gradient for active states + brand: `linear-gradient(135deg, #f0abfc 0%, #7dd3fc 100%)` (magenta → cyan).
- Status tints (used as soft underglows in lists, not as solid badges):
  - OK: `rgba(52, 211, 153, .25)` (emerald)
  - BAD: `rgba(248, 113, 113, .25)` (rose)
  - NEEDS_REVIEW: `rgba(251, 191, 36, .25)` (amber)

## Typography
- System stack: `"SF Pro Display", "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif`. SF is preferred (matches Apple aesthetic); CJK fall-throughs cover Chinese page labels.
- Two weights only: 400 for body, 600 for headings/labels. No 700 — it fights the glass softness.
- Display heading: `text-3xl tracking-tight font-semibold` with `-0.02em` letter-spacing.
- Body: 14px on dense controls, 15px on cards.

## Spatial rhythm
- 4-pt grid (Tailwind's default). Page top padding = 32px (`pt-8`), card internal = 20–24px.
- Glass radii: small controls 12px, cards 20px, hero zones 28px. Avoid sharp corners — liquid glass is round-cornered by nature.

## Motion primitives
- Easings: `cubic-bezier(0.22, 1, 0.36, 1)` for snap-in, `cubic-bezier(0.65, 0, 0.35, 1)` for cross-fades.
- Durations: 150ms (button press), 250ms (hover lift), 600ms (page cross-fade), 40–80s (aurora drift).
- Spring-feeling press: `transform: scale(.97)` + `transition: transform 120ms`.
- `prefers-reduced-motion`: aurora animation disabled; surfaces remain.

## Components rolled (not shadcn — kept dependency-free)
- `GlassCard` / `GlassPane` (compositional wrappers using `.glass-card` / `.glass-pane` utility).
- `GlassButton` (variants: primary / ghost / danger).
- `GlassInput`, `GlassSelect` (native `<input>` / `<select>` with glass class).
- `GlassPill` (segmented control for OK/BAD/NEEDS_REVIEW).
- `AuroraBackground` (slot in app shell; CSS-only, no React state).
- `GlassSpinner` (concentric SVG rings).
- `NavPill` (top-center floating nav with sliding indicator).
- `ThemeToggle` (toggles `class="dark"` on `<html>`, persisted to `localStorage`).
- `PageTransition` (CSS-only opacity cross-fade keyed by `location.pathname`).

## Light vs dark default
- Recommendation: **light default**. The app's main visual content is line-art logos (black on white) and product photos with light backgrounds; the aurora-on-near-white setting frames them best. Dark mode is offered but optional.
