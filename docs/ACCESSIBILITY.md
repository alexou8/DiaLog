# Accessibility

DiaLog targets **WCAG 2.2 level AA**. The product's own accessibility page (`app/(marketing)/accessibility/page.tsx`) states the reasoning directly: the people most likely to be managing glucose day to day are also disproportionately likely to be doing it with imperfect eyesight, unsteady hands, or a screen reader, so this was built in rather than added afterward. This document describes what that commitment actually looks like in code, how it is tested, and where the real gaps are.

## Concrete implementation decisions

### Colour is never the only carrier of meaning

`lib/domain/thresholds.ts`'s `GlucoseBand` type pairs every status with three things, not one: a `tone` (colour), an `icon` (`alert | down | check | up`), and a plain-language `label`/`description` ("Above your target range", "Well below your target range"). The design-system comment at the top of `app/globals.css` states the rule as a property of every colour pair in the palette: "Every status colour is paired with an icon and a text label elsewhere in the UI — colour never carries meaning alone," and the status tones themselves are chosen to stay distinguishable under common forms of colour vision deficiency.

### Chart text alternatives and data tables

Charts are hand-built inline SVG specifically so this contract can be guaranteed (see [ARCHITECTURE.md](ARCHITECTURE.md) and the README's tech-stack rationale) rather than depending on what a third-party charting library happens to expose. `components/charts/primitives.tsx`'s `ChartFrame` wrapper is the shared contract every chart uses:

- The chart itself is `role="img"` with an `aria-label` combining a title and a plain-language summary sentence, so assistive technology announces what the chart shows without needing to parse SVG.
- A visible `<p>` summary repeats that plain-language description for sighted users who want the takeaway without reading the chart.
- A `<details>`-collapsed real `<table>` (with a `<caption>`, `scope="col"` headers, and the same numbers the chart draws) is included in the DOM on every chart, so a screen-reader or keyboard user can read the exact data, not just an approximation.
- Out-of-range points are drawn with both a different shape and colour, per the same "never colour alone" rule, not just a colour change on the SVG mark.

### Focus management

- `:focus-visible` gets a single, always-present treatment (`app/globals.css`): a 3px solid outline in the brand colour with a 2px offset — the comment above it is explicit that this is "never removed."
- A skip link (`.dl-skip-link`, `app/globals.css`) is positioned off-screen (`left: -9999px`) until focused, at which point it becomes visible (`left: 0`) and, when activated, moves focus into `<main id="main">`. This is exercised directly by `tests/e2e/accessibility.spec.ts`'s `assertSkipLink()`, which asserts the first `Tab` press on any page reaches the skip link and that activating it lands focus in or on `<main>`.
- `tests/e2e/keyboard-and-mobile.spec.ts` drives a full "sign up → add a glucose reading" journey using only the keyboard (`Tab`/`Enter`/`ArrowRight`, no mouse) and asserts a real computed focus outline exists (`outlineStyle !== 'none'`, `outlineWidth > 0`) before doing so.

### Touch targets

`app/globals.css` sets a `min-height: 2.75rem` (44px) on `button`, `[role='button']`, `a.dl-target`, and `input[type='submit']` — the WCAG 2.2 AA 2.5.8 minimum target size, applied as a base rule rather than per-component.

### Text scaling

Base type is 17px (`--text-base: 1.0625rem`) by design, before any user preference. A per-account "larger text" preference (`Profile.largeText`, set via the Display settings form, `app/app/settings/DisplayForm.tsx`) sets `data-text="large"` or `"x-large"` on `<html>` (`app/layout.tsx`), which scales a single CSS custom property, `--dl-font-scale` (1 → 1.15 → 1.3), applied to the root `font-size` in `app/globals.css` — so the whole interface scales from one source of truth, not a scattering of component-level font sizes. This is in addition to, not instead of, respecting the browser/OS's own text-zoom setting (`html { -webkit-text-size-adjust: 100%; }` keeps the browser's native zoom working rather than fighting it).

### Reduced motion

Two independent triggers both work: the OS-level `prefers-reduced-motion: reduce` media query (which collapses all animation/transition durations to effectively zero in `app/globals.css`), and an explicit per-account `Profile.reduceMotion` preference that sets `data-motion="reduced"` on `<html>` (`app/layout.tsx`) and is honoured by the same CSS rule — so a user who wants reduced motion in this app specifically, independent of their OS setting, can have it.

### Semantic structure

- Real `<label htmlFor>` bound to every real form control, never a placeholder standing in for a label (`components/ui/form.tsx`'s `Field`).
- Errors are associated to their control via `aria-describedby` (built from `hintId`/`errorId`), marked with `aria-invalid`, and written in plain language ("Please enter a number between 1.1 and 38.9") rather than validation jargon — per that file's own header comment.
- `<html lang>` is set per request from the user's locale (`fr-CA` or `en-CA`) in `app/layout.tsx`, not hardcoded.
- `tests/e2e/accessibility.spec.ts`'s `assertHeadingStructure()` asserts, on every tested page, that there is exactly one `<h1>` and that no heading level is skipped going down the page (h1 → h2 → h3, never h1 → h3).

## How accessibility is tested

`tests/e2e/accessibility.spec.ts` (Playwright + `@axe-core/playwright`) is the automated suite:

- Runs `AxeBuilder` with the `wcag2a`, `wcag2aa`, `wcag21a`, `wcag21aa`, and `wcag22aa` rule tags against every page in `PUBLIC_PAGES` (`/`, `/privacy`, `/accessibility`, `/sign-in`, `/sign-up`) and, signed in as the demo account, every page in `APP_PAGES` (`/app`, `/app/glucose`, `/app/glucose/new`, `/app/insights`, `/app/import`, `/app/settings`, `/app/history`).
- **Serious/critical** axe violations fail the test outright. **Moderate/minor** findings are logged to the console but do not fail the suite — a deliberate choice (see the code comment) so the signal that matters most isn't drowned out, while still surfacing lower-severity findings for a human to triage.
- Heading structure, `<html lang>`, and the skip-link behaviour are asserted on every one of those same pages, on every run — not spot-checked once.
- `tests/e2e/keyboard-and-mobile.spec.ts` separately covers a real keyboard-only task completion (not just "can Tab reach things") and runs its mobile-viewport checks under the `mobile-chromium` Playwright project (`playwright.config.ts`, `devices['Pixel 7']`).

Run it with `npm run test:e2e` (all Playwright specs, including this one) or narrow to just this file with `npx playwright test tests/e2e/accessibility.spec.ts`.

This is automated coverage of the main journeys, not a substitute for testing with real assistive technology and real users — see Known gaps below.

## Known gaps

Stated plainly, matching what the product's own accessibility page (`app/(marketing)/accessibility/page.tsx`) already tells users:

- **French translation covers navigation and shared chrome only.** Page content falls back to English (`lib/i18n/dictionaries.ts`'s own header comment says this directly) — a French-speaking screen-reader user gets English prose on most pages today.
- **Automated testing has a ceiling.** `axe-core` catches a meaningful but partial slice of real accessibility problems (missing alt text, contrast failures, ARIA misuse, structural issues) — it does not catch, for example, whether a screen-reader user's actual task-completion experience through a multi-step flow is good, or whether wording is genuinely clear under cognitive load. There is no recorded assistive-technology (e.g. NVDA/VoiceOver) manual test pass in this repository.
- **No third-party/professional accessibility audit** has been performed on this codebase — this document describes what automated tests and code inspection confirm, not an external certification.
- **Moderate/minor axe findings are logged, not tracked.** `assertNoSeriousViolations()` only fails on serious/critical impact; moderate/minor findings are printed to the CI/local console output and are easy to miss if nobody reads the logs.
- **Colour contrast is asserted by design intent, not by an automated contrast-ratio test.** The design-system comment in `app/globals.css` states every foreground/background pair "meets WCAG 2.2 AA contrast at their intended sizes in both themes," but this is a design claim backed by axe's contrast checks on the tested pages/states, not an exhaustive automated sweep of every colour combination in the app.
