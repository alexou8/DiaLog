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
- Every glucose band has its own point shape in `GlucoseTimeline`: a circle in range, an up or down triangle above or below, and the same triangle enlarged inside a ring for "well above" and "well below". A shape key under the chart explains them. (The critical and caution fills measure about 1.2:1 against each other, so before the ring "well below" and "below" were effectively colour-only.)
- Axis, tick and annotation text is HTML laid over the SVG (`AxisLabel` in `primitives.tsx`), not SVG `<text>`. In a fixed 720-unit viewBox scaled to a phone, SVG text rendered at about 5–6px and ignored the larger-text setting; the HTML labels stay at `text-xs` (12px, more with larger text) on every screen. Y-axis ticks go through the same locale formatting as the table.
- "Not enough data" bars in `BarChart` are hatched and outlined in `ink-muted` (about 7:1 against the surface), with a line under the chart explaining the stripes.

### Focus management

- `:focus-visible` gets a single, always-present treatment (`app/globals.css`): a 3px solid outline in the brand colour with a 2px offset. The rule is deliberately **outside every cascade layer**: when it lived in `@layer base`, a single `outline-none` utility (Tailwind's utilities layer wins over base regardless of specificity) removed it from every Button, ButtonLink and dialog action. Unlayered styles beat all layered ones, so no component utility can switch it off again.
- Invalid fields change shape as well as colour (a thick leading border), carry `aria-invalid`, and still show the focus outline when focused.
- The app shell sets `scroll-padding-top`/`-bottom` to the heights of the sticky header and the fixed bottom bar, so a focused control is never scrolled underneath either (WCAG 2.2 2.4.11).
- A skip link (`.dl-skip-link`, `app/globals.css`) is positioned off-screen (`left: -9999px`) until focused, at which point it becomes visible (`left: 0`) and, when activated, moves focus into `<main id="main">`. This is exercised directly by `tests/e2e/accessibility.spec.ts`'s `assertSkipLink()`, which asserts the first `Tab` press on any page reaches the skip link and that activating it lands focus in or on `<main>`.
- `tests/e2e/keyboard-and-mobile.spec.ts` drives an "open the log panel → add a glucose reading" journey using only the keyboard (`Tab`/`Enter`, no mouse), and separately tabs onto an Input and a Button in both themes and asserts each has a computed focus indicator (an outline, or a ring drawn as a box-shadow).

### Touch targets

`app/globals.css` sets a `min-height: 2.75rem` (44px) on `button`, `[role='button']`, `a.dl-target`, and `input[type='submit']`, applied as a base rule rather than per-component. `a.dl-target` is also `inline-flex`, because `min-height` is ignored on an inline box: on plain inline links the 44px promise used to do nothing. (44px is the AAA 2.5.5 size; AA 2.5.8 needs 24px.)

### Text scaling

Base type is 17px (`--text-base: 1.0625rem`) by design, before any user preference. A per-account "larger text" preference (`Profile.largeText`, set via the Display settings form, `app/app/settings/DisplayForm.tsx`) sets `data-text="large"` or `"x-large"` on `<html>` (applied by `components/AccountPreferences.tsx` from the signed-in shell, `app/app/layout.tsx`, so the public pages stay static), which scales a single CSS custom property, `--dl-font-scale` (1 → 1.15 → 1.3), applied to the root `font-size` in `app/globals.css` — so the whole interface scales from one source of truth, not a scattering of component-level font sizes. This is in addition to, not instead of, respecting the browser/OS's own text-zoom setting (`html { -webkit-text-size-adjust: 100%; }` keeps the browser's native zoom working rather than fighting it).

### Reduced motion

Two independent triggers both work: the OS-level `prefers-reduced-motion: reduce` media query (which collapses all animation/transition durations to effectively zero in `app/globals.css`), and an explicit per-account `Profile.reduceMotion` preference that sets `data-motion="reduced"` on `<html>` (the same `AccountPreferences` component) and is honoured by the same CSS rule — so a user who wants reduced motion in this app specifically, independent of their OS setting, can have it.

### Semantic structure

- Real `<label htmlFor>` bound to every real form control, never a placeholder standing in for a label (`components/ui/form.tsx`'s `Field`).
- Errors are associated to their control via `aria-describedby` (built from `hintId`/`errorId`), marked with `aria-invalid`, and written in plain language ("Please enter a number between 1.1 and 38.9") rather than validation jargon — per that file's own header comment.
- Every form that posts to a Server Action uses `ActionForm` (`components/ui/form.tsx`). A failed submit renders one error summary (`role="alert"`, rendered only on failure) that lists each problem as a link to its field and takes focus. The typed values survive: with JavaScript the form dispatches its action from `onSubmit` inside a transition, which opts out of React 19's automatic form reset; without it, the record and onboarding actions echo the submitted values back and every control reads its default from them. Live regions are limited to the one-line status message, never a whole form or result.
- `<html lang>` is `en-CA`, the language the interface actually renders, not the locale preference. Only the navigation chrome is translated, so it marks itself `lang="fr-CA"` for French-locale users; declaring the whole page French made screen readers read English prose with a French voice (WCAG 3.1.1). The locale preference still drives number and date formatting.
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

- **French translation covers navigation and shared chrome only.** Page content falls back to English (`lib/i18n/dictionaries.ts`'s own header comment says this directly) — a French-speaking screen-reader user gets English prose on most pages today. That prose is correctly declared English, and new chrome strings added since (the mobile "Log" and "More" panels) are English-only until the dictionary grows.
- **Automated testing has a ceiling.** `axe-core` catches a meaningful but partial slice of real accessibility problems (missing alt text, contrast failures, ARIA misuse, structural issues) — it does not catch, for example, whether a screen-reader user's actual task-completion experience through a multi-step flow is good, or whether wording is genuinely clear under cognitive load. There is no recorded assistive-technology (e.g. NVDA/VoiceOver) manual test pass in this repository.
- **No third-party/professional accessibility audit** has been performed on this codebase — this document describes what automated tests and code inspection confirm, not an external certification.
- **Moderate/minor axe findings are logged, not tracked.** `assertNoSeriousViolations()` only fails on serious/critical impact; moderate/minor findings are printed to the CI/local console output and are easy to miss if nobody reads the logs.
- **Colour contrast is asserted by design intent, not by an automated contrast-ratio test.** The design-system comment in `app/globals.css` states every foreground/background pair "meets WCAG 2.2 AA contrast at their intended sizes in both themes," but this is a design claim backed by axe's contrast checks on the tested pages/states, not an exhaustive automated sweep of every colour combination in the app.
