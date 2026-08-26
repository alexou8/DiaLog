import { cva, type VariantProps } from 'class-variance-authority';

/**
 * The health-status tone scale, defined once.
 *
 * Badge, Alert and Stat all need the same soft-fill / border / text triplet,
 * and when it was written out in each of them they drifted: a "caution" badge
 * and a "caution" callout stopped being the same colour. Every tone here
 * resolves to tokens in globals.css, so the dark-theme override moves all
 * three at once.
 *
 * Tone is a presentational scale only. It carries no semantics of its own,
 * which is why it lives apart from the components: a Stat needs the caution
 * colours without also announcing itself as a note. Meaning always comes from
 * the text label and icon the call site supplies.
 */
export const toneVariants = cva('', {
  variants: {
    tone: {
      brand: 'border-brand/30 bg-brand-soft text-brand-ink',
      positive: 'border-positive/30 bg-positive-soft text-positive',
      notice: 'border-notice/30 bg-notice-soft text-notice',
      caution: 'border-caution/30 bg-caution-soft text-caution',
      critical: 'border-critical/30 bg-critical-soft text-critical',
      info: 'border-info/30 bg-info-soft text-info',
      neutral: 'border-line bg-surface-sunken text-ink-muted',
    },
  },
  defaultVariants: { tone: 'neutral' },
});

export type Tone = NonNullable<VariantProps<typeof toneVariants>['tone']>;
