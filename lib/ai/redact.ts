/**
 * Redaction helper: strips free-text notes from an evidence bundle when the
 * selected provider `isExternal` (data would leave this deployment) and the
 * user has not consented to external processing.
 *
 * Rationale: `Finding.basis` and `Finding.caveats` are analyst-authored
 * strings that describe *how* a finding was computed and are safe to send
 * anywhere. But nothing in the current `EvidenceBundle`/`Finding` shape
 * carries verbatim personal free-text (e.g. a symptom note, a diary entry) —
 * `summary` values are typed as `number | string | null`, and any caller
 * that maps free-text notes into the bundle should route them through
 * `summary` under a well-known key (e.g. "recentNoteText") rather than a
 * bespoke field. This function defensively strips exactly that: any
 * `summary` entries whose key looks like it carries free-text notes, plus
 * any `caveats` entries that look like they quote a user note directly
 * (heuristically, anything wrapped in quotation marks).
 *
 * This is a belt-and-suspenders redaction layer, not the only line of
 * defense — the app layer should never put verbatim free text into the
 * bundle in the first place unless the user has consented to it leaving the
 * server. This function exists so that if it slips in anyway, an external
 * provider still never sees it without consent.
 */
import type { EvidenceBundle } from './types';

/** Summary keys treated as carrying user-authored free text rather than a computed metric. */
const FREE_TEXT_SUMMARY_KEYS = /note|comment|freetext|free_text|diary|journal/i;

const QUOTED_TEXT_PATTERN = /["“][^"”]{3,}["”]/;

export interface RedactionResult {
  bundle: EvidenceBundle;
  redacted: boolean;
  redactedFields: string[];
}

/**
 * Redact free-text notes from `bundle` when required.
 *
 * @param bundle the evidence bundle as built by the app layer
 * @param isExternalProvider `provider.isExternal` for the provider about to be called
 * @param hasExternalProcessingConsent whether the user has consented to their
 *   data being sent to an external (non-DiaLog-hosted) AI provider
 */
export function redactForProvider(
  bundle: EvidenceBundle,
  isExternalProvider: boolean,
  hasExternalProcessingConsent: boolean,
): RedactionResult {
  if (!isExternalProvider || hasExternalProcessingConsent) {
    return { bundle, redacted: false, redactedFields: [] };
  }

  const redactedFields: string[] = [];

  const summary: Record<string, number | string | null> = { ...bundle.summary };
  for (const key of Object.keys(summary)) {
    if (FREE_TEXT_SUMMARY_KEYS.test(key)) {
      summary[key] = null;
      redactedFields.push(`summary.${key}`);
    }
  }

  const findings = bundle.findings.map((finding, idx) => {
    if (!finding.caveats || finding.caveats.length === 0) return finding;
    const keptCaveats = finding.caveats.filter((c) => !QUOTED_TEXT_PATTERN.test(c));
    if (keptCaveats.length === finding.caveats.length) return finding;
    redactedFields.push(`findings[${idx}].caveats`);
    return { ...finding, caveats: keptCaveats };
  });

  return {
    bundle: { ...bundle, summary, findings },
    redacted: redactedFields.length > 0,
    redactedFields,
  };
}
