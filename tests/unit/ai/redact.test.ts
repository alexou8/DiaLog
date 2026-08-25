import { describe, it, expect } from 'vitest';
import { redactForProvider } from '@/lib/ai/redact';
import { makeBundle, makeFinding } from './fixtures';

describe('redactForProvider', () => {
  it('does nothing for an internal (non-external) provider', () => {
    const bundle = makeBundle({ summary: { avgGlucose: 140, patientNote: 'felt anxious today' } });
    const result = redactForProvider(bundle, false, false);
    expect(result.redacted).toBe(false);
    expect(result.bundle).toEqual(bundle);
  });

  it('does nothing for an external provider when consent was given', () => {
    const bundle = makeBundle({ summary: { avgGlucose: 140, patientNote: 'felt anxious today' } });
    const result = redactForProvider(bundle, true, true);
    expect(result.redacted).toBe(false);
    expect(result.bundle.summary.patientNote).toBe('felt anxious today');
  });

  it('redacts free-text summary keys for an external provider without consent', () => {
    const bundle = makeBundle({
      summary: { avgGlucose: 140, patientNote: 'felt anxious today', diaryEntry: 'skipped lunch' },
    });
    const result = redactForProvider(bundle, true, false);
    expect(result.redacted).toBe(true);
    expect(result.bundle.summary.patientNote).toBeNull();
    expect(result.bundle.summary.diaryEntry).toBeNull();
    expect(result.bundle.summary.avgGlucose).toBe(140);
    expect(result.redactedFields).toContain('summary.patientNote');
    expect(result.redactedFields).toContain('summary.diaryEntry');
  });

  it('redacts quoted-text caveats for an external provider without consent', () => {
    const bundle = makeBundle({
      findings: [
        makeFinding({
          caveats: [
            'Small sample size',
            'User wrote "I felt terrible after this meal" in their log',
          ],
        }),
      ],
    });
    const result = redactForProvider(bundle, true, false);
    expect(result.redacted).toBe(true);
    expect(result.bundle.findings[0]?.caveats).toEqual(['Small sample size']);
  });

  it('leaves the bundle untouched when there is nothing to redact', () => {
    const bundle = makeBundle();
    const result = redactForProvider(bundle, true, false);
    expect(result.redacted).toBe(false);
    expect(result.redactedFields).toEqual([]);
  });

  it('does not mutate the original bundle object', () => {
    const bundle = makeBundle({ summary: { avgGlucose: 140, patientNote: 'note text' } });
    const original = JSON.parse(JSON.stringify(bundle));
    redactForProvider(bundle, true, false);
    expect(bundle).toEqual(original);
  });
});
