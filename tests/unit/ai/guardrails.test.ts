import { describe, it, expect } from 'vitest';
import {
  validateAssistantAnswer,
  checkMedicalSafety,
  safeFallbackAnswer,
  enforceGrounding,
  enforceClaimConsistency,
  applyAssistantAnswerGuardrails,
  LIMITED_EVIDENCE_CAVEAT,
} from '@/lib/ai/guardrails';
import type { AssistantAnswer } from '@/lib/ai/schemas';
import { makeBundle, makeFinding, makeInsufficientBundle } from './fixtures';

function baseAnswer(overrides: Partial<AssistantAnswer> = {}): AssistantAnswer {
  return {
    shortAnswer: 'Your data shows a mild rise after breakfast.',
    detail: ['Your data shows a mild rise after breakfast.'],
    citedFindingIds: ['finding-1'],
    confidence: 'moderate',
    suggestedQuestionsForClinician: [],
    notEnoughData: false,
    ...overrides,
  };
}

describe('validateAssistantAnswer', () => {
  it('accepts a well-formed payload', () => {
    const result = validateAssistantAnswer(baseAnswer());
    expect(result.ok).toBe(true);
  });

  it('rejects a payload missing required fields', () => {
    const result = validateAssistantAnswer({ shortAnswer: 'x' });
    expect(result.ok).toBe(false);
  });

  it('rejects shortAnswer over 280 chars', () => {
    const result = validateAssistantAnswer(baseAnswer({ shortAnswer: 'x'.repeat(281) }));
    expect(result.ok).toBe(false);
  });

  it('rejects an invalid confidence enum value', () => {
    const result = validateAssistantAnswer({ ...baseAnswer(), confidence: 'certain' });
    expect(result.ok).toBe(false);
  });
});

describe('checkMedicalSafety', () => {
  const unsafeCases: { label: string; text: string }[] = [
    {
      label: 'dosing-instruction (increase insulin)',
      text: 'You should increase your insulin dose tomorrow.',
    },
    {
      label: 'dosing-instruction (decrease units)',
      text: 'Consider decreasing your units of medication.',
    },
    { label: 'take-units', text: 'Take 10 units before dinner.' },
    { label: 'diagnosis', text: 'Based on this, you have diabetes.' },
    { label: 'diagnose-verb', text: 'This data helps diagnose your condition.' },
    { label: 'skip-doctor', text: "You don't need to see a doctor about this." },
    { label: 'clinically-proven', text: 'This approach is clinically proven to help.' },
    { label: 'medical-grade', text: 'This is a medical-grade recommendation.' },
  ];

  it.each(unsafeCases)('flags unsafe pattern: $label', ({ text }) => {
    const answer = baseAnswer({ shortAnswer: text, detail: [text] });
    const result = checkMedicalSafety(answer);
    expect(result.safe).toBe(false);
    expect(result.matchedPatterns.length).toBeGreaterThan(0);
  });

  it('passes safe, evidence-grounded language', () => {
    const answer = baseAnswer({
      shortAnswer: 'Your data shows a pattern worth discussing with your healthcare provider.',
      detail: ['This appears to be associated with later bedtimes on weekends.'],
    });
    const result = checkMedicalSafety(answer);
    expect(result.safe).toBe(true);
    expect(result.matchedPatterns).toEqual([]);
  });

  it('flags unsafe language even when only in suggestedQuestionsForClinician', () => {
    const answer = baseAnswer({
      suggestedQuestionsForClinician: ['Should I increase my insulin dose?'],
    });
    const result = checkMedicalSafety(answer);
    expect(result.safe).toBe(false);
  });
});

describe('safeFallbackAnswer', () => {
  it('is itself schema-valid and marked notEnoughData', () => {
    const fallback = safeFallbackAnswer();
    expect(validateAssistantAnswer(fallback).ok).toBe(true);
    expect(fallback.notEnoughData).toBe(true);
    expect(fallback.citedFindingIds).toEqual([]);
  });
});

describe('enforceGrounding', () => {
  it('drops citedFindingIds not present in the bundle', () => {
    const bundle = makeBundle();
    const answer = baseAnswer({ citedFindingIds: ['finding-1', 'made-up-id'] });
    const result = enforceGrounding(answer, bundle);
    expect(result.value.citedFindingIds).toEqual(['finding-1']);
    expect(result.notes.some((n) => n.includes('dropped'))).toBe(true);
  });

  it('downgrades confidence and appends the caveat when no citations remain grounded', () => {
    const bundle = makeBundle();
    const answer = baseAnswer({ citedFindingIds: ['unknown-1', 'unknown-2'], confidence: 'high' });
    const result = enforceGrounding(answer, bundle);
    expect(result.value.citedFindingIds).toEqual([]);
    expect(result.value.confidence).toBe('low');
    expect(result.value.detail).toContain(LIMITED_EVIDENCE_CAVEAT);
  });

  it('does not downgrade when notEnoughData is already true', () => {
    const bundle = makeBundle();
    const answer = baseAnswer({
      citedFindingIds: ['unknown'],
      notEnoughData: true,
      confidence: 'insufficient',
    });
    const result = enforceGrounding(answer, bundle);
    expect(result.value.confidence).toBe('insufficient');
    expect(result.value.detail).not.toContain(LIMITED_EVIDENCE_CAVEAT);
  });

  it('leaves valid citations untouched', () => {
    const bundle = makeBundle({ findings: [makeFinding({ id: 'a' }), makeFinding({ id: 'b' })] });
    const answer = baseAnswer({ citedFindingIds: ['a', 'b'] });
    const result = enforceGrounding(answer, bundle);
    expect(result.value.citedFindingIds).toEqual(['a', 'b']);
    expect(result.notes).toEqual([]);
  });
});

describe('enforceClaimConsistency', () => {
  it('forces notEnoughData true when all findings are INSUFFICIENT', () => {
    const bundle = makeInsufficientBundle();
    const answer = baseAnswer({
      notEnoughData: false,
      confidence: 'high',
      citedFindingIds: ['finding-insuff'],
    });
    const result = enforceClaimConsistency(answer, bundle);
    expect(result.value.notEnoughData).toBe(true);
    expect(result.value.confidence).toBe('insufficient');
    expect(result.value.citedFindingIds).toEqual([]);
    expect(result.notes.length).toBeGreaterThan(0);
  });

  it('forces notEnoughData true when the bundle has zero findings', () => {
    const bundle = makeBundle({ findings: [] });
    const answer = baseAnswer({ notEnoughData: false });
    const result = enforceClaimConsistency(answer, bundle);
    expect(result.value.notEnoughData).toBe(true);
  });

  it('leaves the answer unchanged when at least one finding has real evidence', () => {
    const bundle = makeBundle();
    const answer = baseAnswer({ notEnoughData: false });
    const result = enforceClaimConsistency(answer, bundle);
    expect(result.value).toEqual(answer);
    expect(result.notes).toEqual([]);
  });
});

describe('applyAssistantAnswerGuardrails (full pipeline)', () => {
  it('rejects and falls back on schema-invalid payloads', () => {
    const bundle = makeBundle();
    const result = applyAssistantAnswerGuardrails({ nonsense: true }, bundle);
    expect(result.rejected).toBe(true);
    expect(result.answer.notEnoughData).toBe(true);
  });

  it('rejects and falls back on unsafe medical language', () => {
    const bundle = makeBundle();
    const unsafe = baseAnswer({ shortAnswer: 'You should stop taking your insulin dose.' });
    const result = applyAssistantAnswerGuardrails(unsafe, bundle);
    expect(result.rejected).toBe(true);
    expect(result.answer).toEqual(safeFallbackAnswer());
  });

  it('grounds and passes through a clean, valid, safe answer', () => {
    const bundle = makeBundle();
    const clean = baseAnswer();
    const result = applyAssistantAnswerGuardrails(clean, bundle);
    expect(result.rejected).toBe(false);
    expect(result.answer.citedFindingIds).toEqual(['finding-1']);
  });

  it('forces notEnoughData when bundle findings are all insufficient, even for an otherwise-clean answer', () => {
    const bundle = makeInsufficientBundle();
    const clean = baseAnswer({ citedFindingIds: [], notEnoughData: false });
    const result = applyAssistantAnswerGuardrails(clean, bundle);
    expect(result.answer.notEnoughData).toBe(true);
  });
});
