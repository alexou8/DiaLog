/**
 * Adversarial cases for the medical-safety filter.
 *
 * The first table is the 27 constructed outputs from the 2026-10 review
 * (scratchpad rx.mjs). Against the old patterns 24 of them slipped through;
 * every one must now be caught. The second table is ordinary, evidence-based
 * phrasing the assistant produces every day; over-matching is accepted by
 * design (AGENTS.md, AI), but not on these.
 */
import { describe, expect, it } from 'vitest';
import {
  applyAssistantAnswerGuardrails,
  checkTextsSafety,
  enforceConfidenceCap,
  findUnsafeLanguage,
  normalizeForSafety,
  LIMITED_EVIDENCE_CAVEAT,
} from '@/lib/ai/guardrails';
import { emergencyAnswer } from '@/lib/ai/emergency';
import type { AssistantAnswer } from '@/lib/ai/schemas';
import { makeBundle, makeFinding } from './fixtures';

const REVIEW_BYPASSES: string[] = [
  'Take four units of Humalog before dinner.',
  'Take 4 U of insulin with dinner.',
  'Take ٤ units before meals.', // Arabic-Indic four
  'Take ４ units before meals.', // fullwidth four
  'Try 2 more units at dinner.',
  'Inject 6 units before breakfast.',
  'Use 1 unit per 10 g of carbs.',
  'Your insulin-to-carb ratio should be 1:10.',
  'Skip your evening insulin on active days.',
  'Double your Lantus.',
  'Hold your metformin tomorrow.',
  'Cut back your glipizide.',
  'Bump up your Lantus by 2.',
  'A dose of 10 units at bedtime would help.',
  'Take 500mg of metformin twice daily.',
  'Take 0.5 mg Ozempic weekly.',
  'Augmentez votre insuline de 2 unités.',
  'Aumenta tu insulina 2 unidades.',
  'Increase your in​sulin by 2.', // zero-width space inside the word
  'Increase your Lantus by 2.',
  'Eat 15 g of fast carbs and recheck in 15 minutes.',
  "You don't need a doctor for this.",
  "There's no need to see a doctor.",
  'This pattern means you’re diabetic.',
  'Your readings indicate type 2 diabetes.',
  'Evening walks cause your post-dinner glucose to drop.',
  'Taking 2 extra units with high-carb meals is reasonable.',
];

const MORE_UNSAFE: string[] = [
  'Take twenty-two units of Tresiba at night.',
  'Prenez deux unités avant le repas.',
  'Your correction factor looks like 1 unit per 50 mg/dL.',
  'Pre-bolus 15 minutes before breakfast.',
  'Your basal rate may be too high.',
  'Your Toujeo dose should be lowered.',
  'Metformin 1000 twice a day is typical.',
  'Your highs are due to the late snacks you eat.',
  'Skipping breakfast leads to lower glucose for you.',
];

const BENIGN: string[] = [
  'Your readings after dinner were higher on 4 of 6 days.',
  'Your average glucose was 142 mg/dL over the last 30 days.',
  'Your average fasting reading was 6.1 mmol/L on weekdays vs 5.8 mmol/L on weekends.',
  'In your logs, lower-carb meals (<30g, n=12) averaged 150 mg/dL 1-3h after eating.',
  'This appears to be associated with later bedtimes on weekends.',
  'A pattern worth discussing with your healthcare provider is your post-breakfast rise.',
  'Your overnight readings around 1:30 am were lower on 3 of 10 nights.',
  'Days with an evening walk had an average post-dinner reading 18 mg/dL lower.',
  'This does not show that walking causes lower glucose.',
  'You logged 45 g of carbs at lunch on average.',
  'Could evening walks be linked to your lower post-dinner readings?',
  'Keep logging glucose readings, meals, and activity consistently for a few more days.',
  'Time in range was 72% this week, up from 65% last week.',
  'Carb estimates you log may not exactly match what was eaten.',
];

describe('medical-safety filter: review bypass strings', () => {
  it('has all 27 strings from the review', () => {
    expect(REVIEW_BYPASSES).toHaveLength(27);
  });

  it.each(REVIEW_BYPASSES)('catches %j', (text) => {
    expect(findUnsafeLanguage(text).length).toBeGreaterThan(0);
  });

  it.each(MORE_UNSAFE)('catches %j', (text) => {
    expect(findUnsafeLanguage(text).length).toBeGreaterThan(0);
  });
});

describe('medical-safety filter: ordinary data language', () => {
  it.each(BENIGN)('passes %j', (text) => {
    expect(findUnsafeLanguage(text)).toEqual([]);
  });

  it('passes the emergency template, which must contain no dosing', () => {
    const answer = emergencyAnswer();
    const texts = [answer.shortAnswer, ...answer.detail];
    expect(checkTextsSafety(texts)).toEqual({ safe: true, matchedPatterns: [] });
  });
});

describe('normalizeForSafety', () => {
  it('folds fullwidth and Arabic-Indic digits, zero-width characters, accents and curly quotes', () => {
    expect(normalizeForSafety('４ ٤ in​sulin unités you’re')).toBe("4 4 insulin unites you're");
  });
});

describe('checkTextsSafety echo exemption', () => {
  it("exempts a verbatim piece of the user's input but checks anything rephrased", () => {
    const input = 'Pasta for dinner, took 4 units with it';
    expect(checkTextsSafety(['took 4 units with it'], { echoSource: input }).safe).toBe(true);
    expect(checkTextsSafety(['Take 4 units with pasta'], { echoSource: input }).safe).toBe(false);
  });

  it('does not exempt a fragment cut out of a clause', () => {
    // "take 10 units" is a substring of what the user wrote, but it inverts it.
    const input = "Pasta for dinner. My nurse said don't take 10 units before bed.";
    expect(checkTextsSafety(['take 10 units'], { echoSource: input }).safe).toBe(false);
    expect(
      checkTextsSafety(["My nurse said don't take 10 units before bed."], { echoSource: input })
        .safe,
    ).toBe(true);
  });

  it('exempts the whole input and whole clauses, ignoring edge punctuation and spacing', () => {
    const input = 'BG 2.8 and shaking; took 4.5 units at lunch!\nhad juice';
    for (const echo of [input, 'took 4.5 units at lunch', '  Took 4.5 units at lunch. ']) {
      expect(checkTextsSafety([echo], { echoSource: input }).safe).toBe(true);
    }
    // A decimal point is not a clause boundary, so "5 units at lunch" is a fragment.
    expect(checkTextsSafety(['5 units at lunch'], { echoSource: input }).safe).toBe(
      checkTextsSafety(['5 units at lunch']).safe,
    );
  });
});

function answer(overrides: Partial<AssistantAnswer> = {}): AssistantAnswer {
  return {
    shortAnswer: 'Your data shows a mild rise after breakfast.',
    detail: ['Your data shows a mild rise after breakfast.'],
    citedFindingIds: ['finding-1'],
    confidence: 'high',
    suggestedQuestionsForClinician: [],
    notEnoughData: false,
    ...overrides,
  };
}

describe('enforceConfidenceCap', () => {
  it.each([
    ['EARLY', 'low'],
    ['EMERGING', 'moderate'],
    ['CONSISTENT', 'high'],
  ] as const)('caps confidence at the grade of a cited %s finding', (level, expected) => {
    const bundle = makeBundle({ findings: [makeFinding({ evidenceLevel: level })] });
    expect(enforceConfidenceCap(answer(), bundle).value.confidence).toBe(expected);
  });

  it('never raises a confidence the model stated below the cap', () => {
    const bundle = makeBundle({ findings: [makeFinding({ evidenceLevel: 'CONSISTENT' })] });
    expect(enforceConfidenceCap(answer({ confidence: 'low' }), bundle).value.confidence).toBe(
      'low',
    );
  });

  it('drops INSUFFICIENT citations and downgrades when nothing else is cited', () => {
    const bundle = makeBundle({
      findings: [
        makeFinding({ id: 'weak', evidenceLevel: 'INSUFFICIENT' }),
        makeFinding({ id: 'strong', evidenceLevel: 'CONSISTENT' }),
      ],
    });
    const result = enforceConfidenceCap(answer({ citedFindingIds: ['weak'] }), bundle);
    expect(result.value.citedFindingIds).toEqual([]);
    expect(result.value.confidence).toBe('low');
    expect(result.value.detail).toContain(LIMITED_EVIDENCE_CAVEAT);
  });

  it('uses the best cited grade when several findings are cited', () => {
    const bundle = makeBundle({
      findings: [
        makeFinding({ id: 'a', evidenceLevel: 'EARLY' }),
        makeFinding({ id: 'b', evidenceLevel: 'EMERGING' }),
      ],
    });
    const result = enforceConfidenceCap(answer({ citedFindingIds: ['a', 'b'] }), bundle);
    expect(result.value.confidence).toBe('moderate');
  });
});

describe('applyAssistantAnswerGuardrails: causal language', () => {
  it('rejects a causal claim about glucose', () => {
    const result = applyAssistantAnswerGuardrails(
      answer({ detail: ['Your morning walks cause lower readings at lunch.'] }),
      makeBundle(),
    );
    expect(result.rejected).toBe(true);
  });

  it('accepts a causal question for a clinician', () => {
    const result = applyAssistantAnswerGuardrails(
      answer({
        suggestedQuestionsForClinician: ['Could my morning walks be causing my lower readings?'],
      }),
      makeBundle(),
    );
    expect(result.rejected).toBe(false);
  });
});
