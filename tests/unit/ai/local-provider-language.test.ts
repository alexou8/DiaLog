import { describe, expect, it } from 'vitest';
import { LocalProvider, __internal } from '@/lib/ai/providers/local';
import { buildAnswerQuestionSystemPrompt, buildMealParseSystemPrompt } from '@/lib/ai/prompts';
import {
  AssistantAnswerJsonSchema,
  AssistantAnswerSchema,
  MealParseJsonSchema,
  MealParseSchema,
} from '@/lib/ai/schemas';
import { makeBundle, makeFinding, makeInsufficientBundle } from './fixtures';

async function ask(question: string, bundle = makeBundle()) {
  const result = await new LocalProvider().complete({
    system: buildAnswerQuestionSystemPrompt(bundle, 'standard'),
    messages: [{ role: 'user', content: question }],
    responseSchema: AssistantAnswerJsonSchema,
    maxTokens: 1000,
  });
  return AssistantAnswerSchema.parse(result.json);
}

describe('LocalProvider: questions not in English', () => {
  it('keeps accented letters when tokenizing', () => {
    expect(__internal.tokenize('Pourquoi ma glycémie monte?')).toContain('glycémie');
  });

  it.each([
    'Pourquoi ma glycémie est-elle haute après le dîner ?',
    '¿Por qué mi glucosa sube después de la cena?',
    'Warum ist mein Blutzucker nach dem Essen so hoch?',
    '为什么我的血糖在早餐后升高？',
  ])('gives a general summary rather than "not enough evidence" for %j', async (question) => {
    const answer = await ask(question);
    expect(answer.notEnoughData).toBe(false);
    expect(answer.shortAnswer).toMatch(/only read questions in English/i);
    expect(answer.citedFindingIds).toEqual(['finding-1']);
  });

  it('still says "not enough evidence" for an unrelated English question', async () => {
    const bundle = makeBundle({
      findings: [
        makeFinding({ id: 'sleep', kind: 'sleep_pattern', statement: 'Sleep is stable.' }),
      ],
    });
    const answer = await ask('How does chocolate cake affect my readings?', bundle);
    expect(answer.notEnoughData).toBe(true);
  });

  it('falls back to "not enough evidence" when a non-English question meets an all-INSUFFICIENT bundle', async () => {
    const answer = await ask('Pourquoi ma glycémie est-elle haute ?', makeInsufficientBundle());
    expect(answer.notEnoughData).toBe(true);
  });
});

describe('LocalProvider: meal parsing without a bundle', () => {
  it('returns a meal-parse shape even when the prompt carries no evidence bundle', async () => {
    const result = await new LocalProvider().complete({
      system: buildMealParseSystemPrompt(),
      messages: [{ role: 'user', content: 'toast and jam' }],
      responseSchema: MealParseJsonSchema,
      maxTokens: 1000,
    });
    expect(MealParseSchema.parse(result.json)).toEqual({
      meals: [],
      exercise: [],
      unparsed: ['toast and jam'],
    });
  });
});
