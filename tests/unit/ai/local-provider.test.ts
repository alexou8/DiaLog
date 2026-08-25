import { describe, it, expect } from 'vitest';
import { LocalProvider } from '@/lib/ai/providers/local';
import {
  buildAnswerQuestionSystemPrompt,
  buildWeeklyNarrativeSystemPrompt,
} from '@/lib/ai/prompts';
import {
  AssistantAnswerJsonSchema,
  WeeklyNarrativeJsonSchema,
  MealParseJsonSchema,
} from '@/lib/ai/schemas';
import { AssistantAnswerSchema, WeeklyNarrativeSchema, MealParseSchema } from '@/lib/ai/schemas';
import { makeBundle, makeFinding, makeInsufficientBundle } from './fixtures';

describe('LocalProvider', () => {
  it('reports available() true and isExternal false', () => {
    const provider = new LocalProvider();
    expect(provider.available()).toBe(true);
    expect(provider.isExternal).toBe(false);
    expect(provider.id).toBe('local');
  });

  it('answers a relevant question by picking findings that keyword-match', async () => {
    const bundle = makeBundle({
      findings: [
        makeFinding({
          id: 'breakfast-1',
          kind: 'post_meal_spike',
          statement: 'Your glucose tends to rise after breakfast.',
          evidenceLevel: 'CONSISTENT',
          sampleSize: 40,
        }),
        makeFinding({
          id: 'exercise-1',
          kind: 'exercise_effect',
          statement: 'Evening walks are associated with lower overnight readings.',
          evidenceLevel: 'EMERGING',
          sampleSize: 20,
        }),
      ],
    });
    const provider = new LocalProvider();
    const req = {
      system: buildAnswerQuestionSystemPrompt(bundle, 'standard'),
      messages: [{ role: 'user' as const, content: 'What happens to my glucose after breakfast?' }],
      responseSchema: AssistantAnswerJsonSchema,
      maxTokens: 1000,
    };
    const result = await provider.complete(req);
    const parsed = AssistantAnswerSchema.parse(result.json);
    expect(parsed.notEnoughData).toBe(false);
    expect(parsed.citedFindingIds).toContain('breakfast-1');
    expect(parsed.citedFindingIds).not.toContain('exercise-1');
    expect(parsed.shortAnswer.length).toBeLessThanOrEqual(280);
  });

  it('produces the not-enough-data path when nothing relevant is found', async () => {
    const bundle = makeBundle({
      findings: [
        makeFinding({
          id: 'unrelated',
          kind: 'sleep_pattern',
          statement: 'Sleep duration is stable.',
          evidenceLevel: 'CONSISTENT',
        }),
      ],
    });
    const provider = new LocalProvider();
    const req = {
      system: buildAnswerQuestionSystemPrompt(bundle, 'standard'),
      messages: [{ role: 'user' as const, content: 'How does chocolate cake affect my readings?' }],
      responseSchema: AssistantAnswerJsonSchema,
      maxTokens: 1000,
    };
    const result = await provider.complete(req);
    const parsed = AssistantAnswerSchema.parse(result.json);
    expect(parsed.notEnoughData).toBe(true);
    expect(parsed.citedFindingIds).toEqual([]);
    expect(parsed.shortAnswer).toMatch(/don't have enough evidence/i);
    expect(parsed.detail.length).toBeGreaterThan(1);
  });

  it('produces the not-enough-data path when all findings are INSUFFICIENT', async () => {
    const bundle = makeInsufficientBundle();
    const provider = new LocalProvider();
    const req = {
      system: buildAnswerQuestionSystemPrompt(bundle, 'standard'),
      messages: [{ role: 'user' as const, content: 'Tell me about post_meal_spike patterns.' }],
      responseSchema: AssistantAnswerJsonSchema,
      maxTokens: 1000,
    };
    const result = await provider.complete(req);
    const parsed = AssistantAnswerSchema.parse(result.json);
    expect(parsed.notEnoughData).toBe(true);
  });

  it('builds a deterministic weekly narrative from findings', async () => {
    const bundle = makeBundle({
      findings: [
        makeFinding({
          id: 'a',
          evidenceLevel: 'CONSISTENT',
          statement: 'Weekday mornings run higher.',
        }),
        makeFinding({ id: 'b', evidenceLevel: 'EARLY', statement: 'Weekend evenings look lower.' }),
      ],
    });
    const provider = new LocalProvider();
    const req = {
      system: buildWeeklyNarrativeSystemPrompt(bundle, 'standard'),
      messages: [{ role: 'user' as const, content: 'Summarize this period.' }],
      responseSchema: WeeklyNarrativeJsonSchema,
      maxTokens: 1000,
    };
    const result = await provider.complete(req);
    const parsed = WeeklyNarrativeSchema.parse(result.json);
    expect(parsed.headline).toContain('Weekday mornings run higher');
    expect(parsed.sections.length).toBeGreaterThan(0);
  });

  it('produces the not-enough-data narrative when there is no usable evidence', async () => {
    const bundle = makeInsufficientBundle();
    const provider = new LocalProvider();
    const req = {
      system: buildWeeklyNarrativeSystemPrompt(bundle, 'standard'),
      messages: [{ role: 'user' as const, content: 'Summarize this period.' }],
      responseSchema: WeeklyNarrativeJsonSchema,
      maxTokens: 1000,
    };
    const result = await provider.complete(req);
    const parsed = WeeklyNarrativeSchema.parse(result.json);
    expect(parsed.headline).toMatch(/not enough data/i);
    expect(parsed.sections).toEqual([]);
  });

  it('never fabricates meal-parse numbers — returns everything unparsed', async () => {
    const bundle = makeBundle();
    const provider = new LocalProvider();
    const req = {
      system: buildAnswerQuestionSystemPrompt(bundle, 'standard'),
      messages: [{ role: 'user' as const, content: 'had a bowl of cereal and a banana' }],
      responseSchema: MealParseJsonSchema,
      maxTokens: 1000,
    };
    const result = await provider.complete(req);
    const parsed = MealParseSchema.parse(result.json);
    expect(parsed.meals).toEqual([]);
    expect(parsed.unparsed).toEqual(['had a bowl of cereal and a banana']);
  });

  it('falls back safely when no evidence bundle marker is present in the system prompt', async () => {
    const provider = new LocalProvider();
    const req = {
      system: 'no bundle embedded here',
      messages: [{ role: 'user' as const, content: 'anything?' }],
      responseSchema: AssistantAnswerJsonSchema,
      maxTokens: 1000,
    };
    const result = await provider.complete(req);
    const parsed = AssistantAnswerSchema.parse(result.json);
    expect(parsed.notEnoughData).toBe(true);
  });
});
