import { describe, it, expect } from 'vitest';
import {
  answerQuestion,
  parseMealText,
  summarizePeriod,
  parseNoteText,
  assertNoRawRecords,
} from '@/lib/ai/pipeline';
import { LocalProvider } from '@/lib/ai/providers/local';
import {
  AIProviderError,
  type AIProvider,
  type CompletionRequest,
  type CompletionResult,
} from '@/lib/ai/provider';
import { makeBundle, makeFinding, makeInsufficientBundle } from './fixtures';

class ThrowingProvider implements AIProvider {
  id = 'broken';
  name = 'Broken';
  isExternal = true;
  available(): boolean {
    return true;
  }
  async complete(_req: CompletionRequest): Promise<CompletionResult> {
    throw new AIProviderError(this.id, 'http_error', 'boom', 500);
  }
}

class UnsafeProvider implements AIProvider {
  id = 'unsafe';
  name = 'Unsafe';
  isExternal = true;
  available(): boolean {
    return true;
  }
  async complete(_req: CompletionRequest): Promise<CompletionResult> {
    return {
      json: {
        shortAnswer: 'You should increase your insulin dose.',
        detail: ['You should increase your insulin dose.'],
        citedFindingIds: [],
        confidence: 'high',
        suggestedQuestionsForClinician: [],
        notEnoughData: false,
      },
      providerId: this.id,
    };
  }
}

class GoodProvider implements AIProvider {
  id = 'good';
  name = 'Good';
  isExternal = true;
  available(): boolean {
    return true;
  }
  async complete(req: CompletionRequest): Promise<CompletionResult> {
    const schema = req.responseSchema as { properties?: Record<string, unknown> };
    const props = schema.properties ? Object.keys(schema.properties) : [];
    if (props.includes('meals')) {
      return {
        json: { meals: [], exercise: [], unparsed: ['unrecognized text'] },
        providerId: this.id,
      };
    }
    if (props.includes('candidates')) {
      return { json: { candidates: [], unparsed: ['unrecognized note'] }, providerId: this.id };
    }
    return {
      json: {
        shortAnswer: 'Your data shows a mild rise after breakfast.',
        detail: ['Your data shows a mild rise after breakfast.'],
        citedFindingIds: ['finding-1'],
        confidence: 'moderate',
        suggestedQuestionsForClinician: [],
        notEnoughData: false,
      },
      providerId: this.id,
    };
  }
}

describe('assertNoRawRecords', () => {
  it('accepts a well-formed EvidenceBundle', () => {
    expect(() => assertNoRawRecords(makeBundle())).not.toThrow();
  });

  it('throws when summary contains an array', () => {
    const bundle = makeBundle();
    // @ts-expect-error deliberately violating the type to test the runtime guard
    bundle.summary.rawReadings = [{ mgdl: 100 }, { mgdl: 110 }];
    expect(() => assertNoRawRecords(bundle)).toThrow(/raw records/i);
  });

  it('throws when a finding metrics field contains an array', () => {
    const bundle = makeBundle({
      findings: [
        makeFinding({ metrics: { avgRiseMgdl: 42, rawPoints: [1, 2, 3] as unknown as number } }),
      ],
    });
    expect(() => assertNoRawRecords(bundle)).toThrow(/raw records/i);
  });

  it('throws when findings is not an array of Finding-shaped objects', () => {
    const bundle = makeBundle();
    // @ts-expect-error deliberately violating the type to test the runtime guard
    bundle.findings = [{ notAFinding: true }];
    expect(() => assertNoRawRecords(bundle)).toThrow();
  });
});

describe('answerQuestion', () => {
  it('runs against the local provider end to end', async () => {
    const bundle = makeBundle();
    const result = await answerQuestion({
      question: 'What happens after breakfast?',
      bundle,
      detailLevel: 'standard',
      provider: new LocalProvider(),
    });
    expect(result.providerId).toBe('local');
    expect(result.usedFallback).toBe(false);
    expect(
      result.answer.citedFindingIds.every((id) => bundle.findings.some((f) => f.id === id)),
    ).toBe(true);
  });

  it('falls back to the local provider when the primary provider throws', async () => {
    const bundle = makeBundle();
    const result = await answerQuestion({
      question: 'What happens after breakfast?',
      bundle,
      detailLevel: 'standard',
      provider: new ThrowingProvider(),
    });
    expect(result.usedFallback).toBe(true);
    expect(result.providerId).toBe('local');
  });

  it('falls back to the local provider when the primary provider emits unsafe medical language', async () => {
    const bundle = makeBundle();
    const result = await answerQuestion({
      question: 'What happens after breakfast?',
      bundle,
      detailLevel: 'standard',
      provider: new UnsafeProvider(),
    });
    expect(result.usedFallback).toBe(true);
    expect(result.answer.shortAnswer).not.toMatch(/insulin dose/i);
  });

  it('forces notEnoughData when the bundle has only insufficient evidence, even from a "good" provider', async () => {
    const bundle = makeInsufficientBundle();
    const result = await answerQuestion({
      question: 'What happens after breakfast?',
      bundle,
      detailLevel: 'standard',
      provider: new GoodProvider(),
    });
    expect(result.answer.notEnoughData).toBe(true);
  });

  it('throws before calling the provider when the bundle contains raw records', async () => {
    const bundle = makeBundle();
    // @ts-expect-error deliberately violating the type to test the runtime guard
    bundle.summary.rawReadings = [{ mgdl: 100 }];
    await expect(
      answerQuestion({
        question: 'x',
        bundle,
        detailLevel: 'standard',
        provider: new GoodProvider(),
      }),
    ).rejects.toThrow(/raw records/i);
  });
});

describe('parseMealText', () => {
  it('runs against the local provider and returns the text unparsed', async () => {
    const result = await parseMealText({ text: 'ate a sandwich', provider: new LocalProvider() });
    expect(result.parse.meals).toEqual([]);
    expect(result.parse.unparsed).toEqual(['ate a sandwich']);
  });

  it('falls back to local on a throwing provider', async () => {
    const result = await parseMealText({
      text: 'ate a sandwich',
      provider: new ThrowingProvider(),
    });
    expect(result.usedFallback).toBe(true);
    expect(result.providerId).toBe('local');
  });

  it('passes through a well-formed structured result from a good provider', async () => {
    const result = await parseMealText({ text: 'something weird', provider: new GoodProvider() });
    expect(result.usedFallback).toBe(false);
    expect(result.parse.unparsed).toEqual(['unrecognized text']);
  });
});

describe('summarizePeriod', () => {
  it('runs against the local provider end to end', async () => {
    const bundle = makeBundle();
    const result = await summarizePeriod({
      bundle,
      detailLevel: 'standard',
      provider: new LocalProvider(),
    });
    expect(result.providerId).toBe('local');
    expect(result.narrative.headline).toBeTruthy();
  });

  it('falls back to local on a throwing provider', async () => {
    const bundle = makeBundle();
    const result = await summarizePeriod({
      bundle,
      detailLevel: 'standard',
      provider: new ThrowingProvider(),
    });
    expect(result.usedFallback).toBe(true);
  });
});

describe('parseNoteText', () => {
  it('runs against the local provider and returns the text unparsed', async () => {
    const result = await parseNoteText({
      text: 'felt dizzy after lunch',
      provider: new LocalProvider(),
    });
    expect(result.structure.candidates).toEqual([]);
    expect(result.structure.unparsed).toEqual(['felt dizzy after lunch']);
  });

  it('falls back to local on a throwing provider', async () => {
    const result = await parseNoteText({ text: 'felt dizzy', provider: new ThrowingProvider() });
    expect(result.usedFallback).toBe(true);
  });
});
