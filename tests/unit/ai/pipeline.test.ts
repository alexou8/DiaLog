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
      externalConsent: true,
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
      externalConsent: true,
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
      externalConsent: true,
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
      externalConsent: true,
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
        externalConsent: true,
      }),
    ).rejects.toThrow(/raw records/i);
  });
});

describe('parseMealText', () => {
  it('runs against the local provider and returns the text unparsed', async () => {
    const result = await parseMealText({
      text: 'ate a sandwich',
      provider: new LocalProvider(),
      externalConsent: true,
    });
    expect(result.parse.meals).toEqual([]);
    expect(result.parse.unparsed).toEqual(['ate a sandwich']);
  });

  it('falls back to local on a throwing provider', async () => {
    const result = await parseMealText({
      text: 'ate a sandwich',
      provider: new ThrowingProvider(),
      externalConsent: true,
    });
    expect(result.usedFallback).toBe(true);
    expect(result.providerId).toBe('local');
  });

  it('passes through a well-formed structured result from a good provider', async () => {
    const result = await parseMealText({
      text: 'something weird',
      provider: new GoodProvider(),
      externalConsent: true,
    });
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
      externalConsent: true,
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
      externalConsent: true,
    });
    expect(result.usedFallback).toBe(true);
  });
});

describe('parseNoteText', () => {
  it('runs against the local provider and returns the text unparsed', async () => {
    const result = await parseNoteText({
      text: 'felt dizzy after lunch',
      provider: new LocalProvider(),
      externalConsent: true,
    });
    expect(result.structure.candidates).toEqual([]);
    expect(result.structure.unparsed).toEqual(['felt dizzy after lunch']);
  });

  it('falls back to local on a throwing provider', async () => {
    const result = await parseNoteText({
      text: 'felt dizzy',
      provider: new ThrowingProvider(),
      externalConsent: true,
    });
    expect(result.usedFallback).toBe(true);
  });
});

/** An external provider that records every request and answers with a fixed payload per task. */
class RecordingExternalProvider implements AIProvider {
  id = 'recording';
  name = 'Recording';
  isExternal = true;
  calls: CompletionRequest[] = [];
  constructor(
    private readonly payloads: Partial<
      Record<'answer' | 'meal' | 'narrative' | 'note', unknown>
    > = {},
  ) {}
  available(): boolean {
    return true;
  }
  async complete(req: CompletionRequest): Promise<CompletionResult> {
    this.calls.push(req);
    const props = Object.keys((req.responseSchema as { properties?: object }).properties ?? {});
    const task = props.includes('meals')
      ? 'meal'
      : props.includes('headline')
        ? 'narrative'
        : props.includes('candidates')
          ? 'note'
          : 'answer';
    const json = this.payloads[task];
    if (json === undefined) throw new AIProviderError(this.id, 'http_error', 'no payload', 500);
    return { json, providerId: this.id };
  }
}

const groundedAnswerPayload = {
  shortAnswer: 'Your data shows a mild rise after breakfast.',
  detail: ['Your data shows a mild rise after breakfast.'],
  citedFindingIds: ['finding-1'],
  confidence: 'moderate',
  suggestedQuestionsForClinician: [],
  notEnoughData: false,
};

describe('consent gate inside the pipeline', () => {
  it('never calls an external provider without consent, for any task', async () => {
    const provider = new RecordingExternalProvider({
      answer: groundedAnswerPayload,
      meal: { meals: [], exercise: [], unparsed: [] },
      narrative: {
        headline: 'x',
        sections: [],
        whatChanged: '',
        whatWentWell: '',
        whatToExploreNext: '',
        questionsForClinician: [],
      },
      note: { candidates: [], unparsed: [] },
    });
    const bundle = makeBundle();
    const answer = await answerQuestion({
      question: 'I took 20u Lantus, what happens after breakfast?',
      bundle,
      detailLevel: 'standard',
      provider,
      externalConsent: false,
    });
    const meal = await parseMealText({ text: 'toast and jam', provider, externalConsent: false });
    const narrative = await summarizePeriod({
      bundle,
      detailLevel: 'standard',
      provider,
      externalConsent: false,
    });
    const note = await parseNoteText({ text: 'felt dizzy', provider, externalConsent: false });

    expect(provider.calls).toHaveLength(0);
    for (const result of [answer, meal, narrative, note]) {
      expect(result.providerId).toBe('local');
      expect(result.usedFallback).toBe(false);
    }
  });

  it('calls the external provider, with the question, once the user has consented', async () => {
    const provider = new RecordingExternalProvider({ answer: groundedAnswerPayload });
    const result = await answerQuestion({
      question: 'What happens after breakfast?',
      bundle: makeBundle(),
      detailLevel: 'standard',
      provider,
      externalConsent: true,
    });
    expect(provider.calls).toHaveLength(1);
    expect(provider.calls[0]!.messages[0]!.content).toBe('What happens after breakfast?');
    expect(result.providerId).toBe('recording');
  });
});

describe('emergency questions', () => {
  it('returns the fixed safety answer without calling any provider', async () => {
    const provider = new RecordingExternalProvider();
    const result = await answerQuestion({
      question: "My sugar is 45 and I'm shaking, what do I do?",
      bundle: makeBundle(),
      detailLevel: 'standard',
      provider,
      externalConsent: true,
    });
    expect(result.emergency).toBe(true);
    expect(provider.calls).toHaveLength(0);
    expect(result.answer.shortAnswer).toMatch(/emergency/i);
  });

  it('answers an emergency even when the bundle would fail the raw-record check', async () => {
    const bundle = makeBundle();
    // @ts-expect-error deliberately violating the type to test ordering
    bundle.summary.rawReadings = [{ mgdl: 100 }];
    const result = await answerQuestion({
      question: 'my husband is unconscious',
      bundle,
      detailLevel: 'standard',
      provider: new LocalProvider(),
      externalConsent: false,
    });
    expect(result.emergency).toBe(true);
  });
});

describe('emergency quick-log text', () => {
  it('flags it with the safety template and never calls the provider', async () => {
    const provider = new RecordingExternalProvider();
    const result = await parseMealText({
      text: 'BG 2.8 and shaking, had juice',
      provider,
      externalConsent: true,
    });
    expect(result.emergency).toBe(true);
    expect(result.emergencyAnswer?.shortAnswer).toMatch(/emergency/i);
    expect(provider.calls).toHaveLength(0);
    expect(result.providerId).toBe('local');
  });

  it('leaves ordinary quick-log text unflagged', async () => {
    const result = await parseMealText({
      text: 'toast and jam',
      provider: new LocalProvider(),
      externalConsent: false,
    });
    expect(result.emergency).toBe(false);
    expect(result).not.toHaveProperty('emergencyAnswer');
  });
});

describe('confidence cap through the pipeline', () => {
  it('clamps a model claiming high confidence on an EARLY finding', async () => {
    const provider = new RecordingExternalProvider({
      answer: { ...groundedAnswerPayload, confidence: 'high' },
    });
    const result = await answerQuestion({
      question: 'What happens after breakfast?',
      bundle: makeBundle({ findings: [makeFinding({ evidenceLevel: 'EARLY' })] }),
      detailLevel: 'standard',
      provider,
      externalConsent: true,
    });
    expect(result.usedFallback).toBe(false);
    expect(result.answer.confidence).toBe('low');
  });
});

describe('safety filter on meal, narrative and note output', () => {
  it('rejects a meal parse whose model-written text carries dosing advice', async () => {
    const provider = new RecordingExternalProvider({
      meal: {
        meals: [
          {
            description: 'Pasta. Take 4 units of Humalog with it.',
            mealType: 'dinner',
            estimatedCarbsG: 80,
            estimatedProteinG: 12,
            estimatedFatG: 10,
            estimatedFiberG: 4,
            estimatedCalories: 600,
            timeLocal: null,
            confidence: 'medium',
          },
        ],
        exercise: [],
        unparsed: [],
      },
    });
    const result = await parseMealText({
      text: 'pasta for dinner',
      provider,
      externalConsent: true,
    });
    expect(result.usedFallback).toBe(true);
    expect(result.parse.meals).toEqual([]);
    expect(result.parse.unparsed).toEqual(['pasta for dinner']);
  });

  it("does not reject the user's own words echoed back in unparsed", async () => {
    const text = 'pasta for dinner, took 4 units with it';
    const provider = new RecordingExternalProvider({
      meal: { meals: [], exercise: [], unparsed: ['took 4 units with it'] },
    });
    const result = await parseMealText({ text, provider, externalConsent: true });
    expect(result.usedFallback).toBe(false);
    expect(result.providerId).toBe('recording');
  });

  it('rejects a weekly narrative with causal or dosing language', async () => {
    const provider = new RecordingExternalProvider({
      narrative: {
        headline: 'A good week',
        sections: [{ heading: 'Walks', body: 'Evening walks caused your glucose to drop.' }],
        whatChanged: 'Less variability.',
        whatWentWell: 'Consistent logging.',
        whatToExploreNext: 'Lower your basal insulin on walk days.',
        questionsForClinician: [],
      },
    });
    const result = await summarizePeriod({
      bundle: makeBundle(),
      detailLevel: 'standard',
      provider,
      externalConsent: true,
    });
    expect(result.usedFallback).toBe(true);
    expect(result.providerId).toBe('local');
  });

  it('rejects a medication candidate carrying a dose the user never wrote', async () => {
    const provider = new RecordingExternalProvider({
      note: {
        candidates: [
          {
            type: 'medication',
            summary: 'Morning insulin',
            fields: [{ key: 'units', value: 10 }],
            confidence: 'low',
          },
        ],
        unparsed: [],
      },
    });
    const result = await parseNoteText({
      text: 'took my morning insulin',
      provider,
      externalConsent: true,
    });
    expect(result.usedFallback).toBe(true);
    expect(result.structure.candidates).toEqual([]);
  });
});

describe('local provider task dispatch', () => {
  it('parses meal and note text locally in one call, without reporting a fallback', async () => {
    const meal = await parseMealText({
      text: 'ate a sandwich',
      provider: new LocalProvider(),
      externalConsent: false,
    });
    const note = await parseNoteText({
      text: 'felt dizzy',
      provider: new LocalProvider(),
      externalConsent: false,
    });
    expect(meal.usedFallback).toBe(false);
    expect(note.usedFallback).toBe(false);
  });
});
