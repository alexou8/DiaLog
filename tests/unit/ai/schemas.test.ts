import { describe, it, expect } from 'vitest';
import {
  AssistantAnswerSchema,
  MealParseSchema,
  WeeklyNarrativeSchema,
  NoteStructureSchema,
} from '@/lib/ai/schemas';

describe('AssistantAnswerSchema', () => {
  const good = {
    shortAnswer: 'Your data shows a mild rise after breakfast.',
    detail: ['detail line'],
    citedFindingIds: ['f1'],
    confidence: 'moderate',
    suggestedQuestionsForClinician: [],
    notEnoughData: false,
  };

  it('parses a good payload', () => {
    expect(AssistantAnswerSchema.safeParse(good).success).toBe(true);
  });

  it('rejects extra-long shortAnswer', () => {
    expect(AssistantAnswerSchema.safeParse({ ...good, shortAnswer: 'a'.repeat(300) }).success).toBe(
      false,
    );
  });

  it('rejects a bad confidence value', () => {
    expect(AssistantAnswerSchema.safeParse({ ...good, confidence: 'super-sure' }).success).toBe(
      false,
    );
  });

  it('rejects missing fields', () => {
    const { notEnoughData: _drop, ...rest } = good;
    expect(AssistantAnswerSchema.safeParse(rest).success).toBe(false);
  });
});

describe('MealParseSchema', () => {
  const good = {
    meals: [
      {
        description: 'oatmeal with berries',
        mealType: 'breakfast',
        estimatedCarbsG: 40,
        estimatedProteinG: 8,
        estimatedFatG: 5,
        estimatedFiberG: 6,
        estimatedCalories: 300,
        timeLocal: '08:15',
        confidence: 'medium',
      },
    ],
    exercise: [
      {
        activity: 'walking',
        durationMin: 30,
        intensity: 'light',
        timeLocal: null,
        confidence: 'low',
      },
    ],
    unparsed: [],
  };

  it('parses a good payload', () => {
    expect(MealParseSchema.safeParse(good).success).toBe(true);
  });

  it('rejects a bad mealType', () => {
    const bad = { ...good, meals: [{ ...good.meals[0], mealType: 'brunch' }] };
    expect(MealParseSchema.safeParse(bad).success).toBe(false);
  });

  it('rejects a malformed timeLocal', () => {
    const bad = { ...good, meals: [{ ...good.meals[0], timeLocal: '9am' }] };
    expect(MealParseSchema.safeParse(bad).success).toBe(false);
  });

  it('rejects negative estimated grams', () => {
    const bad = { ...good, meals: [{ ...good.meals[0], estimatedCarbsG: -5 }] };
    expect(MealParseSchema.safeParse(bad).success).toBe(false);
  });
});

describe('WeeklyNarrativeSchema', () => {
  const good = {
    headline: 'A steady week',
    sections: [{ heading: 'Overview', body: 'Things were stable.' }],
    whatChanged: 'Nothing major.',
    whatWentWell: 'Consistent logging.',
    whatToExploreNext: 'Keep it up.',
    questionsForClinician: [],
  };

  it('parses a good payload', () => {
    expect(WeeklyNarrativeSchema.safeParse(good).success).toBe(true);
  });

  it('rejects a section missing a body', () => {
    const bad = { ...good, sections: [{ heading: 'Overview' }] };
    expect(WeeklyNarrativeSchema.safeParse(bad).success).toBe(false);
  });
});

describe('NoteStructureSchema', () => {
  it('parses a good payload', () => {
    const good = {
      candidates: [
        { type: 'meal', summary: 'toast for breakfast', fields: { carbs: 20 }, confidence: 'low' },
      ],
      unparsed: ['felt tired'],
    };
    expect(NoteStructureSchema.safeParse(good).success).toBe(true);
  });

  it('rejects an invalid candidate type', () => {
    const bad = {
      candidates: [{ type: 'vitals', summary: 'x', fields: {}, confidence: 'low' }],
      unparsed: [],
    };
    expect(NoteStructureSchema.safeParse(bad).success).toBe(false);
  });
});
