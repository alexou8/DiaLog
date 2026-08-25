/**
 * zod schemas (runtime validation) plus hand-written JSON Schema mirrors
 * (sent to LLM providers so they can constrain generation) for every
 * structured output the AI layer produces. Keep the two in sync by hand —
 * zod does not currently export a lossless JSON Schema of its own that
 * matches what Anthropic tool `input_schema` / OpenAI `json_schema` expect,
 * so schema drift is caught by the "matches fixtures" unit tests instead.
 */
import { z } from 'zod';

// ---------------------------------------------------------------------------
// AssistantAnswer
// ---------------------------------------------------------------------------

export const CONFIDENCE_LEVELS = ['insufficient', 'low', 'moderate', 'high'] as const;
export type ConfidenceLevel = (typeof CONFIDENCE_LEVELS)[number];

export const AssistantAnswerSchema = z.object({
  shortAnswer: z.string().max(280),
  detail: z.array(z.string()),
  citedFindingIds: z.array(z.string()),
  confidence: z.enum(CONFIDENCE_LEVELS),
  suggestedQuestionsForClinician: z.array(z.string()),
  notEnoughData: z.boolean(),
});
export type AssistantAnswer = z.infer<typeof AssistantAnswerSchema>;

export const AssistantAnswerJsonSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    shortAnswer: { type: 'string', maxLength: 280 },
    detail: { type: 'array', items: { type: 'string' } },
    citedFindingIds: { type: 'array', items: { type: 'string' } },
    confidence: { type: 'string', enum: [...CONFIDENCE_LEVELS] },
    suggestedQuestionsForClinician: { type: 'array', items: { type: 'string' } },
    notEnoughData: { type: 'boolean' },
  },
  required: [
    'shortAnswer',
    'detail',
    'citedFindingIds',
    'confidence',
    'suggestedQuestionsForClinician',
    'notEnoughData',
  ],
} as const;

// ---------------------------------------------------------------------------
// MealParse
// ---------------------------------------------------------------------------

export const MEAL_TYPES = ['breakfast', 'lunch', 'dinner', 'snack', 'unknown'] as const;
export const INTENSITY_LEVELS = ['light', 'moderate', 'vigorous'] as const;
export const PARSE_CONFIDENCE = ['low', 'medium', 'high'] as const;

const timeLocalSchema = z
  .string()
  .regex(/^([01]\d|2[0-3]):[0-5]\d$/)
  .nullable();

export const MealItemSchema = z.object({
  description: z.string(),
  mealType: z.enum(MEAL_TYPES),
  /** ESTIMATE: model-inferred, not measured. */
  estimatedCarbsG: z.number().min(0),
  /** ESTIMATE: model-inferred, not measured. */
  estimatedProteinG: z.number().min(0),
  /** ESTIMATE: model-inferred, not measured. */
  estimatedFatG: z.number().min(0),
  /** ESTIMATE: model-inferred, not measured. */
  estimatedFiberG: z.number().min(0),
  /** ESTIMATE: model-inferred, not measured. */
  estimatedCalories: z.number().min(0),
  timeLocal: timeLocalSchema,
  confidence: z.enum(PARSE_CONFIDENCE),
});

export const ExerciseItemSchema = z.object({
  activity: z.string(),
  durationMin: z.number().min(0),
  intensity: z.enum(INTENSITY_LEVELS),
  timeLocal: timeLocalSchema,
  confidence: z.enum(PARSE_CONFIDENCE),
});

export const MealParseSchema = z.object({
  meals: z.array(MealItemSchema),
  exercise: z.array(ExerciseItemSchema),
  unparsed: z.array(z.string()),
});
export type MealParse = z.infer<typeof MealParseSchema>;

export const MealParseJsonSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    meals: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          description: { type: 'string' },
          mealType: { type: 'string', enum: [...MEAL_TYPES] },
          estimatedCarbsG: { type: 'number', minimum: 0 },
          estimatedProteinG: { type: 'number', minimum: 0 },
          estimatedFatG: { type: 'number', minimum: 0 },
          estimatedFiberG: { type: 'number', minimum: 0 },
          estimatedCalories: { type: 'number', minimum: 0 },
          timeLocal: { type: ['string', 'null'], pattern: '^([01]\\d|2[0-3]):[0-5]\\d$' },
          confidence: { type: 'string', enum: [...PARSE_CONFIDENCE] },
        },
        required: [
          'description',
          'mealType',
          'estimatedCarbsG',
          'estimatedProteinG',
          'estimatedFatG',
          'estimatedFiberG',
          'estimatedCalories',
          'timeLocal',
          'confidence',
        ],
      },
    },
    exercise: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          activity: { type: 'string' },
          durationMin: { type: 'number', minimum: 0 },
          intensity: { type: 'string', enum: [...INTENSITY_LEVELS] },
          timeLocal: { type: ['string', 'null'], pattern: '^([01]\\d|2[0-3]):[0-5]\\d$' },
          confidence: { type: 'string', enum: [...PARSE_CONFIDENCE] },
        },
        required: ['activity', 'durationMin', 'intensity', 'timeLocal', 'confidence'],
      },
    },
    unparsed: { type: 'array', items: { type: 'string' } },
  },
  required: ['meals', 'exercise', 'unparsed'],
} as const;

// ---------------------------------------------------------------------------
// WeeklyNarrative
// ---------------------------------------------------------------------------

export const NarrativeSectionSchema = z.object({
  heading: z.string(),
  body: z.string(),
});

export const WeeklyNarrativeSchema = z.object({
  headline: z.string(),
  sections: z.array(NarrativeSectionSchema),
  whatChanged: z.string(),
  whatWentWell: z.string(),
  whatToExploreNext: z.string(),
  questionsForClinician: z.array(z.string()),
});
export type WeeklyNarrative = z.infer<typeof WeeklyNarrativeSchema>;

export const WeeklyNarrativeJsonSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    headline: { type: 'string' },
    sections: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          heading: { type: 'string' },
          body: { type: 'string' },
        },
        required: ['heading', 'body'],
      },
    },
    whatChanged: { type: 'string' },
    whatWentWell: { type: 'string' },
    whatToExploreNext: { type: 'string' },
    questionsForClinician: { type: 'array', items: { type: 'string' } },
  },
  required: [
    'headline',
    'sections',
    'whatChanged',
    'whatWentWell',
    'whatToExploreNext',
    'questionsForClinician',
  ],
} as const;

// ---------------------------------------------------------------------------
// NoteStructure
// ---------------------------------------------------------------------------

export const CANDIDATE_RECORD_TYPES = [
  'glucose',
  'meal',
  'exercise',
  'medication',
  'symptom',
  'note',
] as const;

export const CandidateRecordSchema = z.object({
  type: z.enum(CANDIDATE_RECORD_TYPES),
  summary: z.string(),
  /** Free-form key/value fields extracted for this candidate record, all still estimates. */
  fields: z.record(z.string(), z.union([z.string(), z.number(), z.null()])),
  confidence: z.enum(PARSE_CONFIDENCE),
});

export const NoteStructureSchema = z.object({
  candidates: z.array(CandidateRecordSchema),
  unparsed: z.array(z.string()),
});
export type NoteStructure = z.infer<typeof NoteStructureSchema>;

export const NoteStructureJsonSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    candidates: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          type: { type: 'string', enum: [...CANDIDATE_RECORD_TYPES] },
          summary: { type: 'string' },
          fields: {
            type: 'object',
            additionalProperties: { type: ['string', 'number', 'null'] },
          },
          confidence: { type: 'string', enum: [...PARSE_CONFIDENCE] },
        },
        required: ['type', 'summary', 'fields', 'confidence'],
      },
    },
    unparsed: { type: 'array', items: { type: 'string' } },
  },
  required: ['candidates', 'unparsed'],
} as const;
