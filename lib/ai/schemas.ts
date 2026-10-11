/**
 * zod schemas (runtime validation) plus hand-written JSON Schema mirrors
 * (sent to LLM providers so they can constrain generation) for every
 * structured output the AI layer produces. Keep the two in sync by hand —
 * zod does not currently export a lossless JSON Schema of its own that
 * matches what Anthropic `output_config.format` / OpenAI `json_schema` expect,
 * so schema drift is caught by the "matches fixtures" unit tests instead.
 *
 * Every string and array is capped. An unbounded `detail` or an empty
 * `shortAnswer` with `notEnoughData: false` used to validate (review L2), and
 * the caps also bound how much model text the safety filter has to scan. The
 * providers strip the keywords their structured-output mode does not support
 * (`providers/schema-compat.ts`); zod still enforces them here.
 */
import { z } from 'zod';

/** Caps shared by the zod schemas and their JSON Schema mirrors. */
export const LIMITS = {
  shortAnswer: 280,
  paragraph: 1200,
  /** Local answers wrap a whole finding statement in a clinician question, so this is generous. */
  question: 600,
  label: 200,
  detailItems: 20,
  citedIds: 20,
  clinicianQuestions: 5,
  /** `unparsed` echoes the user's input, which is capped at 600 chars upstream. */
  unparsedText: 1000,
  listItems: 20,
  noteFields: 12,
  fieldKey: 60,
} as const;

// ---------------------------------------------------------------------------
// AssistantAnswer
// ---------------------------------------------------------------------------

export const CONFIDENCE_LEVELS = ['insufficient', 'low', 'moderate', 'high'] as const;
export type ConfidenceLevel = (typeof CONFIDENCE_LEVELS)[number];

export const AssistantAnswerSchema = z.object({
  shortAnswer: z.string().min(1).max(LIMITS.shortAnswer),
  detail: z.array(z.string().max(LIMITS.paragraph)).max(LIMITS.detailItems),
  citedFindingIds: z.array(z.string().max(LIMITS.label)).max(LIMITS.citedIds),
  confidence: z.enum(CONFIDENCE_LEVELS),
  suggestedQuestionsForClinician: z
    .array(z.string().max(LIMITS.question))
    .max(LIMITS.clinicianQuestions),
  notEnoughData: z.boolean(),
});
export type AssistantAnswer = z.infer<typeof AssistantAnswerSchema>;

export const AssistantAnswerJsonSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    shortAnswer: { type: 'string', minLength: 1, maxLength: LIMITS.shortAnswer },
    detail: {
      type: 'array',
      maxItems: LIMITS.detailItems,
      items: { type: 'string', maxLength: LIMITS.paragraph },
    },
    citedFindingIds: {
      type: 'array',
      maxItems: LIMITS.citedIds,
      items: { type: 'string', maxLength: LIMITS.label },
    },
    confidence: { type: 'string', enum: [...CONFIDENCE_LEVELS] },
    suggestedQuestionsForClinician: {
      type: 'array',
      maxItems: LIMITS.clinicianQuestions,
      items: { type: 'string', maxLength: LIMITS.question },
    },
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

/** Upper bounds on estimates: far above any real meal or session, low enough to reject nonsense. */
const MAX_GRAMS = 2000;
const MAX_CALORIES = 20000;
const MAX_DURATION_MIN = 1440;

export const MealItemSchema = z.object({
  description: z.string().max(LIMITS.label),
  mealType: z.enum(MEAL_TYPES),
  /** ESTIMATE: model-inferred, not measured. */
  estimatedCarbsG: z.number().min(0).max(MAX_GRAMS),
  /** ESTIMATE: model-inferred, not measured. */
  estimatedProteinG: z.number().min(0).max(MAX_GRAMS),
  /** ESTIMATE: model-inferred, not measured. */
  estimatedFatG: z.number().min(0).max(MAX_GRAMS),
  /** ESTIMATE: model-inferred, not measured. */
  estimatedFiberG: z.number().min(0).max(MAX_GRAMS),
  /** ESTIMATE: model-inferred, not measured. */
  estimatedCalories: z.number().min(0).max(MAX_CALORIES),
  timeLocal: timeLocalSchema,
  confidence: z.enum(PARSE_CONFIDENCE),
});

export const ExerciseItemSchema = z.object({
  activity: z.string().max(LIMITS.label),
  durationMin: z.number().min(0).max(MAX_DURATION_MIN),
  intensity: z.enum(INTENSITY_LEVELS),
  timeLocal: timeLocalSchema,
  confidence: z.enum(PARSE_CONFIDENCE),
});

export const MealParseSchema = z.object({
  meals: z.array(MealItemSchema).max(LIMITS.listItems),
  exercise: z.array(ExerciseItemSchema).max(LIMITS.listItems),
  unparsed: z.array(z.string().max(LIMITS.unparsedText)).max(LIMITS.listItems),
});
export type MealParse = z.infer<typeof MealParseSchema>;

const TIME_LOCAL_JSON = {
  type: ['string', 'null'],
  pattern: '^([01]\\d|2[0-3]):[0-5]\\d$',
} as const;

export const MealParseJsonSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    meals: {
      type: 'array',
      maxItems: LIMITS.listItems,
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          description: { type: 'string', maxLength: LIMITS.label },
          mealType: { type: 'string', enum: [...MEAL_TYPES] },
          estimatedCarbsG: { type: 'number', minimum: 0, maximum: MAX_GRAMS },
          estimatedProteinG: { type: 'number', minimum: 0, maximum: MAX_GRAMS },
          estimatedFatG: { type: 'number', minimum: 0, maximum: MAX_GRAMS },
          estimatedFiberG: { type: 'number', minimum: 0, maximum: MAX_GRAMS },
          estimatedCalories: { type: 'number', minimum: 0, maximum: MAX_CALORIES },
          timeLocal: TIME_LOCAL_JSON,
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
      maxItems: LIMITS.listItems,
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          activity: { type: 'string', maxLength: LIMITS.label },
          durationMin: { type: 'number', minimum: 0, maximum: MAX_DURATION_MIN },
          intensity: { type: 'string', enum: [...INTENSITY_LEVELS] },
          timeLocal: TIME_LOCAL_JSON,
          confidence: { type: 'string', enum: [...PARSE_CONFIDENCE] },
        },
        required: ['activity', 'durationMin', 'intensity', 'timeLocal', 'confidence'],
      },
    },
    unparsed: {
      type: 'array',
      maxItems: LIMITS.listItems,
      items: { type: 'string', maxLength: LIMITS.unparsedText },
    },
  },
  required: ['meals', 'exercise', 'unparsed'],
} as const;

// ---------------------------------------------------------------------------
// WeeklyNarrative
// ---------------------------------------------------------------------------

export const NarrativeSectionSchema = z.object({
  heading: z.string().max(LIMITS.label),
  body: z.string().max(LIMITS.paragraph),
});

export const WeeklyNarrativeSchema = z.object({
  headline: z.string().min(1).max(LIMITS.paragraph),
  sections: z.array(NarrativeSectionSchema).max(8),
  whatChanged: z.string().max(LIMITS.paragraph),
  whatWentWell: z.string().max(LIMITS.paragraph),
  whatToExploreNext: z.string().max(LIMITS.paragraph),
  questionsForClinician: z.array(z.string().max(LIMITS.question)).max(LIMITS.clinicianQuestions),
});
export type WeeklyNarrative = z.infer<typeof WeeklyNarrativeSchema>;

export const WeeklyNarrativeJsonSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    headline: { type: 'string', minLength: 1, maxLength: LIMITS.paragraph },
    sections: {
      type: 'array',
      maxItems: 8,
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          heading: { type: 'string', maxLength: LIMITS.label },
          body: { type: 'string', maxLength: LIMITS.paragraph },
        },
        required: ['heading', 'body'],
      },
    },
    whatChanged: { type: 'string', maxLength: LIMITS.paragraph },
    whatWentWell: { type: 'string', maxLength: LIMITS.paragraph },
    whatToExploreNext: { type: 'string', maxLength: LIMITS.paragraph },
    questionsForClinician: {
      type: 'array',
      maxItems: LIMITS.clinicianQuestions,
      items: { type: 'string', maxLength: LIMITS.question },
    },
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

/**
 * One extracted key/value field, still an estimate.
 *
 * A list of `{ key, value }` rather than a free-form record: a record is
 * `additionalProperties: { ... }` in JSON Schema, which OpenAI strict mode and
 * Anthropic structured outputs both reject (they require `additionalProperties:
 * false` on every object), so `parseNoteText` was a guaranteed 400 and a
 * silent local fallback against either provider (review L3).
 */
export const CandidateFieldSchema = z.object({
  key: z.string().min(1).max(LIMITS.fieldKey),
  value: z.union([z.string().max(LIMITS.label), z.number(), z.null()]),
});

export const CandidateRecordSchema = z.object({
  type: z.enum(CANDIDATE_RECORD_TYPES),
  summary: z.string().max(LIMITS.question),
  fields: z.array(CandidateFieldSchema).max(LIMITS.noteFields),
  confidence: z.enum(PARSE_CONFIDENCE),
});

export const NoteStructureSchema = z.object({
  candidates: z.array(CandidateRecordSchema).max(LIMITS.listItems),
  unparsed: z.array(z.string().max(LIMITS.unparsedText)).max(LIMITS.listItems),
});
export type NoteStructure = z.infer<typeof NoteStructureSchema>;

export const NoteStructureJsonSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    candidates: {
      type: 'array',
      maxItems: LIMITS.listItems,
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          type: { type: 'string', enum: [...CANDIDATE_RECORD_TYPES] },
          summary: { type: 'string', maxLength: LIMITS.question },
          fields: {
            type: 'array',
            maxItems: LIMITS.noteFields,
            items: {
              type: 'object',
              additionalProperties: false,
              properties: {
                key: { type: 'string', minLength: 1, maxLength: LIMITS.fieldKey },
                value: {
                  anyOf: [
                    { type: 'string', maxLength: LIMITS.label },
                    { type: 'number' },
                    { type: 'null' },
                  ],
                },
              },
              required: ['key', 'value'],
            },
          },
          confidence: { type: 'string', enum: [...PARSE_CONFIDENCE] },
        },
        required: ['type', 'summary', 'fields', 'confidence'],
      },
    },
    unparsed: {
      type: 'array',
      maxItems: LIMITS.listItems,
      items: { type: 'string', maxLength: LIMITS.unparsedText },
    },
  },
  required: ['candidates', 'unparsed'],
} as const;
