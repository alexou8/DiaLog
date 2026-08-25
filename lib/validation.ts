/**
 * Request/form validation schemas.
 *
 * Everything entering the application from outside — forms, API bodies, query
 * strings — is parsed here before it reaches domain logic. Error messages are
 * written for the person reading them, not for the developer.
 */
import { z } from 'zod';

export const emailSchema = z
  .string()
  .trim()
  .toLowerCase()
  .min(3, 'Please enter your email address.')
  .max(254, 'That email address is too long.')
  .email('That does not look like an email address. Please check it and try again.');

export const signUpSchema = z.object({
  email: emailSchema,
  password: z.string().min(1, 'Please choose a password.'),
  displayName: z.string().trim().max(80).optional().or(z.literal('')),
});

export const signInSchema = z.object({
  email: emailSchema,
  password: z.string().min(1, 'Please enter your password.'),
});

/** A date-time entered as a local wall-clock value from a form control. */
export const localDateTimeSchema = z
  .string()
  .regex(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$/, 'Please enter a valid date and time.');

export const glucoseEntrySchema = z.object({
  value: z.coerce.number().finite('Please enter a number.'),
  unit: z.enum(['MGDL', 'MMOLL']),
  takenAt: localDateTimeSchema,
  context: z.enum(['FASTING', 'BEFORE_MEAL', 'AFTER_MEAL', 'BEDTIME', 'RANDOM', 'UNKNOWN']),
  note: z
    .string()
    .trim()
    .max(500, 'Please keep notes under 500 characters.')
    .optional()
    .or(z.literal('')),
});

const optionalNumber = z
  .union([z.coerce.number().nonnegative('Please enter zero or more.'), z.literal('')])
  .optional()
  .transform((v) => (v === '' || v === undefined ? null : Number(v)));

export const mealEntrySchema = z.object({
  takenAt: localDateTimeSchema,
  mealType: z.enum(['BREAKFAST', 'LUNCH', 'DINNER', 'SNACK', 'OTHER']),
  description: z.string().trim().min(1, 'Please describe what you ate.').max(300),
  carbsG: optionalNumber,
  proteinG: optionalNumber,
  fatG: optionalNumber,
  fiberG: optionalNumber,
  calories: optionalNumber,
  portion: z.string().trim().max(80).optional().or(z.literal('')),
  note: z.string().trim().max(500).optional().or(z.literal('')),
  estimateSource: z.enum(['USER_ENTERED', 'AI_ESTIMATE', 'IMPORTED']).default('USER_ENTERED'),
});

export const exerciseEntrySchema = z.object({
  takenAt: localDateTimeSchema,
  activity: z.string().trim().min(1, 'Please say what you did.').max(80),
  durationMin: z.coerce
    .number()
    .int('Please enter whole minutes.')
    .min(1, 'Please enter at least one minute.')
    .max(1440, 'Please enter 1440 minutes or fewer.'),
  intensity: z.enum(['LIGHT', 'MODERATE', 'VIGOROUS']),
  distanceKm: optionalNumber,
  steps: optionalNumber,
  note: z.string().trim().max(500).optional().or(z.literal('')),
});

export const sleepEntrySchema = z.object({
  takenAt: localDateTimeSchema,
  endedAt: localDateTimeSchema,
  quality: z.coerce.number().int().min(1).max(5).optional().nullable(),
  note: z.string().trim().max(500).optional().or(z.literal('')),
});

export const medicationEntrySchema = z.object({
  takenAt: localDateTimeSchema,
  name: z.string().trim().min(1, 'Please enter the name as it appears on the package.').max(120),
  dose: z.string().trim().max(60).optional().or(z.literal('')),
  route: z.string().trim().max(40).optional().or(z.literal('')),
  note: z.string().trim().max(500).optional().or(z.literal('')),
});

export const weightEntrySchema = z.object({
  takenAt: localDateTimeSchema,
  weight: z.coerce.number().positive('Please enter a weight greater than zero.'),
  unit: z.enum(['KG', 'LB']),
  note: z.string().trim().max(500).optional().or(z.literal('')),
});

export const bloodPressureEntrySchema = z.object({
  takenAt: localDateTimeSchema,
  systolic: z.coerce
    .number()
    .int()
    .min(50, 'Please check that number.')
    .max(300, 'Please check that number.'),
  diastolic: z.coerce
    .number()
    .int()
    .min(30, 'Please check that number.')
    .max(200, 'Please check that number.'),
  pulse: z
    .union([z.coerce.number().int().min(20).max(250), z.literal('')])
    .optional()
    .transform((v) => (v === '' || v === undefined ? null : Number(v))),
  note: z.string().trim().max(500).optional().or(z.literal('')),
});

export const moodEntrySchema = z.object({
  takenAt: localDateTimeSchema,
  mood: z.coerce.number().int().min(1).max(5),
  stress: z.coerce.number().int().min(1).max(5).optional().nullable(),
  note: z.string().trim().max(500).optional().or(z.literal('')),
});

export const onboardingSchema = z.object({
  displayName: z.string().trim().max(80).optional().or(z.literal('')),
  condition: z.enum([
    'PREDIABETES',
    'TYPE_1',
    'TYPE_2',
    'GESTATIONAL',
    'CURIOUS',
    'PREFER_NOT_TO_SAY',
  ]),
  glucoseUnit: z.enum(['MGDL', 'MMOLL']),
  timezone: z.string().min(1).max(64),
  goals: z.array(z.string().max(60)).max(6).default([]),
});

export const preferencesSchema = z.object({
  displayName: z.string().trim().max(80).optional().or(z.literal('')),
  glucoseUnit: z.enum(['MGDL', 'MMOLL']),
  locale: z.string().min(2).max(10),
  timezone: z.string().min(1).max(64),
  targetLow: z.coerce.number().positive(),
  targetHigh: z.coerce.number().positive(),
  detailLevel: z.enum(['SIMPLE', 'STANDARD', 'DETAILED']),
  largeText: z.coerce.boolean().optional().default(false),
  reduceMotion: z.coerce.boolean().optional().default(false),
  aiEnabled: z.coerce.boolean().optional().default(false),
  externalAiConsent: z.coerce.boolean().optional().default(false),
});

export const assistantQuestionSchema = z.object({
  question: z
    .string()
    .trim()
    .min(3, 'Please type a question.')
    .max(500, 'Please keep questions under 500 characters.'),
  conversationId: z.string().cuid().optional(),
  detailLevel: z.enum(['simple', 'standard', 'detailed']).default('standard'),
});

export const quickLogSchema = z.object({
  text: z.string().trim().min(3, 'Please describe what you would like to log.').max(600),
});

/** Collapse a ZodError into `{ field: message }` for form rendering. */
export function fieldErrors(error: z.ZodError): Record<string, string> {
  const out: Record<string, string> = {};
  for (const issue of error.issues) {
    const key = issue.path.join('.') || 'form';
    out[key] ??= issue.message;
  }
  return out;
}

export const hydrationEntrySchema = z.object({
  takenAt: localDateTimeSchema,
  volume: z.coerce.number().positive('Please enter an amount greater than zero.'),
  unit: z.enum(['ML', 'CUP', 'FL_OZ']),
});

export const symptomEntrySchema = z.object({
  takenAt: localDateTimeSchema,
  symptom: z.string().trim().min(1, 'Please say what you noticed.').max(120),
  severity: z.coerce.number().int().min(1).max(5).optional().nullable(),
  note: z.string().trim().max(500).optional().or(z.literal('')),
});
