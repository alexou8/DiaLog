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

export const forgotPasswordSchema = z.object({ email: emailSchema });
export const passwordResetTokenSchema = z.string().regex(/^[A-Za-z0-9_-]{43}$/);
export const resetPasswordSchema = z.object({
  proof: z.string().min(1).max(2048),
  newPassword: z.string(),
  confirmPassword: z.string(),
});

const LOCAL_DATE_TIME = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?$/;

/**
 * True when every component of a `YYYY-MM-DDTHH:mm[:ss]` value names a real
 * calendar instant. The regex alone accepted `2025-13-45T99:99`; that became an
 * Invalid Date and `Intl.DateTimeFormat#formatToParts` then threw a RangeError
 * deep in lib/domain/time.ts, which surfaced as a 500 instead of a field error.
 * Built in UTC and compared back component by component, because `Date`
 * silently rolls overflow forward (Feb 30 becomes Mar 2) rather than failing.
 */
export function isRealLocalDateTime(value: string): boolean {
  const m = LOCAL_DATE_TIME.exec(value);
  if (!m) return false;
  const [y, mo, d, h, mi, s] = m.slice(1).map((part) => Number(part ?? 0)) as [
    number,
    number,
    number,
    number,
    number,
    number,
  ];
  const date = new Date(Date.UTC(y, mo - 1, d, h, mi, s));
  return (
    date.getUTCFullYear() === y &&
    date.getUTCMonth() === mo - 1 &&
    date.getUTCDate() === d &&
    date.getUTCHours() === h &&
    date.getUTCMinutes() === mi &&
    date.getUTCSeconds() === s
  );
}

/** A date-time entered as a local wall-clock value from a form control. */
export const localDateTimeSchema = z
  .string()
  .regex(LOCAL_DATE_TIME, 'Please enter a valid date and time.')
  .refine(isRealLocalDateTime, 'Please enter a valid date and time.');

let supportedTimeZones: ReadonlySet<string> | null = null;

/**
 * True for an IANA time zone this runtime can format in.
 *
 * Timezone and locale used to be any short string, stored as-is. A crafted
 * value such as `Foo/Bar` was then handed to `Intl.DateTimeFormat` on every
 * later page and import, which throws a RangeError — breaking that person's
 * record entry and imports until the value was fixed by hand.
 *
 * `Intl.supportedValuesOf('timeZone')` is the allowlist, plus `UTC`, which
 * Node's list omits. Not the allowlist alone, though: it holds one canonical
 * name per zone, and which name is canonical differs between ICU versions —
 * Node 24 lists `Asia/Calcutta` while current browsers report `Asia/Kolkata`
 * from `resolvedOptions().timeZone`, which is exactly what onboarding submits.
 * So a well-formed IANA-shaped name the runtime actually accepts is allowed
 * too; anything it accepts cannot cause the RangeError this guards against.
 */
export function isSupportedTimeZone(value: string): boolean {
  supportedTimeZones ??= new Set([...Intl.supportedValuesOf('timeZone'), 'UTC']);
  if (supportedTimeZones.has(value)) return true;
  if (!/^[A-Za-z][A-Za-z0-9_+-]*(?:\/[A-Za-z0-9_+-]+){0,2}$/.test(value)) return false;
  try {
    new Intl.DateTimeFormat('en-US', { timeZone: value });
    return true;
  } catch {
    return false;
  }
}

/** True for a structurally valid BCP 47 language tag (see isSupportedTimeZone). */
export function isValidLocale(value: string): boolean {
  try {
    return Intl.getCanonicalLocales(value).length === 1;
  } catch {
    return false;
  }
}

const timezoneSchema = z
  .string()
  .min(1)
  .max(64)
  .refine(isSupportedTimeZone, 'Please choose a time zone from the list.');

const localeSchema = z
  .string()
  .min(2)
  .max(10)
  .refine(isValidLocale, 'Please choose a language and region from the list.');

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
  timezone: timezoneSchema,
  goals: z.array(z.string().max(60)).max(6).default([]),
});

export const preferencesSchema = z.object({
  displayName: z.string().trim().max(80).optional().or(z.literal('')),
  glucoseUnit: z.enum(['MGDL', 'MMOLL']),
  locale: localeSchema,
  timezone: timezoneSchema,
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
