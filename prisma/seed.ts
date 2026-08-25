/**
 * Development seed.
 *
 * Generates a demo account with a realistic-looking three months of records:
 * fasting/post-meal readings that respond to carbohydrate load and to evening
 * walks, occasional missed days, a couple of genuinely unusual readings, and a
 * gentle downward drift in the last few weeks. This exists so the dashboard,
 * analytics and reports can be exercised end to end — it is SYNTHETIC data and
 * carries no clinical meaning whatsoever.
 */
import { PrismaClient, type GlucoseContext, type MealType, type Prisma } from '@prisma/client';
import bcrypt from 'bcryptjs';
import { createHash } from 'node:crypto';

const prisma = new PrismaClient();

const DEMO_EMAIL = 'demo@dialog.health';
const DEMO_PASSWORD = 'demo-account-2026';
const DAYS = 90;
const TIMEZONE = 'America/Toronto';

/** Deterministic PRNG so repeated seeds produce the same demo history. */
function mulberry32(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const random = mulberry32(20260824);

/** Box–Muller, for readings that scatter like measurements rather than uniformly. */
function gaussian(mean: number, sd: number): number {
  const u = Math.max(random(), 1e-9);
  const v = Math.max(random(), 1e-9);
  return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function key(type: string, at: Date, value: number | null | undefined, extra = ''): string {
  const minute = Math.floor(at.getTime() / 60_000);
  const rounded = value == null ? '' : Math.round(value * 100) / 100;
  return createHash('sha256')
    .update([type, '', minute, rounded, extra.toLowerCase()].join('|'))
    .digest('hex')
    .slice(0, 32);
}

function at(dayOffset: number, hour: number, minute = 0): Date {
  const base = new Date();
  base.setUTCHours(0, 0, 0, 0);
  const date = new Date(base.getTime() - dayOffset * 86_400_000);
  // The demo timezone is UTC-4/-5; adding the offset keeps meals at meal times.
  date.setUTCHours(hour + 4, minute, 0, 0);
  return date;
}

async function main() {
  await prisma.user.deleteMany({ where: { email: DEMO_EMAIL } });

  const user = await prisma.user.create({
    data: {
      email: DEMO_EMAIL,
      passwordHash: await bcrypt.hash(DEMO_PASSWORD, 10),
      profile: {
        create: {
          displayName: 'Marie',
          glucoseUnit: 'MMOLL',
          locale: 'en-CA',
          timezone: TIMEZONE,
          condition: 'TYPE_2',
          targetLowMgdl: 72,
          targetHighMgdl: 180,
          goals: ['understand', 'food', 'movement'],
          onboardingCompletedAt: new Date(),
        },
      },
    },
  });

  const glucose: Prisma.GlucoseReadingCreateManyInput[] = [];
  const meals: Prisma.MealCreateManyInput[] = [];
  const exercise: Prisma.ExerciseSessionCreateManyInput[] = [];
  const sleep: Prisma.SleepSessionCreateManyInput[] = [];
  const medications: Prisma.MedicationEventCreateManyInput[] = [];

  const DINNERS: { description: string; carbs: number }[] = [
    { description: 'Salmon, rice and green beans', carbs: 52 },
    { description: 'Chicken stir-fry with noodles', carbs: 68 },
    { description: 'Lentil soup and bread', carbs: 45 },
    { description: 'Pasta with tomato sauce', carbs: 82 },
    { description: 'Roast chicken and potatoes', carbs: 40 },
    { description: 'Beef chili and cornbread', carbs: 61 },
  ];

  for (let day = DAYS; day >= 0; day--) {
    // A few skipped days, the way real logging actually goes.
    if (random() < 0.12) continue;

    // A slow improvement over the most recent five weeks.
    const drift = day < 35 ? -(35 - day) * 0.25 : 0;
    const walkedAfterDinner = random() < (day < 35 ? 0.62 : 0.28);
    const sleptWell = random() < 0.7;

    // --- Morning: fasting reading
    const fasting = gaussian(126 + drift + (sleptWell ? 0 : 6), 11);
    glucose.push({
      userId: user.id,
      takenAt: at(day, 7, 15),
      valueMgdl: Math.round(fasting),
      context: 'FASTING' as GlucoseContext,
      source: 'SEED',
      dedupeKey: key('glucose', at(day, 7, 15), Math.round(fasting)),
    });

    medications.push({
      userId: user.id,
      takenAt: at(day, 7, 30),
      name: 'Metformin',
      dose: '500 mg',
      source: 'SEED',
      dedupeKey: key('medication', at(day, 7, 30), null, 'metformin'),
    });

    // --- Breakfast and its response
    const breakfastCarbs = Math.round(gaussian(38, 9));
    meals.push({
      userId: user.id,
      takenAt: at(day, 8, 0),
      mealType: 'BREAKFAST' as MealType,
      description: random() < 0.5 ? 'Oatmeal with berries' : 'Toast, eggs and coffee',
      carbsG: breakfastCarbs,
      proteinG: Math.round(gaussian(14, 4)),
      fatG: Math.round(gaussian(11, 4)),
      source: 'SEED',
      dedupeKey: key('meal', at(day, 8, 0), null, 'breakfast'),
    });
    const postBreakfast = fasting + breakfastCarbs * 0.9 + gaussian(0, 14);
    glucose.push({
      userId: user.id,
      takenAt: at(day, 10, 0),
      valueMgdl: Math.round(postBreakfast),
      context: 'AFTER_MEAL' as GlucoseContext,
      source: 'SEED',
      dedupeKey: key('glucose', at(day, 10, 0), Math.round(postBreakfast)),
    });

    // --- Dinner and its response, which is where the walk shows up
    const dinner = DINNERS[Math.floor(random() * DINNERS.length)]!;
    meals.push({
      userId: user.id,
      takenAt: at(day, 18, 30),
      mealType: 'DINNER' as MealType,
      description: dinner.description,
      carbsG: dinner.carbs,
      proteinG: Math.round(gaussian(28, 6)),
      fatG: Math.round(gaussian(18, 5)),
      fiberG: Math.round(gaussian(6, 2)),
      source: 'SEED',
      dedupeKey: key('meal', at(day, 18, 30), null, dinner.description),
    });

    if (walkedAfterDinner) {
      const minutes = Math.round(gaussian(24, 6));
      exercise.push({
        userId: user.id,
        takenAt: at(day, 19, 15),
        endedAt: at(day, 19, 15 + minutes),
        activity: 'Walking',
        durationMin: Math.max(10, minutes),
        intensity: 'LIGHT',
        source: 'SEED',
        dedupeKey: key('exercise', at(day, 19, 15), minutes, 'walking'),
      });
    }

    const postDinner =
      fasting + dinner.carbs * 0.95 + (walkedAfterDinner ? -21 : 6) + gaussian(0, 15) + drift;
    glucose.push({
      userId: user.id,
      takenAt: at(day, 20, 30),
      valueMgdl: Math.round(postDinner),
      context: 'AFTER_MEAL' as GlucoseContext,
      source: 'SEED',
      dedupeKey: key('glucose', at(day, 20, 30), Math.round(postDinner)),
    });

    // --- Bedtime reading, most nights
    if (random() < 0.55) {
      const bedtime = gaussian(138 + drift, 13);
      glucose.push({
        userId: user.id,
        takenAt: at(day, 22, 30),
        valueMgdl: Math.round(bedtime),
        context: 'BEDTIME' as GlucoseContext,
        source: 'SEED',
        dedupeKey: key('glucose', at(day, 22, 30), Math.round(bedtime)),
      });
    }

    const sleepMinutes = Math.round(sleptWell ? gaussian(455, 30) : gaussian(340, 35));
    sleep.push({
      userId: user.id,
      takenAt: at(day + 1, 23, 15),
      endedAt: at(day, 6, 45),
      durationMin: sleepMinutes,
      quality: sleptWell ? 4 : 2,
      source: 'SEED',
      dedupeKey: key('sleep', at(day + 1, 23, 15), sleepMinutes),
    });
  }

  // Two readings that are genuinely unusual for this person, so the anomaly
  // detector has something real to find.
  for (const [day, value] of [
    [9, 268],
    [23, 61],
  ] as const) {
    glucose.push({
      userId: user.id,
      takenAt: at(day, 15, 40),
      valueMgdl: value,
      context: 'RANDOM' as GlucoseContext,
      note: value > 200 ? 'Felt off, big lunch out' : 'Skipped lunch',
      source: 'SEED',
      dedupeKey: key('glucose', at(day, 15, 40), value),
    });
  }

  await prisma.glucoseReading.createMany({ data: glucose, skipDuplicates: true });
  await prisma.meal.createMany({ data: meals, skipDuplicates: true });
  await prisma.exerciseSession.createMany({ data: exercise, skipDuplicates: true });
  await prisma.sleepSession.createMany({ data: sleep, skipDuplicates: true });
  await prisma.medicationEvent.createMany({ data: medications, skipDuplicates: true });

  console.log(
    `Seeded ${DEMO_EMAIL} (password: ${DEMO_PASSWORD}) with ${glucose.length} readings, ${meals.length} meals, ${exercise.length} activity sessions, ${sleep.length} nights of sleep.`,
  );
  console.log('This is synthetic demonstration data. It has no clinical meaning.');
}

main()
  .catch((error) => {
    console.error(error);
    process.exit(1);
  })
  .finally(() => prisma.$disconnect());
