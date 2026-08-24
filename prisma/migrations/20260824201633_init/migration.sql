-- CreateEnum
CREATE TYPE "public"."GlucoseUnit" AS ENUM ('MGDL', 'MMOLL');

-- CreateEnum
CREATE TYPE "public"."ConditionType" AS ENUM ('PREDIABETES', 'TYPE_1', 'TYPE_2', 'GESTATIONAL', 'CURIOUS', 'PREFER_NOT_TO_SAY');

-- CreateEnum
CREATE TYPE "public"."DetailLevel" AS ENUM ('SIMPLE', 'STANDARD', 'DETAILED');

-- CreateEnum
CREATE TYPE "public"."DataSource" AS ENUM ('MANUAL', 'IMPORT', 'AI_ASSISTED', 'DEVICE', 'SEED');

-- CreateEnum
CREATE TYPE "public"."DeviceKind" AS ENUM ('GLUCOSE_METER', 'CGM', 'BLOOD_PRESSURE_MONITOR', 'SCALE', 'WEARABLE', 'PHONE_HEALTH_PLATFORM', 'OTHER');

-- CreateEnum
CREATE TYPE "public"."ImportStatus" AS ENUM ('PENDING', 'COMPLETED', 'FAILED');

-- CreateEnum
CREATE TYPE "public"."GlucoseContext" AS ENUM ('FASTING', 'BEFORE_MEAL', 'AFTER_MEAL', 'BEDTIME', 'RANDOM', 'UNKNOWN');

-- CreateEnum
CREATE TYPE "public"."MealType" AS ENUM ('BREAKFAST', 'LUNCH', 'DINNER', 'SNACK', 'OTHER');

-- CreateEnum
CREATE TYPE "public"."EstimateSource" AS ENUM ('USER_ENTERED', 'AI_ESTIMATE', 'IMPORTED');

-- CreateEnum
CREATE TYPE "public"."Intensity" AS ENUM ('LIGHT', 'MODERATE', 'VIGOROUS');

-- CreateEnum
CREATE TYPE "public"."EvidenceLevel" AS ENUM ('INSUFFICIENT', 'EARLY', 'EMERGING', 'CONSISTENT');

-- CreateEnum
CREATE TYPE "public"."InsightSource" AS ENUM ('STATISTICAL', 'ML', 'REFERENCE');

-- CreateTable
CREATE TABLE "public"."User" (
    "id" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "passwordHash" TEXT NOT NULL,
    "tokenVersion" INTEGER NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "lastLoginAt" TIMESTAMP(3),

    CONSTRAINT "User_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."PasswordResetToken" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "tokenHash" TEXT NOT NULL,
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "usedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "PasswordResetToken_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."Profile" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "displayName" TEXT,
    "glucoseUnit" "public"."GlucoseUnit" NOT NULL DEFAULT 'MMOLL',
    "locale" TEXT NOT NULL DEFAULT 'en-CA',
    "timezone" TEXT NOT NULL DEFAULT 'America/Toronto',
    "condition" "public"."ConditionType" NOT NULL DEFAULT 'PREFER_NOT_TO_SAY',
    "targetLowMgdl" DOUBLE PRECISION NOT NULL DEFAULT 70,
    "targetHighMgdl" DOUBLE PRECISION NOT NULL DEFAULT 180,
    "goals" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "detailLevel" "public"."DetailLevel" NOT NULL DEFAULT 'STANDARD',
    "largeText" BOOLEAN NOT NULL DEFAULT false,
    "reduceMotion" BOOLEAN NOT NULL DEFAULT false,
    "aiEnabled" BOOLEAN NOT NULL DEFAULT true,
    "externalAiConsentAt" TIMESTAMP(3),
    "onboardingCompletedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Profile_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."Device" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "label" TEXT NOT NULL,
    "vendor" TEXT,
    "model" TEXT,
    "kind" "public"."DeviceKind" NOT NULL DEFAULT 'GLUCOSE_METER',
    "connectorId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Device_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."ImportBatch" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "connectorId" TEXT NOT NULL,
    "connectorName" TEXT NOT NULL,
    "filename" TEXT NOT NULL,
    "mimeType" TEXT,
    "byteSize" INTEGER NOT NULL DEFAULT 0,
    "status" "public"."ImportStatus" NOT NULL DEFAULT 'PENDING',
    "rowsTotal" INTEGER NOT NULL DEFAULT 0,
    "rowsImported" INTEGER NOT NULL DEFAULT 0,
    "rowsDuplicate" INTEGER NOT NULL DEFAULT 0,
    "rowsRejected" INTEGER NOT NULL DEFAULT 0,
    "errorMessage" TEXT,
    "deviceId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "finishedAt" TIMESTAMP(3),

    CONSTRAINT "ImportBatch_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."ImportIssue" (
    "id" TEXT NOT NULL,
    "batchId" TEXT NOT NULL,
    "rowNumber" INTEGER NOT NULL,
    "code" TEXT NOT NULL,
    "message" TEXT NOT NULL,
    "rawRow" TEXT,

    CONSTRAINT "ImportIssue_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."GlucoseReading" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "takenAt" TIMESTAMP(3) NOT NULL,
    "valueMgdl" DOUBLE PRECISION NOT NULL,
    "context" "public"."GlucoseContext" NOT NULL DEFAULT 'UNKNOWN',
    "note" TEXT,
    "source" "public"."DataSource" NOT NULL DEFAULT 'MANUAL',
    "deviceId" TEXT,
    "importBatchId" TEXT,
    "externalId" TEXT,
    "rawPayload" JSONB,
    "dedupeKey" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "GlucoseReading_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."Meal" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "takenAt" TIMESTAMP(3) NOT NULL,
    "mealType" "public"."MealType" NOT NULL DEFAULT 'OTHER',
    "description" TEXT NOT NULL,
    "carbsG" DOUBLE PRECISION,
    "proteinG" DOUBLE PRECISION,
    "fatG" DOUBLE PRECISION,
    "fiberG" DOUBLE PRECISION,
    "calories" DOUBLE PRECISION,
    "portion" TEXT,
    "note" TEXT,
    "estimateSource" "public"."EstimateSource" NOT NULL DEFAULT 'USER_ENTERED',
    "source" "public"."DataSource" NOT NULL DEFAULT 'MANUAL',
    "importBatchId" TEXT,
    "rawPayload" JSONB,
    "dedupeKey" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Meal_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."FoodItem" (
    "id" TEXT NOT NULL,
    "mealId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "quantity" TEXT,
    "carbsG" DOUBLE PRECISION,
    "proteinG" DOUBLE PRECISION,
    "fatG" DOUBLE PRECISION,
    "fiberG" DOUBLE PRECISION,
    "calories" DOUBLE PRECISION,

    CONSTRAINT "FoodItem_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."ExerciseSession" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "takenAt" TIMESTAMP(3) NOT NULL,
    "endedAt" TIMESTAMP(3),
    "activity" TEXT NOT NULL,
    "durationMin" INTEGER NOT NULL,
    "intensity" "public"."Intensity" NOT NULL DEFAULT 'MODERATE',
    "distanceKm" DOUBLE PRECISION,
    "steps" INTEGER,
    "note" TEXT,
    "source" "public"."DataSource" NOT NULL DEFAULT 'MANUAL',
    "dedupeKey" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ExerciseSession_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."SleepSession" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "takenAt" TIMESTAMP(3) NOT NULL,
    "endedAt" TIMESTAMP(3) NOT NULL,
    "durationMin" INTEGER NOT NULL,
    "quality" INTEGER,
    "note" TEXT,
    "source" "public"."DataSource" NOT NULL DEFAULT 'MANUAL',
    "dedupeKey" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "SleepSession_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."MedicationEvent" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "takenAt" TIMESTAMP(3) NOT NULL,
    "name" TEXT NOT NULL,
    "dose" TEXT,
    "route" TEXT,
    "note" TEXT,
    "source" "public"."DataSource" NOT NULL DEFAULT 'MANUAL',
    "dedupeKey" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "MedicationEvent_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."WeightMeasurement" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "takenAt" TIMESTAMP(3) NOT NULL,
    "weightKg" DOUBLE PRECISION NOT NULL,
    "note" TEXT,
    "source" "public"."DataSource" NOT NULL DEFAULT 'MANUAL',
    "dedupeKey" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "WeightMeasurement_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."BloodPressureMeasurement" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "takenAt" TIMESTAMP(3) NOT NULL,
    "systolic" INTEGER NOT NULL,
    "diastolic" INTEGER NOT NULL,
    "pulse" INTEGER,
    "note" TEXT,
    "source" "public"."DataSource" NOT NULL DEFAULT 'MANUAL',
    "dedupeKey" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "BloodPressureMeasurement_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."HydrationEvent" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "takenAt" TIMESTAMP(3) NOT NULL,
    "volumeMl" INTEGER NOT NULL,
    "source" "public"."DataSource" NOT NULL DEFAULT 'MANUAL',
    "dedupeKey" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "HydrationEvent_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."SymptomEntry" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "takenAt" TIMESTAMP(3) NOT NULL,
    "symptom" TEXT NOT NULL,
    "severity" INTEGER,
    "note" TEXT,
    "source" "public"."DataSource" NOT NULL DEFAULT 'MANUAL',
    "dedupeKey" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "SymptomEntry_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."MoodEntry" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "takenAt" TIMESTAMP(3) NOT NULL,
    "mood" INTEGER NOT NULL,
    "stress" INTEGER,
    "note" TEXT,
    "source" "public"."DataSource" NOT NULL DEFAULT 'MANUAL',
    "dedupeKey" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "MoodEntry_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."NoteEntry" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "takenAt" TIMESTAMP(3) NOT NULL,
    "text" TEXT NOT NULL,
    "source" "public"."DataSource" NOT NULL DEFAULT 'MANUAL',
    "dedupeKey" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "NoteEntry_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."Insight" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "kind" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "summary" TEXT NOT NULL,
    "detail" TEXT,
    "evidenceLevel" "public"."EvidenceLevel" NOT NULL,
    "sampleSize" INTEGER NOT NULL DEFAULT 0,
    "source" "public"."InsightSource" NOT NULL DEFAULT 'STATISTICAL',
    "evidence" JSONB NOT NULL,
    "periodStart" TIMESTAMP(3) NOT NULL,
    "periodEnd" TIMESTAMP(3) NOT NULL,
    "generatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Insight_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."AIConversation" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "title" TEXT NOT NULL DEFAULT 'New conversation',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "AIConversation_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."AIMessage" (
    "id" TEXT NOT NULL,
    "conversationId" TEXT NOT NULL,
    "role" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "evidence" JSONB,
    "providerId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "AIMessage_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."AuditEvent" (
    "id" TEXT NOT NULL,
    "userId" TEXT,
    "action" TEXT NOT NULL,
    "entity" TEXT,
    "entityId" TEXT,
    "detail" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "AuditEvent_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "User_email_key" ON "public"."User"("email");

-- CreateIndex
CREATE UNIQUE INDEX "PasswordResetToken_tokenHash_key" ON "public"."PasswordResetToken"("tokenHash");

-- CreateIndex
CREATE INDEX "PasswordResetToken_userId_idx" ON "public"."PasswordResetToken"("userId");

-- CreateIndex
CREATE UNIQUE INDEX "Profile_userId_key" ON "public"."Profile"("userId");

-- CreateIndex
CREATE INDEX "Device_userId_idx" ON "public"."Device"("userId");

-- CreateIndex
CREATE INDEX "ImportBatch_userId_createdAt_idx" ON "public"."ImportBatch"("userId", "createdAt");

-- CreateIndex
CREATE INDEX "ImportIssue_batchId_idx" ON "public"."ImportIssue"("batchId");

-- CreateIndex
CREATE INDEX "GlucoseReading_userId_takenAt_idx" ON "public"."GlucoseReading"("userId", "takenAt");

-- CreateIndex
CREATE UNIQUE INDEX "GlucoseReading_userId_dedupeKey_key" ON "public"."GlucoseReading"("userId", "dedupeKey");

-- CreateIndex
CREATE INDEX "Meal_userId_takenAt_idx" ON "public"."Meal"("userId", "takenAt");

-- CreateIndex
CREATE UNIQUE INDEX "Meal_userId_dedupeKey_key" ON "public"."Meal"("userId", "dedupeKey");

-- CreateIndex
CREATE INDEX "FoodItem_mealId_idx" ON "public"."FoodItem"("mealId");

-- CreateIndex
CREATE INDEX "ExerciseSession_userId_takenAt_idx" ON "public"."ExerciseSession"("userId", "takenAt");

-- CreateIndex
CREATE UNIQUE INDEX "ExerciseSession_userId_dedupeKey_key" ON "public"."ExerciseSession"("userId", "dedupeKey");

-- CreateIndex
CREATE INDEX "SleepSession_userId_takenAt_idx" ON "public"."SleepSession"("userId", "takenAt");

-- CreateIndex
CREATE UNIQUE INDEX "SleepSession_userId_dedupeKey_key" ON "public"."SleepSession"("userId", "dedupeKey");

-- CreateIndex
CREATE INDEX "MedicationEvent_userId_takenAt_idx" ON "public"."MedicationEvent"("userId", "takenAt");

-- CreateIndex
CREATE UNIQUE INDEX "MedicationEvent_userId_dedupeKey_key" ON "public"."MedicationEvent"("userId", "dedupeKey");

-- CreateIndex
CREATE INDEX "WeightMeasurement_userId_takenAt_idx" ON "public"."WeightMeasurement"("userId", "takenAt");

-- CreateIndex
CREATE UNIQUE INDEX "WeightMeasurement_userId_dedupeKey_key" ON "public"."WeightMeasurement"("userId", "dedupeKey");

-- CreateIndex
CREATE INDEX "BloodPressureMeasurement_userId_takenAt_idx" ON "public"."BloodPressureMeasurement"("userId", "takenAt");

-- CreateIndex
CREATE UNIQUE INDEX "BloodPressureMeasurement_userId_dedupeKey_key" ON "public"."BloodPressureMeasurement"("userId", "dedupeKey");

-- CreateIndex
CREATE INDEX "HydrationEvent_userId_takenAt_idx" ON "public"."HydrationEvent"("userId", "takenAt");

-- CreateIndex
CREATE UNIQUE INDEX "HydrationEvent_userId_dedupeKey_key" ON "public"."HydrationEvent"("userId", "dedupeKey");

-- CreateIndex
CREATE INDEX "SymptomEntry_userId_takenAt_idx" ON "public"."SymptomEntry"("userId", "takenAt");

-- CreateIndex
CREATE UNIQUE INDEX "SymptomEntry_userId_dedupeKey_key" ON "public"."SymptomEntry"("userId", "dedupeKey");

-- CreateIndex
CREATE INDEX "MoodEntry_userId_takenAt_idx" ON "public"."MoodEntry"("userId", "takenAt");

-- CreateIndex
CREATE UNIQUE INDEX "MoodEntry_userId_dedupeKey_key" ON "public"."MoodEntry"("userId", "dedupeKey");

-- CreateIndex
CREATE INDEX "NoteEntry_userId_takenAt_idx" ON "public"."NoteEntry"("userId", "takenAt");

-- CreateIndex
CREATE UNIQUE INDEX "NoteEntry_userId_dedupeKey_key" ON "public"."NoteEntry"("userId", "dedupeKey");

-- CreateIndex
CREATE INDEX "Insight_userId_generatedAt_idx" ON "public"."Insight"("userId", "generatedAt");

-- CreateIndex
CREATE INDEX "AIConversation_userId_updatedAt_idx" ON "public"."AIConversation"("userId", "updatedAt");

-- CreateIndex
CREATE INDEX "AIMessage_conversationId_createdAt_idx" ON "public"."AIMessage"("conversationId", "createdAt");

-- CreateIndex
CREATE INDEX "AuditEvent_userId_createdAt_idx" ON "public"."AuditEvent"("userId", "createdAt");

-- AddForeignKey
ALTER TABLE "public"."PasswordResetToken" ADD CONSTRAINT "PasswordResetToken_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."Profile" ADD CONSTRAINT "Profile_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."Device" ADD CONSTRAINT "Device_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."ImportBatch" ADD CONSTRAINT "ImportBatch_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."ImportBatch" ADD CONSTRAINT "ImportBatch_deviceId_fkey" FOREIGN KEY ("deviceId") REFERENCES "public"."Device"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."ImportIssue" ADD CONSTRAINT "ImportIssue_batchId_fkey" FOREIGN KEY ("batchId") REFERENCES "public"."ImportBatch"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."GlucoseReading" ADD CONSTRAINT "GlucoseReading_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."GlucoseReading" ADD CONSTRAINT "GlucoseReading_deviceId_fkey" FOREIGN KEY ("deviceId") REFERENCES "public"."Device"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."GlucoseReading" ADD CONSTRAINT "GlucoseReading_importBatchId_fkey" FOREIGN KEY ("importBatchId") REFERENCES "public"."ImportBatch"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."Meal" ADD CONSTRAINT "Meal_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."Meal" ADD CONSTRAINT "Meal_importBatchId_fkey" FOREIGN KEY ("importBatchId") REFERENCES "public"."ImportBatch"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."FoodItem" ADD CONSTRAINT "FoodItem_mealId_fkey" FOREIGN KEY ("mealId") REFERENCES "public"."Meal"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."ExerciseSession" ADD CONSTRAINT "ExerciseSession_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."SleepSession" ADD CONSTRAINT "SleepSession_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."MedicationEvent" ADD CONSTRAINT "MedicationEvent_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."WeightMeasurement" ADD CONSTRAINT "WeightMeasurement_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."BloodPressureMeasurement" ADD CONSTRAINT "BloodPressureMeasurement_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."HydrationEvent" ADD CONSTRAINT "HydrationEvent_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."SymptomEntry" ADD CONSTRAINT "SymptomEntry_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."MoodEntry" ADD CONSTRAINT "MoodEntry_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."NoteEntry" ADD CONSTRAINT "NoteEntry_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."Insight" ADD CONSTRAINT "Insight_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."AIConversation" ADD CONSTRAINT "AIConversation_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."AIMessage" ADD CONSTRAINT "AIMessage_conversationId_fkey" FOREIGN KEY ("conversationId") REFERENCES "public"."AIConversation"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."AuditEvent" ADD CONSTRAINT "AuditEvent_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
