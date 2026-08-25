-- AlterTable
ALTER TABLE "public"."ExerciseSession" ADD COLUMN     "externalId" TEXT,
ADD COLUMN     "importBatchId" TEXT,
ADD COLUMN     "rawPayload" JSONB;

-- AlterTable
ALTER TABLE "public"."MedicationEvent" ADD COLUMN     "externalId" TEXT,
ADD COLUMN     "importBatchId" TEXT,
ADD COLUMN     "rawPayload" JSONB;

-- AlterTable
ALTER TABLE "public"."NoteEntry" ADD COLUMN     "externalId" TEXT,
ADD COLUMN     "importBatchId" TEXT,
ADD COLUMN     "rawPayload" JSONB;

-- AlterTable
ALTER TABLE "public"."SleepSession" ADD COLUMN     "externalId" TEXT,
ADD COLUMN     "importBatchId" TEXT,
ADD COLUMN     "rawPayload" JSONB;

-- AlterTable
ALTER TABLE "public"."WeightMeasurement" ADD COLUMN     "externalId" TEXT,
ADD COLUMN     "importBatchId" TEXT,
ADD COLUMN     "rawPayload" JSONB;

-- AddForeignKey
ALTER TABLE "public"."ExerciseSession" ADD CONSTRAINT "ExerciseSession_importBatchId_fkey" FOREIGN KEY ("importBatchId") REFERENCES "public"."ImportBatch"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."SleepSession" ADD CONSTRAINT "SleepSession_importBatchId_fkey" FOREIGN KEY ("importBatchId") REFERENCES "public"."ImportBatch"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."MedicationEvent" ADD CONSTRAINT "MedicationEvent_importBatchId_fkey" FOREIGN KEY ("importBatchId") REFERENCES "public"."ImportBatch"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."WeightMeasurement" ADD CONSTRAINT "WeightMeasurement_importBatchId_fkey" FOREIGN KEY ("importBatchId") REFERENCES "public"."ImportBatch"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."NoteEntry" ADD CONSTRAINT "NoteEntry_importBatchId_fkey" FOREIGN KEY ("importBatchId") REFERENCES "public"."ImportBatch"("id") ON DELETE SET NULL ON UPDATE CASCADE;
