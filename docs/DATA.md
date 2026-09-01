# Data model

Canonical documentation of DiaLog's data architecture. The source of truth is
`prisma/schema.prisma` and the migrations under `prisma/migrations/`; this
document explains what the schema means and why it is shaped the way it is. If
the two ever disagree, the schema wins and this file is wrong.

Verified against the schema and every Prisma call site in `app/`, `lib/` and
`prisma/` as of the September 2026 audit.

## Principles

**Canonical storage units.** Glucose is stored in mg/dL, mass in kg, volume in
mL, distance in km, duration in minutes — always, regardless of what the user
entered or what they see. The display unit is a per-user preference on
`Profile`. All conversion goes through `lib/domain/units.ts`; nothing converts
inline. A reading is therefore comparable across a user's whole history even if
they switched units halfway through.

**Ownership is part of the query, not a later check.** Every health record
carries a `userId`, and every read or write pairs the record id with its owner
— `deleteMany({ where: { id, userId } })`, never `delete({ where: { id } })`.
A guessed or leaked id from another account simply does not match. This rule
lives in `lib/db/health-records.ts`, and it applies to any table reachable with
a client-supplied id, including `AIConversation` (see the audit note below).

**Content-addressed dedupe.** Every importable record has a `dedupeKey` that is
derived from the record's own content, with `@@unique([userId, dedupeKey])`.
Re-importing the same export is a no-op rather than a duplicate. The uniqueness
constraint is what enforces that, so nothing may bypass it.

**Provenance is retained.** Imported records keep `source`, `externalId`,
`rawPayload` and a nullable `importBatchId`, so any row can be traced back to
the file it came from, and a whole import can be undone.

## Entities

Ownership column: how a row is tied to the account that owns it. Sensitivity:
`Health` means it is personal health information.

| Model                      | PK          | Owner path                        | Key constraints / indexes                                                 | Sensitivity                                         |
| -------------------------- | ----------- | --------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------- |
| `User`                     | `id` (cuid) | itself                            | `@@unique(email)`                                                         | PII (email, `passwordHash`)                         |
| `AuthIdentity`             | `id`        | `userId`                          | `@@unique([provider, providerAccountId])`, `@@unique([userId, provider])` | PII                                                 |
| `PasswordResetToken`       | `id`        | `userId`                          | `@@unique(tokenHash)`                                                     | Auth secret (hashed)                                |
| `Profile`                  | `id`        | `userId` (1:1)                    | `@@unique(userId)`                                                        | Health (condition, targets)                         |
| `Device`                   | `id`        | `userId`                          | `@@index(userId)`                                                         | Low (device metadata)                               |
| `ImportBatch`              | `id`        | `userId`                          | `@@index([userId, createdAt])`                                            | Metadata                                            |
| `ImportIssue`              | `id`        | via `batchId`                     | `@@index(batchId)`                                                        | Health — `rawRow` holds a rejected raw row          |
| `GlucoseReading`           | `id`        | `userId`                          | `@@unique([userId, dedupeKey])`, `@@index([userId, takenAt])`             | Health (core)                                       |
| `Meal`                     | `id`        | `userId`                          | same shape                                                                | Health                                              |
| `FoodItem`                 | `id`        | via `mealId`                      | `@@index(mealId)`                                                         | Health                                              |
| `ExerciseSession`          | `id`        | `userId`                          | same shape                                                                | Health                                              |
| `SleepSession`             | `id`        | `userId`                          | same shape                                                                | Health                                              |
| `MedicationEvent`          | `id`        | `userId`                          | same shape                                                                | Health                                              |
| `WeightMeasurement`        | `id`        | `userId`                          | same shape                                                                | Health                                              |
| `BloodPressureMeasurement` | `id`        | `userId`                          | same shape                                                                | Health                                              |
| `HydrationEvent`           | `id`        | `userId`                          | same shape                                                                | Health                                              |
| `SymptomEntry`             | `id`        | `userId`                          | same shape                                                                | Health                                              |
| `MoodEntry`                | `id`        | `userId`                          | same shape                                                                | Health (mental health)                              |
| `NoteEntry`                | `id`        | `userId`                          | same shape                                                                | Health (free text)                                  |
| `Insight`                  | `id`        | `userId`                          | `@@index([userId, generatedAt])`                                          | Health-derived — **currently unwritten, see below** |
| `AIConversation`           | `id`        | `userId`                          | `@@index([userId, updatedAt])`                                            | Health (discussion about the user's data)           |
| `AIMessage`                | `id`        | via `conversationId`              | `@@index([conversationId, createdAt])`                                    | Health (`content`, `evidence`)                      |
| `AuditEvent`               | `id`        | `userId` (**nullable by design**) | `@@index([userId, createdAt])`                                            | Security metadata only — never health values        |

### Timestamp conventions

`takenAt` means "when this actually happened" and is used uniformly across all
eleven health-record models — it is the column every range scan and chart is
built on. `endedAt` is a genuinely different concept and appears only where an
event has duration (`ExerciseSession`, `SleepSession`). `createdAt` /
`updatedAt` are row bookkeeping and are never used as the clinical time. There
is no `occurredAt` or `loggedAt`; the vocabulary is consistent.

### Provenance asymmetry

`GlucoseReading`, `Meal`, `ExerciseSession`, `SleepSession`, `MedicationEvent`,
`WeightMeasurement` and `NoteEntry` carry import provenance
(`importBatchId`, `externalId`, `rawPayload`). `BloodPressureMeasurement`,
`HydrationEvent`, `SymptomEntry` and `MoodEntry` do not, because no connector
emits those kinds — the `NormalizedRecord` union in `lib/import/types.ts` has
no variant for them. This is deliberate, not an oversight, but it means adding
an import connector for any of those four requires a migration adding the
provenance columns first, following the pattern of
`20260825020301_import_provenance_all_record_types`.

## Validation

Zod schemas in `lib/validation.ts` are the single validation layer for user
input; the database enforces structure (types, nullability, uniqueness,
referential integrity) but not domain ranges.

Bounded fields — `MoodEntry.mood`, `SymptomEntry.severity` and
`SleepSession.quality` (all 1–5) — are constrained only in Zod, with no
Postgres `CHECK`. Anything writing through Prisma directly (a script, a seed, a
future connector) can therefore insert an out-of-range value. Likewise
`Profile.targetLowMgdl` and `targetHighMgdl` are not cross-validated as
`low < high` at either layer. Both are accepted, documented limitations rather
than defects in the current user-facing paths, which all go through Zod.

## Deletion and retention

Two distinct destructive operations, with deliberately different scopes.

**Delete all records** (`deleteAllRecordsAction`) clears the eleven health
tables plus `AIConversation` and `ImportBatch`, in one transaction. The last
two matter: an AI conversation stores the question, the answer and the evidence
behind it, and an import batch's `ImportIssue.rawRow` holds raw rejected health
rows. Before the September 2026 audit both were left behind, so a user who had
asked the app to delete everything still had free-text health discussion and
raw imported rows in the database. `AIMessage` and `ImportIssue` cascade from
those parents. The account itself, its profile and its preferences survive.

**Delete account** (`prisma.user.delete`) relies on database cascades. Every
model owned by a user cascades from `User`, directly or transitively
(`FoodItem` via `Meal`, `AIMessage` via `AIConversation`, `ImportIssue` via
`ImportBatch`), so one delete removes everything with no orphans — verified by
an integration test.

`AuditEvent` is the single deliberate exception. Its `userId` is nullable
precisely so that a security event outlives the account it names, and the
foreign key is `ON DELETE SET NULL`. Under the `ON DELETE CASCADE` it shipped
with, deleting an account also destroyed the `auth.account_delete` event
written moments earlier — the audit trail erased itself exactly when it
mattered. Migration
`20260901000000_audit_event_survives_account_delete` corrects this. The
retained row carries no user id and no health values, so it is no longer
personal data, but the action stays on the record.

`Device` and `ImportBatch` references from health records are `ON DELETE SET
NULL`, so deleting a device or undoing a batch never removes the readings
themselves — it only detaches their provenance.

## Query patterns

1. **Windowed range scan** — `WHERE userId = ? AND takenAt BETWEEN ? AND ?
ORDER BY takenAt`. The dashboard, glucose log, health page, reports and
   `loadAnalyticsWindow` all use this shape. Backed by `@@index([userId,
takenAt])` on every health model; Postgres walks the b-tree backwards for
   `DESC` without needing a separate index.
2. **Keyset pagination** — `pageGlucose` orders by `(takenAt DESC, id DESC)`
   with an id cursor, using the same composite index plus the primary key.
   Keyset rather than `OFFSET` so deep pages stay cheap and stable under
   concurrent writes.
3. **Dedupe pre-check** — before an import commits, `existingKeysFor` selects
   only `dedupeKey` for the user, per kind. Served by the
   `@@unique([userId, dedupeKey])` index.
4. **Ownership-scoped mutation** — `deleteMany({ where: { id, userId } })`.
5. **Aggregate reporting** — `count` / `aggregate` scoped by
   `{ userId, takenAt: { gte } }`.

Indexing is deliberately lean: one composite index per health table matching
the dominant access path, plus the dedupe unique. The audit found no missing
index for any query the application actually issues, and no unused ones.

## Data flows

### Import

`parseFile` → `detectConnector` → connector `parse()` → `normalize` → `dedupe`
→ `summary` → user reviews the preview → `commitImport`.

Connectors are pure transforms: they never touch the database, and they never
silently drop a row — anything not convertible to a `NormalizedRecord` must
produce a `RowIssue`. Registration order in
`lib/import/connectors/registry.ts` is the tie-break, so specific connectors
must precede generic fallbacks.

`commitImport` creates the `ImportBatch` as `PENDING`, writes every record type
inside one `$transaction` (with `skipDuplicates` as a second line of defence
behind the unique constraint), then marks the batch `COMPLETED`. That final
status update is outside the transaction, so a crash in the gap leaves a batch
stuck `PENDING` with its rows written — a status inconsistency, not data loss.
`undoImport` verifies batch ownership first, then deletes the rows
transactionally.

XML input is parsed with a hard rejection of any `DOCTYPE` internal subset:
entity-expansion bombs blow up _after_ the file-size check passes, so
`MAX_FILE_BYTES` cannot bound them. None of the supported formats uses a DTD.

### Export

`export-service.ts` queries every model independently, each scoped by
`userId`, and assembles one versioned JSON payload or per-type RFC4180 CSV.
The queries are not wrapped in a single snapshot, so a concurrent write during
an export can leave different tables representing slightly different instants.
Under Postgres read-committed this is expected and the impact on a personal
export is negligible.

### Analytics and AI

`analyzeUser()` loads raw records; `toEvidenceBundle()` is the only function
allowed to build the `EvidenceBundle` the AI layer sees. Raw health records
never reach a provider — an integration test asserts the bundle contains
nothing shaped like a list of readings.

## Known issues and migration notes

- **`Insight` is unwritten.** The model, its `evidence` JSON column and its
  index exist, but no application code reads or writes the table; insights are
  computed per request and never persisted. Retained rather than dropped
  because removing it is a destructive migration and the intent may be to
  persist them later. Flagged for a product decision, not silently deleted.
- **No password-reset flow.** `PasswordResetToken` is modelled but unused; no
  route creates or consumes one. When built, it needs the same rigour as the
  rest of `lib/auth/`: hashed at rest, single-use, short TTL, and identical
  responses regardless of whether the email exists.
- **Seed coverage is partial.** `prisma/seed.ts` creates users, profiles,
  glucose, meals, exercise, sleep and medication only. The other record types,
  devices and import batches are not seeded, so screens showing them look empty
  against seeded data.

Any future schema change must be weighed against existing data, migration
order, rollback, and application compatibility. Additive nullable columns and
foreign-key action changes (like the `AuditEvent` fix above) are safe to ship
independently; column removals and type changes are not.
