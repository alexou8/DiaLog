-- AuditEvent.userId was ON DELETE CASCADE, so deleting an account destroyed
-- that account's entire audit trail -- including the `auth.account_delete`
-- event written immediately before the delete. AuditEvent.userId is nullable
-- specifically so a security event can outlive the account it names, so the
-- constraint is switched to ON DELETE SET NULL to match that intent.
--
-- Non-destructive: no rows are added, removed or rewritten; only the foreign
-- key action changes. Safe to roll back by restoring ON DELETE CASCADE.
ALTER TABLE "public"."AuditEvent" DROP CONSTRAINT "AuditEvent_userId_fkey";

ALTER TABLE "public"."AuditEvent" ADD CONSTRAINT "AuditEvent_userId_fkey"
  FOREIGN KEY ("userId") REFERENCES "public"."User"("id")
  ON DELETE SET NULL ON UPDATE CASCADE;
