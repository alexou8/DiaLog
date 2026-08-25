import { prisma } from '@/lib/db/prisma';

/**
 * Record a security-relevant action. Never pass health values or free text
 * entered by the user — `detail` is for non-sensitive context such as a
 * connector id or a record count.
 */
export async function audit(params: {
  userId?: string | null;
  action: string;
  entity?: string;
  entityId?: string;
  detail?: string;
}): Promise<void> {
  try {
    await prisma.auditEvent.create({
      data: {
        userId: params.userId ?? null,
        action: params.action,
        entity: params.entity ?? null,
        entityId: params.entityId ?? null,
        detail: params.detail ?? null,
      },
    });
  } catch {
    // Auditing must never break the user's request.
  }
}
