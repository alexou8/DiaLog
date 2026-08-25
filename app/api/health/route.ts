import { NextResponse } from 'next/server';
import { prisma } from '@/lib/db/prisma';

export const dynamic = 'force-dynamic';

/**
 * Liveness/readiness probe for uptime monitoring.
 *
 * Reports whether the process is up and whether the database answers. It
 * deliberately exposes no counts, no user information and no configuration —
 * only what an operator needs to know.
 */
export async function GET() {
  const startedAt = Date.now();
  let database = false;
  try {
    await prisma.$queryRaw`SELECT 1`;
    database = true;
  } catch {
    database = false;
  }

  return NextResponse.json(
    {
      status: database ? 'ok' : 'degraded',
      database,
      uptimeSeconds: Math.round(process.uptime()),
      checkedInMs: Date.now() - startedAt,
    },
    { status: database ? 200 : 503, headers: { 'Cache-Control': 'no-store' } },
  );
}
