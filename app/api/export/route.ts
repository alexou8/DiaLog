/**
 * GET /api/export?format=json|csv&type=<recordType>
 *
 * Streams the signed-in user's own data as a downloadable file. There is no
 * userId query parameter — the account is always the one attached to the
 * session cookie, so this endpoint can never be pointed at someone else's
 * data by editing the URL.
 */
import { NextRequest, NextResponse } from 'next/server';
import { getCurrentUser } from '@/lib/auth/current-user';
import { audit } from '@/lib/auth/audit';
import { rateLimit } from '@/lib/auth/rate-limit';
import { buildCsvExport, buildJsonExport, EXPORT_RECORD_TYPES, type ExportRecordType } from '@/lib/services/export-service';

export const dynamic = 'force-dynamic';

const EXPORT_LIMIT = 20;
const EXPORT_WINDOW_MS = 60 * 60_000;

function isExportRecordType(value: string | null): value is ExportRecordType {
  return !!value && (EXPORT_RECORD_TYPES as readonly string[]).includes(value);
}

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

export async function GET(request: NextRequest): Promise<Response> {
  const user = await getCurrentUser();
  if (!user) {
    return NextResponse.json({ error: 'Please sign in to download your data.' }, { status: 401 });
  }

  const limit = rateLimit(`export:${user.id}`, EXPORT_LIMIT, EXPORT_WINDOW_MS);
  if (!limit.ok) {
    return NextResponse.json(
      { error: 'Too many export requests. Please wait a while and try again.' },
      { status: 429, headers: { 'Retry-After': String(limit.retryAfterSeconds) } },
    );
  }

  const format = request.nextUrl.searchParams.get('format') === 'csv' ? 'csv' : 'json';
  const noStore = { 'Cache-Control': 'no-store' } as const;

  if (format === 'csv') {
    const typeParam = request.nextUrl.searchParams.get('type');
    if (!isExportRecordType(typeParam)) {
      return NextResponse.json(
        { error: `Please choose a record type: ${EXPORT_RECORD_TYPES.join(', ')}.` },
        { status: 400, headers: noStore },
      );
    }

    const { csv, filename } = await buildCsvExport(user.id, typeParam);
    await audit({ userId: user.id, action: 'data.export', entity: typeParam, detail: 'csv' });

    return new Response(csv, {
      status: 200,
      headers: {
        'Content-Type': 'text/csv; charset=utf-8',
        'Content-Disposition': `attachment; filename="${filename}-${today()}.csv"`,
        'Cache-Control': 'no-store',
      },
    });
  }

  const data = await buildJsonExport(user.id);
  await audit({ userId: user.id, action: 'data.export', detail: 'json' });

  return new Response(JSON.stringify(data, null, 2), {
    status: 200,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Content-Disposition': `attachment; filename="dialog-export-${today()}.json"`,
      'Cache-Control': 'no-store',
    },
  });
}
