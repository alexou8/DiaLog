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
import {
  csvExportFilename,
  EXPORT_RECORD_TYPES,
  openCsvExport,
  openJsonExport,
  type ExportRecordType,
} from '@/lib/services/export-service';

export const dynamic = 'force-dynamic';

const EXPORT_LIMIT = 20;
const EXPORT_WINDOW_MS = 60 * 60_000;

function isExportRecordType(value: string | null): value is ExportRecordType {
  return !!value && (EXPORT_RECORD_TYPES as readonly string[]).includes(value);
}

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

/**
 * Stream the export instead of building it in memory. An export is now the
 * whole history (no row cap — see lib/services/export-service.ts), which for
 * years of CGM data is far more than should sit in one function's memory or
 * one buffered response body. Pull-based, so a slow download slows the
 * database reads rather than queueing pages up.
 *
 * If a page fails mid-way the stream is errored, which aborts the transfer:
 * the browser reports a failed download instead of saving a file that ends
 * early but looks complete. The thrown error is never logged as-is — it may
 * be a Prisma error carrying query arguments — only a fixed line is.
 *
 * The profile and the first page are read by `open*Export` BEFORE the
 * Response exists: a missing profile or a database that is down must become
 * a 500 with a message, not a 200 whose body aborts (and the audit row is
 * only written once that first read succeeds, so a failed export is never
 * recorded as one).
 */
function toStream(chunks: AsyncGenerator<string>): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const { value, done } = await chunks.next();
        if (done) controller.close();
        else controller.enqueue(encoder.encode(value));
      } catch {
        console.error('Data export failed while streaming. Check database connectivity.');
        controller.error(new Error('Export failed'));
      }
    },
    async cancel() {
      await chunks.return(undefined);
    },
  });
}

async function openExport(
  open: () => Promise<AsyncGenerator<string>>,
): Promise<AsyncGenerator<string> | null> {
  try {
    return await open();
  } catch {
    console.error('Data export could not start. Check database connectivity.');
    return null;
  }
}

function exportUnavailable(headers: Record<string, string>): Response {
  return NextResponse.json(
    { error: 'Your export could not be prepared right now. Please try again in a moment.' },
    { status: 500, headers },
  );
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

    const chunks = await openExport(() => openCsvExport(user.id, typeParam));
    if (!chunks) return exportUnavailable(noStore);
    await audit({ userId: user.id, action: 'data.export', entity: typeParam, detail: 'csv' });

    return new Response(toStream(chunks), {
      status: 200,
      headers: {
        'Content-Type': 'text/csv; charset=utf-8',
        'Content-Disposition': `attachment; filename="${csvExportFilename(typeParam)}-${today()}.csv"`,
        'Cache-Control': 'no-store',
      },
    });
  }

  const chunks = await openExport(() => openJsonExport(user.id));
  if (!chunks) return exportUnavailable(noStore);
  await audit({ userId: user.id, action: 'data.export', detail: 'json' });

  return new Response(toStream(chunks), {
    status: 200,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Content-Disposition': `attachment; filename="dialog-export-${today()}.json"`,
      'Cache-Control': 'no-store',
    },
  });
}
