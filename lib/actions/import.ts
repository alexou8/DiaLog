'use server';

import { revalidatePath } from 'next/cache';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { audit } from '@/lib/auth/audit';
import { RATE_LIMITS, rateLimit } from '@/lib/auth/rate-limit';
import {
  commitImport,
  ImportError,
  prepareImport,
  undoImport,
} from '@/lib/services/import-service';
import type { ImportSummary } from '@/lib/import/summary';

export interface ImportState {
  ok: boolean;
  stage: 'idle' | 'reviewed' | 'committed';
  message?: string;
  connectorName?: string;
  summary?: ImportSummary;
  preview?: { kind: string; when: string; detail: string }[];
  imported?: number;
}

const ALLOWED_EXTENSIONS = ['.csv', '.tsv', '.txt', '.json', '.xml', '.xlsx', '.xls'];

/** Shared file extraction and validation for both stages. */
async function readUpload(formData: FormData) {
  const file = formData.get('file');
  if (!(file instanceof File) || file.size === 0) {
    throw new ImportError('Please choose a file to import.');
  }
  const lower = file.name.toLowerCase();
  if (!ALLOWED_EXTENSIONS.some((extension) => lower.endsWith(extension))) {
    throw new ImportError(
      `DiaLog can read ${ALLOWED_EXTENSIONS.join(', ')} files. That file has a different extension.`,
    );
  }
  return { file, bytes: await file.arrayBuffer() };
}

function describe(
  record: { kind: string; takenAt: Date },
  timezone: string,
  locale: string,
): string {
  return new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(record.takenAt);
}

/** Stage one: parse the file and report what would happen. Writes nothing. */
export async function analyzeImportAction(
  _prev: ImportState | null,
  formData: FormData,
): Promise<ImportState> {
  const user = await requireOnboardedUser();
  const limit = rateLimit(
    `import:${user.id}`,
    RATE_LIMITS.import.limit,
    RATE_LIMITS.import.windowMs,
  );
  if (!limit.ok) {
    return {
      ok: false,
      stage: 'idle',
      message: 'That is a lot of imports at once. Please try again shortly.',
    };
  }

  try {
    const { file, bytes } = await readUpload(formData);
    const prepared = await prepareImport({
      userId: user.id,
      filename: file.name,
      mimeType: file.type,
      bytes,
      timezone: user.profile.timezone,
    });

    return {
      ok: true,
      stage: 'reviewed',
      connectorName: prepared.connectorName,
      summary: prepared.summary,
      preview: prepared.preview.map((record) => ({
        kind: record.kind,
        when: describe(record, user.profile.timezone, user.profile.locale),
        detail:
          record.kind === 'glucose'
            ? `${Math.round(record.valueMgdl)} mg/dL (stored value)`
            : record.kind === 'meal'
              ? record.description
              : record.kind === 'medication'
                ? record.name
                : record.kind === 'exercise'
                  ? `${record.activity}, ${record.durationMin} min`
                  : 'Not recorded',
      })),
    };
  } catch (error) {
    if (error instanceof ImportError) return { ok: false, stage: 'idle', message: error.message };
    throw error;
  }
}

/** Stage two: the same file again, this time saved. */
export async function commitImportAction(
  _prev: ImportState | null,
  formData: FormData,
): Promise<ImportState> {
  const user = await requireOnboardedUser();
  const limit = rateLimit(
    `import:${user.id}`,
    RATE_LIMITS.import.limit,
    RATE_LIMITS.import.windowMs,
  );
  if (!limit.ok) {
    return {
      ok: false,
      stage: 'idle',
      message: 'That is a lot of imports at once. Please try again shortly.',
    };
  }

  try {
    const { file, bytes } = await readUpload(formData);
    const prepared = await prepareImport({
      userId: user.id,
      filename: file.name,
      mimeType: file.type,
      bytes,
      timezone: user.profile.timezone,
    });

    const { imported } = await commitImport({
      userId: user.id,
      filename: file.name,
      mimeType: file.type,
      byteSize: file.size,
      prepared,
    });

    await audit({
      userId: user.id,
      action: 'data.import',
      detail: `${prepared.connectorId}:${imported}`,
    });
    revalidatePath('/app');
    revalidatePath('/app/glucose');
    revalidatePath('/app/import');

    return {
      ok: true,
      stage: 'committed',
      connectorName: prepared.connectorName,
      summary: prepared.summary,
      imported,
    };
  } catch (error) {
    if (error instanceof ImportError) return { ok: false, stage: 'idle', message: error.message };
    throw error;
  }
}

export async function undoImportAction(formData: FormData): Promise<void> {
  const user = await requireOnboardedUser();
  const batchId = String(formData.get('batchId') ?? '');
  if (!batchId) return;
  const removed = await undoImport(user.id, batchId);
  await audit({
    userId: user.id,
    action: 'data.import_undo',
    entityId: batchId,
    detail: String(removed),
  });
  revalidatePath('/app/import');
  revalidatePath('/app');
}
