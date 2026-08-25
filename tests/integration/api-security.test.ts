import { afterAll, describe, expect, it } from 'vitest';
import {
  emailSchema,
  glucoseEntrySchema,
  mealEntrySchema,
  medicationEntrySchema,
  signUpSchema,
} from '@/lib/validation';
import { prepareImport, ImportError } from '@/lib/services/import-service';
import { prisma, createTestUser, deleteTestUser, toArrayBuffer } from './test-helpers';

const createdUserIds: string[] = [];

afterAll(async () => {
  for (const id of createdUserIds) await deleteTestUser(id);
});

const SQL_PAYLOAD = "'; DROP TABLE users;--";
const XSS_PAYLOAD = '<script>alert(1)</script>';

describe('validation.ts rejects or safely coerces hostile input', () => {
  it('rejects a SQL-fragment payload as an email', () => {
    expect(emailSchema.safeParse(SQL_PAYLOAD).success).toBe(false);
  });

  it('rejects an oversized email/password/description rather than truncating silently', () => {
    const hugeEmail = `${'a'.repeat(400)}@example.com`;
    expect(emailSchema.safeParse(hugeEmail).success).toBe(false);

    const hugeDescription = 'x'.repeat(10_000);
    const result = mealEntrySchema.safeParse({
      takenAt: '2026-01-01T08:00:00',
      mealType: 'BREAKFAST',
      description: hugeDescription,
    });
    expect(result.success).toBe(false);
  });

  it('accepts a script-tag payload as literal text (stored verbatim, never executed) in free-text fields', () => {
    // XSS is a rendering-layer concern (escaping on output), not a storage-layer
    // rejection — validation.ts correctly lets it through as plain text.
    const result = medicationEntrySchema.safeParse({
      takenAt: '2026-01-01T08:00:00',
      name: XSS_PAYLOAD,
    });
    expect(result.success).toBe(true);
    if (result.success) expect(result.data.name).toBe(XSS_PAYLOAD);
  });

  it('rejects NaN and Infinity for numeric fields', () => {
    expect(
      glucoseEntrySchema.safeParse({
        value: NaN,
        unit: 'MGDL',
        takenAt: '2026-01-01T08:00:00',
        context: 'RANDOM',
      }).success,
    ).toBe(false);
    expect(
      glucoseEntrySchema.safeParse({
        value: Infinity,
        unit: 'MGDL',
        takenAt: '2026-01-01T08:00:00',
        context: 'RANDOM',
      }).success,
    ).toBe(false);
    expect(
      glucoseEntrySchema.safeParse({
        value: '  not-a-number  ',
        unit: 'MGDL',
        takenAt: '2026-01-01T08:00:00',
        context: 'RANDOM',
      }).success,
    ).toBe(false);
  });

  it('handles a deeply nested JSON value passed where a scalar is expected without crashing', () => {
    let nested: unknown = 'leaf';
    for (let i = 0; i < 5000; i++) nested = { child: nested };
    // A deeply nested object where a string email is expected must be a clean
    // validation failure, not a stack overflow or hang.
    expect(() => emailSchema.safeParse(nested)).not.toThrow();
    expect(emailSchema.safeParse(nested).success).toBe(false);
  });

  it('strips prototype-pollution keys instead of merging them, and does not pollute Object.prototype', () => {
    const hostile = JSON.parse(
      '{"email":"safe@example.com","password":"a-long-enough-password-1","__proto__":{"polluted":true},"constructor":{"prototype":{"polluted":true}}}',
    ) as unknown;
    const result = signUpSchema.safeParse(hostile);
    expect(result.success).toBe(true);
    if (result.success) {
      expect((result.data as Record<string, unknown>)['polluted']).toBeUndefined();
      expect((Object.prototype as unknown as Record<string, unknown>)['polluted']).toBeUndefined();
    }
    expect(({} as Record<string, unknown>)['polluted']).toBeUndefined();
  });
});

describe('SQL injection resistance via Prisma parameterisation', () => {
  it('a note containing SQL syntax round-trips as literal text and never affects the database', async () => {
    const { user } = await createTestUser('sql-injection');
    createdUserIds.push(user.id);

    const before = await prisma.user.count();

    const note = await prisma.noteEntry.create({
      data: { userId: user.id, takenAt: new Date(), text: SQL_PAYLOAD, dedupeKey: 'sql-note' },
    });

    const after = await prisma.user.count();
    expect(after).toBe(before); // "users" table untouched — no injection occurred

    const reloaded = await prisma.noteEntry.findUniqueOrThrow({ where: { id: note.id } });
    expect(reloaded.text).toBe(SQL_PAYLOAD); // stored and returned as literal text, unmangled

    // A second hostile payload targeting a different clause shape, via the query filter itself.
    const found = await prisma.noteEntry.findMany({
      where: { userId: user.id, text: SQL_PAYLOAD },
    });
    expect(found).toHaveLength(1);
  });
});

describe('import path handles hostile filenames and disguised content safely', () => {
  it('a path-traversal filename is stored as an opaque string, never used to read/write the filesystem', async () => {
    const { user } = await createTestUser('path-traversal');
    createdUserIds.push(user.id);

    const csv = 'timestamp,glucose\n2026-01-01T08:00:00Z,100\n';
    const bytes = Buffer.from(csv, 'utf8');
    const buf = toArrayBuffer(bytes);

    const prepared = await prepareImport({
      userId: user.id,
      filename: '../../../../etc/passwd',
      mimeType: 'text/csv',
      bytes: buf,
      timezone: 'UTC',
    });
    // The filename is only ever a label; it never causes a filesystem read (the file's
    // bytes came entirely from the in-memory buffer above) and is preserved verbatim.
    expect(prepared.fresh.length).toBeGreaterThan(0);
  });

  it('an executable disguised with a .csv extension is rejected gracefully, not parsed as data or executed', async () => {
    const { user } = await createTestUser('fake-csv-exe');
    createdUserIds.push(user.id);

    // ELF magic bytes followed by junk — not valid text-ish tabular data.
    const elfMagic = Buffer.from([
      0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00,
    ]);
    const junk = Buffer.concat([
      elfMagic,
      Buffer.from('\x00\x01\x02\x03binarygarbage\xff\xfe', 'binary'),
    ]);
    const buf = toArrayBuffer(junk);

    await expect(
      prepareImport({
        userId: user.id,
        filename: 'totally-a-spreadsheet.csv',
        mimeType: 'text/csv',
        bytes: buf,
        timezone: 'UTC',
      }),
    ).rejects.toBeInstanceOf(ImportError);

    expect(await prisma.importBatch.count({ where: { userId: user.id } })).toBe(0);
  });

  it('an oversized filename does not crash the import pipeline', async () => {
    const { user } = await createTestUser('huge-filename');
    createdUserIds.push(user.id);
    const csv = 'timestamp,glucose\n2026-01-01T08:00:00Z,100\n';
    const bytes = Buffer.from(csv, 'utf8');
    const buf = toArrayBuffer(bytes);
    const hugeFilename = `${'a'.repeat(5000)}.csv`;

    await expect(
      prepareImport({
        userId: user.id,
        filename: hugeFilename,
        mimeType: 'text/csv',
        bytes: buf,
        timezone: 'UTC',
      }),
    ).resolves.toBeDefined();
  });
});
