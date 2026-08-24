import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { parseCsv } from '@/lib/import/parse';
import { dialogLegacyConnector } from '@/lib/import/connectors/dialog-legacy';
import type { GlucoseRecord, MealRecord, MedicationRecord, ParsedFile } from '@/lib/import/types';

const NOW = new Date('2026-08-24T12:00:00Z');

function loadFixture(relPath: string): ParsedFile {
  const text = readFileSync(path.join(process.cwd(), relPath), 'utf8');
  return { filename: path.basename(relPath), mimeType: 'text/csv', text, rows: parseCsv(text) };
}

describe('dialogLegacyConnector — long/event format (ml/data/sample_logs.csv)', () => {
  const file = loadFixture('ml/data/sample_logs.csv');

  it('detects the long format with high confidence', async () => {
    expect(dialogLegacyConnector.detect(file)).toBeGreaterThan(0.8);
  });

  it('parses med/meal/glucose events without issues', async () => {
    const result = await dialogLegacyConnector.parse(file, { now: NOW });
    expect(result.issues).toHaveLength(0);
    expect(result.records.length).toBeGreaterThan(0);
  });

  it('produces a medication record for a "med" event', async () => {
    const result = await dialogLegacyConnector.parse(file, { now: NOW });
    const med = result.records.find((r) => r.kind === 'medication') as MedicationRecord | undefined;
    expect(med).toBeDefined();
    expect(med?.name).toBe('Metformin');
    expect(med?.dose).toBe('1');
  });

  it('produces a meal record with carbsG for a "meal" event', async () => {
    const result = await dialogLegacyConnector.parse(file, { now: NOW });
    const meal = result.records.find((r) => r.kind === 'meal') as MealRecord | undefined;
    expect(meal?.carbsG).toBe(45);
  });

  it('produces a glucose record already in mg/dL for a "glucose" event', async () => {
    const result = await dialogLegacyConnector.parse(file, { now: NOW });
    const glucose = result.records.find((r) => r.kind === 'glucose') as GlucoseRecord | undefined;
    expect(glucose?.valueMgdl).toBe(168);
    expect(result.detectedUnit).toBe('MGDL');
  });
});

describe('dialogLegacyConnector — wide/hourly format (ml/data/sample_glucose_data.csv)', () => {
  const file = loadFixture('ml/data/sample_glucose_data.csv');

  it('detects the wide format with high confidence', async () => {
    expect(dialogLegacyConnector.detect(file)).toBeGreaterThan(0.8);
  });

  it('parses every hourly row into at least a glucose record', async () => {
    const result = await dialogLegacyConnector.parse(file, { now: NOW });
    expect(result.issues).toHaveLength(0);
    const glucoseRecords = result.records.filter((r) => r.kind === 'glucose');
    expect(glucoseRecords.length).toBe(result.rowsTotal);
  });

  it('also derives meal, medication, and exercise records from co-occurring columns', async () => {
    const result = await dialogLegacyConnector.parse(file, { now: NOW });
    expect(result.records.some((r) => r.kind === 'meal')).toBe(true);
    expect(result.records.some((r) => r.kind === 'medication')).toBe(true);
    expect(result.records.some((r) => r.kind === 'exercise')).toBe(true);
  });
});
