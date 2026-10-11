/**
 * The import preview used to print "99 mg/dL (stored value)" for everyone.
 * Display unit is a per-user preference (AGENTS.md "Units"), so preview text
 * goes through lib/domain/units like every other screen.
 */
import { describe, expect, it } from 'vitest';
import { previewDetail } from '@/lib/import/preview';
import type { NormalizedRecord } from '@/lib/import/types';

const T = new Date('2026-01-10T08:00:00Z');
const glucose: NormalizedRecord = {
  kind: 'glucose',
  takenAt: T,
  valueMgdl: 99.1,
  context: 'UNKNOWN',
  raw: {},
};

describe('previewDetail', () => {
  it("shows glucose in the user's preferred unit", () => {
    expect(previewDetail(glucose, 'MMOLL', 'en-CA')).toBe('5.5 mmol/L');
    expect(previewDetail(glucose, 'MGDL', 'en-CA')).toBe('99 mg/dL');
  });

  it("uses the user's locale for the decimal separator", () => {
    expect(previewDetail(glucose, 'MMOLL', 'fr-CA')).toBe('5,5 mmol/L');
  });

  it('describes other record kinds without units', () => {
    expect(
      previewDetail(
        { kind: 'exercise', takenAt: T, activity: 'walk', durationMin: 20, raw: {} },
        'MMOLL',
        'en-CA',
      ),
    ).toBe('walk, 20 min');
    expect(
      previewDetail(
        { kind: 'meal', takenAt: T, description: 'Oatmeal', mealType: 'BREAKFAST', raw: {} },
        'MGDL',
        'en-CA',
      ),
    ).toBe('Oatmeal');
  });
});
