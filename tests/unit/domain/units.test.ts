/**
 * lib/domain/units.ts is the single canonical conversion point: storage is
 * always mg/dL and the display unit is a per-user preference, so a rounding
 * or direction error here silently misreports every glucose value in the
 * product. It was previously only exercised indirectly, through the import
 * normalisation tests, which never call toMgdl/fromMgdl themselves.
 */
import { describe, expect, it } from 'vitest';
import {
  GLUCOSE_ENTRY_BOUNDS,
  MGDL_PER_MMOLL,
  formatGlucose,
  formatGlucoseWithUnit,
  fromMgdl,
  inferGlucoseUnit,
  isPlausibleGlucose,
  mgdlToMmoll,
  mmollToMgdl,
  toMgdl,
  unitLabel,
  unitPrecision,
} from '@/lib/domain/units';

describe('round-tripping through the user-facing unit', () => {
  it('returns the original mg/dL value for every clinically relevant reading', () => {
    for (let mgdl = 20; mgdl <= 600; mgdl += 1) {
      expect(toMgdl(fromMgdl(mgdl, 'MMOLL'), 'MMOLL')).toBeCloseTo(mgdl, 9);
      // mg/dL is the canonical unit, so its round trip must be exact.
      expect(toMgdl(fromMgdl(mgdl, 'MGDL'), 'MGDL')).toBe(mgdl);
    }
  });

  it('round-trips mmol/L values back to themselves', () => {
    for (const mmoll of [2.2, 3.9, 5.5, 7.0, 10.0, 13.9, 22.2, 33.3]) {
      expect(mgdlToMmoll(mmollToMgdl(mmoll))).toBeCloseTo(mmoll, 9);
    }
  });
});

describe('conversion direction and magnitude', () => {
  it('uses the molar factor in the right direction', () => {
    // A wrong direction is the failure mode that matters: 5.5 mmol/L is
    // ~99 mg/dL (normal), while 5.5 mg/dL would be a medical emergency.
    expect(mmollToMgdl(5.5)).toBeCloseTo(99.1, 1);
    expect(mgdlToMmoll(99.1)).toBeCloseTo(5.5, 1);
    expect(MGDL_PER_MMOLL).toBeCloseTo(18.0182, 4);
  });

  it('treats mg/dL as canonical and does not convert it', () => {
    expect(fromMgdl(120, 'MGDL')).toBe(120);
    expect(toMgdl(120, 'MGDL')).toBe(120);
  });

  it('handles zero without dividing or multiplying into NaN', () => {
    expect(fromMgdl(0, 'MMOLL')).toBe(0);
    expect(toMgdl(0, 'MMOLL')).toBe(0);
    expect(Number.isFinite(mgdlToMmoll(0))).toBe(true);
  });
});

describe('display formatting', () => {
  it('uses the conventional precision per unit', () => {
    expect(unitPrecision('MGDL')).toBe(0);
    expect(unitPrecision('MMOLL')).toBe(1);
    expect(formatGlucose(99.1, 'MGDL')).toBe('99');
    expect(formatGlucose(99.1, 'MMOLL')).toBe('5.5');
  });

  it('appends the matching unit label', () => {
    expect(unitLabel('MMOLL')).toBe('mmol/L');
    expect(unitLabel('MGDL')).toBe('mg/dL');
    expect(formatGlucoseWithUnit(99.1, 'MMOLL')).toBe('5.5 mmol/L');
    expect(formatGlucoseWithUnit(120, 'MGDL')).toBe('120 mg/dL');
  });
});

describe('entry plausibility bounds', () => {
  it('accepts values inside the bounds and rejects values outside them', () => {
    for (const unit of ['MGDL', 'MMOLL'] as const) {
      const { min, max } = GLUCOSE_ENTRY_BOUNDS[unit];
      expect(isPlausibleGlucose(min, unit)).toBe(true);
      expect(isPlausibleGlucose(max, unit)).toBe(true);
      expect(isPlausibleGlucose(min - 0.1, unit)).toBe(false);
      expect(isPlausibleGlucose(max + 0.1, unit)).toBe(false);
    }
  });

  it('rejects a mmol/L number entered as if it were mg/dL', () => {
    // 5.5 is a normal mmol/L reading but an impossible mg/dL one; catching
    // this at entry is the point of having separate bounds per unit.
    expect(isPlausibleGlucose(5.5, 'MMOLL')).toBe(true);
    expect(isPlausibleGlucose(5.5, 'MGDL')).toBe(false);
  });
});

describe('inferGlucoseUnit', () => {
  it('infers mmol/L from a small-magnitude series and mg/dL from a large one', () => {
    expect(inferGlucoseUnit([4.2, 5.5, 6.1, 7.8])).toBe('MMOLL');
    expect(inferGlucoseUnit([88, 104, 132, 156])).toBe('MGDL');
  });

  it('returns null rather than guessing when there is nothing to go on', () => {
    expect(inferGlucoseUnit([])).toBeNull();
  });
});
