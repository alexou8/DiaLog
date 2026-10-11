/**
 * lib/domain/units.ts is the single canonical conversion point: storage is
 * always mg/dL and the display unit is a per-user preference, so a rounding
 * or direction error here silently misreports every glucose value in the
 * product. It was previously only exercised indirectly, through the import
 * normalisation tests, which never call toMgdl/fromMgdl themselves.
 */
import { describe, expect, it } from 'vitest';
import { classifyGlucose, normalizeTargetRange } from '@/lib/domain/thresholds';
import {
  GLUCOSE_ENTRY_BOUNDS,
  MGDL_PER_MMOLL,
  formatGlucose,
  formatGlucoseWithUnit,
  fromMgdl,
  inferGlucoseUnit,
  isPlausibleGlucose,
  maxOf,
  mgdlToMmoll,
  minOf,
  mmollToMgdl,
  targetToMgdl,
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

describe('large inputs', () => {
  // `Math.max(...values)` throws RangeError past ~120k arguments; one CGM
  // export of a few months exceeds that.
  const big = Array.from({ length: 300_000 }, (_, i) => 70 + (i % 200));

  it('infers the unit of a 300k-reading series without overflowing the stack', () => {
    expect(inferGlucoseUnit(big)).toBe('MGDL');
    expect(inferGlucoseUnit(big.map((v) => v / 18))).toBe('MMOLL');
  });

  it('finds min and max of a 300k array, ignoring non-finite values', () => {
    expect(maxOf(big)).toBe(269);
    expect(minOf(big)).toBe(70);
    expect(maxOf([Number.NaN, 3, Infinity])).toBe(3);
    expect(maxOf([])).toBeNull();
  });
});

describe('equivalent entry bounds across units', () => {
  it('derives the mmol/L bounds from the mg/dL bounds at display precision', () => {
    const mg = GLUCOSE_ENTRY_BOUNDS.MGDL;
    const mmol = GLUCOSE_ENTRY_BOUNDS.MMOLL;
    // Each mmol/L bound is exactly how the mg/dL bound displays.
    expect(formatGlucose(mg.min, 'MMOLL')).toBe(mmol.min.toFixed(1));
    expect(formatGlucose(mg.max, 'MMOLL')).toBe(mmol.max.toFixed(1));
    // ...and converts back to within one mmol/L display step of it.
    expect(Math.abs(toMgdl(mmol.min, 'MMOLL') - mg.min)).toBeLessThan(MGDL_PER_MMOLL * 0.05);
    expect(Math.abs(toMgdl(mmol.max, 'MMOLL') - mg.max)).toBeLessThan(MGDL_PER_MMOLL * 0.05);
  });
});

describe('target range rounding', () => {
  it('snaps a mmol/L target to whole mg/dL', () => {
    expect(targetToMgdl(3.9, 'MMOLL')).toBe(70);
    expect(targetToMgdl(10, 'MMOLL')).toBe(180);
    expect(targetToMgdl(70, 'MGDL')).toBe(70);
  });

  it('classifies a reading equal to an unrounded stored target as in range', () => {
    // 3.9 mmol/L used to be stored as 70.27 mg/dL, making 70 "below range".
    const legacy = { lowMgdl: toMgdl(3.9, 'MMOLL'), highMgdl: toMgdl(10, 'MMOLL') };
    expect(normalizeTargetRange(legacy)).toEqual({ lowMgdl: 70, highMgdl: 180 });
    expect(classifyGlucose(70, legacy).id).toBe('in-range');
    expect(classifyGlucose(69, legacy).id).toBe('low');
  });

  // Readings entered in mmol/L are stored unrounded (10.0 → 180.18 mg/dL).
  // Compared raw against a whole-mg/dL target they fell just outside a bound
  // they equal on screen, so both sides are compared at whole mg/dL.
  it.each([
    { reading: 4.2, low: 4.2, high: 10.0 },
    { reading: 10.0, low: 3.9, high: 10.0 },
    { reading: 9.0, low: 4.0, high: 9.0 },
  ])('classifies a mmol/L reading of $reading equal to a target bound as in range', (c) => {
    const range = {
      lowMgdl: targetToMgdl(c.low, 'MMOLL'),
      highMgdl: targetToMgdl(c.high, 'MMOLL'),
    };
    const unrounded = { lowMgdl: toMgdl(c.low, 'MMOLL'), highMgdl: toMgdl(c.high, 'MMOLL') };
    expect(classifyGlucose(toMgdl(c.reading, 'MMOLL'), range).id).toBe('in-range');
    expect(classifyGlucose(toMgdl(c.reading, 'MMOLL'), unrounded).id).toBe('in-range');
  });

  it.each([
    { reading: 76, low: 76, high: 180 },
    { reading: 180, low: 70, high: 180 },
    { reading: 162, low: 72, high: 162 },
  ])('classifies a mg/dL reading of $reading equal to a target bound as in range', (c) => {
    const range = { lowMgdl: targetToMgdl(c.low, 'MGDL'), highMgdl: targetToMgdl(c.high, 'MGDL') };
    expect(classifyGlucose(toMgdl(c.reading, 'MGDL'), range).id).toBe('in-range');
  });

  it('still classifies the next mmol/L display step past a bound as out of range', () => {
    const range = { lowMgdl: targetToMgdl(4.2, 'MMOLL'), highMgdl: targetToMgdl(10.0, 'MMOLL') };
    expect(classifyGlucose(toMgdl(4.1, 'MMOLL'), range).id).toBe('low');
    expect(classifyGlucose(toMgdl(10.1, 'MMOLL'), range).id).toBe('above-range');
  });
});
