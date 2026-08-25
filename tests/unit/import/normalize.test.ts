import { describe, expect, it } from 'vitest';
import {
  detectFileGlucoseUnit,
  mapGlucoseContext,
  normalizeGlucoseValue,
} from '@/lib/import/normalize';

describe('normalizeGlucoseValue', () => {
  it('converts a plausible mmol/L value to canonical mg/dL', () => {
    const r = normalizeGlucoseValue(7, 'MMOLL');
    expect(r.valueMgdl).toBeCloseTo(126.13, 1);
    expect(r.issue).toBeUndefined();
  });

  it('passes through a plausible mg/dL value unchanged', () => {
    const r = normalizeGlucoseValue(120, 'MGDL');
    expect(r.valueMgdl).toBe(120);
  });

  it('rejects an out-of-range mg/dL value', () => {
    const r = normalizeGlucoseValue(1500, 'MGDL');
    expect(r.valueMgdl).toBeNull();
    expect(r.issue).toBe('OUT_OF_RANGE');
  });

  it('rejects an out-of-range mmol/L value', () => {
    const r = normalizeGlucoseValue(0.2, 'MMOLL');
    expect(r.valueMgdl).toBeNull();
    expect(r.issue).toBe('OUT_OF_RANGE');
  });

  it('rejects a non-numeric value', () => {
    const r = normalizeGlucoseValue(NaN, 'MGDL');
    expect(r.issue).toBe('INVALID_VALUE');
  });

  it('rejects a null/undefined value', () => {
    expect(normalizeGlucoseValue(null, 'MGDL').issue).toBe('INVALID_VALUE');
    expect(normalizeGlucoseValue(undefined, 'MGDL').issue).toBe('INVALID_VALUE');
  });

  it('flags unknown unit when unit cannot be determined', () => {
    const r = normalizeGlucoseValue(120, null);
    expect(r.issue).toBe('UNKNOWN_UNIT');
  });
});

describe('mapGlucoseContext', () => {
  it('maps fasting phrasing', () => {
    expect(mapGlucoseContext('Fasting')).toBe('FASTING');
    expect(mapGlucoseContext('fasted overnight')).toBe('FASTING');
  });

  it('maps before-meal phrasing', () => {
    expect(mapGlucoseContext('before breakfast')).toBe('BEFORE_MEAL');
    expect(mapGlucoseContext('pre-meal')).toBe('BEFORE_MEAL');
  });

  it('maps after-meal / post-meal phrasing', () => {
    expect(mapGlucoseContext('after dinner')).toBe('AFTER_MEAL');
    expect(mapGlucoseContext('post meal')).toBe('AFTER_MEAL');
  });

  it('maps bedtime phrasing', () => {
    expect(mapGlucoseContext('bedtime')).toBe('BEDTIME');
    expect(mapGlucoseContext('before bed')).toBe('BEDTIME');
  });

  it('maps random/any-time phrasing', () => {
    expect(mapGlucoseContext('random')).toBe('RANDOM');
  });

  it('falls back to UNKNOWN for unrecognised or empty text', () => {
    expect(mapGlucoseContext('')).toBe('UNKNOWN');
    expect(mapGlucoseContext(undefined)).toBe('UNKNOWN');
    expect(mapGlucoseContext('gibberish text')).toBe('UNKNOWN');
  });
});

describe('detectFileGlucoseUnit', () => {
  it('prefers an explicit unit over inference', () => {
    expect(detectFileGlucoseUnit([7, 8, 9], 'MMOLL')).toBe('MMOLL');
  });

  it('infers mg/dL from large values', () => {
    expect(detectFileGlucoseUnit([110, 140, 180])).toBe('MGDL');
  });

  it('infers mmol/L from small values', () => {
    expect(detectFileGlucoseUnit([5.5, 7.2, 9.1])).toBe('MMOLL');
  });
});
