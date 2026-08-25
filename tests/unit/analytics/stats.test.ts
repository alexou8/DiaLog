import { describe, expect, it } from 'vitest';
import {
  cohensD,
  coefficientOfVariation,
  iqr,
  mad,
  mannKendall,
  mean,
  median,
  modifiedZScores,
  pearson,
  quantile,
  spearman,
  stdDev,
  tDistributionTwoSidedPValue,
  theilSen,
  linearRegression,
  welchTTest,
} from '@/lib/analytics/stats';

describe('mean', () => {
  it('computes the arithmetic mean', () => {
    expect(mean([1, 2, 3, 4])).toBeCloseTo(2.5);
  });
  it('returns null for empty input', () => {
    expect(mean([])).toBeNull();
  });
  it('ignores non-finite values', () => {
    expect(mean([1, 2, NaN, Infinity])).toBeCloseTo(1.5);
  });
});

describe('median / quantile', () => {
  it('handles odd-length arrays', () => {
    expect(median([3, 1, 2])).toBe(2);
  });
  it('averages the two middle values for even-length arrays', () => {
    expect(median([1, 2, 3, 4])).toBe(2.5);
  });
  it('returns null for empty input', () => {
    expect(median([])).toBeNull();
  });
  it('matches known linear-interpolation quantiles (numpy "linear" method)', () => {
    // [1,2,3,4,5,6,7,8,9,10], p=0.25 -> idx=2.25 -> 3 + 0.25*(4-3) = 3.25
    const xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    expect(quantile(xs, 0.25)).toBeCloseTo(3.25);
    expect(quantile(xs, 0.75)).toBeCloseTo(7.75);
    expect(quantile(xs, 0)).toBe(1);
    expect(quantile(xs, 1)).toBe(10);
  });
});

describe('stdDev', () => {
  it('computes sample standard deviation', () => {
    // classic textbook example: [2,4,4,4,5,5,7,9], sample sd = 2.13809...
    const xs = [2, 4, 4, 4, 5, 5, 7, 9];
    expect(stdDev(xs)).toBeCloseTo(2.13809, 4);
  });
  it('returns null for n<2', () => {
    expect(stdDev([5])).toBeNull();
    expect(stdDev([])).toBeNull();
  });
});

describe('coefficientOfVariation', () => {
  it('is sd/mean', () => {
    const xs = [2, 4, 4, 4, 5, 5, 7, 9];
    const cv = coefficientOfVariation(xs);
    expect(cv).toBeCloseTo(2.13809 / 5, 4);
  });
  it('returns null when mean is 0', () => {
    expect(coefficientOfVariation([-1, 1])).toBeNull();
  });
});

describe('iqr', () => {
  it('computes Q3-Q1', () => {
    const xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    expect(iqr(xs)).toBeCloseTo(7.75 - 3.25);
  });
  it('returns null for n<2', () => {
    expect(iqr([1])).toBeNull();
  });
});

describe('mad', () => {
  it('computes scaled median absolute deviation', () => {
    // data: [1,1,2,2,4,6,9]; median=2; abs devs=[1,1,0,0,2,4,7]; median dev=1
    const xs = [1, 1, 2, 2, 4, 6, 9];
    expect(mad(xs, 1)).toBeCloseTo(1);
    expect(mad(xs)).toBeCloseTo(1.4826);
  });
  it('returns null for empty input', () => {
    expect(mad([])).toBeNull();
  });
});

describe('modifiedZScores', () => {
  it('flags a clear outlier against a baseline', () => {
    const baseline = [100, 102, 98, 101, 99, 100, 103, 97, 100, 101];
    const scores = modifiedZScores([...baseline, 250], baseline);
    const last = scores[scores.length - 1];
    expect(last).not.toBeNull();
    expect(Math.abs(last as number)).toBeGreaterThan(3.5);
  });
  it('returns null when the baseline has zero MAD', () => {
    const scores = modifiedZScores([5, 5, 5, 9], [5, 5, 5]);
    expect(scores.every((s) => s === null)).toBe(true);
  });
});

describe('pearson', () => {
  it('is 1 for a perfect positive linear relationship', () => {
    expect(pearson([1, 2, 3, 4], [2, 4, 6, 8])).toBeCloseTo(1);
  });
  it('is -1 for a perfect negative linear relationship', () => {
    expect(pearson([1, 2, 3, 4], [8, 6, 4, 2])).toBeCloseTo(-1);
  });
  it('returns null for n<2', () => {
    expect(pearson([1], [2])).toBeNull();
  });
  it('returns null for zero variance', () => {
    expect(pearson([1, 1, 1], [1, 2, 3])).toBeNull();
  });
});

describe('spearman', () => {
  it('is 1 for a perfectly monotonic (non-linear) relationship', () => {
    expect(spearman([1, 2, 3, 4], [1, 4, 9, 16])).toBeCloseTo(1);
  });
  it('handles ties via average rank', () => {
    const result = spearman([1, 2, 2, 4], [1, 2, 2, 4]);
    expect(result).toBeCloseTo(1);
  });
});

describe('linearRegression', () => {
  it('recovers a known exact line', () => {
    const xs = [1, 2, 3, 4, 5];
    const ys = xs.map((x) => 2 * x + 1);
    const result = linearRegression(xs, ys);
    expect(result?.slope).toBeCloseTo(2);
    expect(result?.intercept).toBeCloseTo(1);
    expect(result?.r2).toBeCloseTo(1);
  });
  it('returns null for n<2', () => {
    expect(linearRegression([1], [2])).toBeNull();
  });
});

describe('theilSen', () => {
  it('recovers a known exact line, robust to one outlier', () => {
    const xs = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    const ys = xs.map((x) => 3 * x + 2);
    ys[4] = 1000; // outlier in the middle
    const result = theilSen(xs, ys);
    expect(result?.slope).toBeCloseTo(3, 0);
  });
  it('returns null for n<2', () => {
    expect(theilSen([1], [2])).toBeNull();
  });
});

describe('mannKendall', () => {
  it('detects a clear increasing trend with a small p-value', () => {
    const ys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const result = mannKendall(ys);
    expect(result?.s).toBeGreaterThan(0);
    expect(result?.tau).toBeCloseTo(1);
    expect(result?.pValue).toBeLessThan(0.05);
  });
  it('finds no significant trend in randomly shuffled flat-ish data around a constant', () => {
    const ys = [5, 5, 5, 5, 5, 5, 5, 5];
    const result = mannKendall(ys);
    expect(result?.s).toBe(0);
    expect(result?.pValue).toBe(1);
  });
  it('returns null for n<2', () => {
    expect(mannKendall([1])).toBeNull();
  });
});

describe('welchTTest / tDistributionTwoSidedPValue', () => {
  it('finds a significant difference between clearly separated groups', () => {
    const a = [10, 11, 12, 9, 10, 11, 12, 10];
    const b = [20, 21, 19, 22, 20, 21, 19, 20];
    const result = welchTTest(a, b);
    expect(result).not.toBeNull();
    expect(result!.pValue).toBeLessThan(0.001);
    expect(result!.t).toBeLessThan(0);
  });
  it('finds no significant difference between identical-distribution samples', () => {
    const a = [10, 12, 11, 10, 13, 9, 11, 10];
    const b = [10, 12, 11, 10, 13, 9, 11, 10];
    const result = welchTTest(a, b);
    expect(result!.t).toBeCloseTo(0);
    expect(result!.pValue).toBeCloseTo(1, 1);
  });
  it('returns null below n=2 in either group', () => {
    expect(welchTTest([1], [1, 2])).toBeNull();
    expect(welchTTest([1, 2], [1])).toBeNull();
  });
  it('t distribution p-value approaches normal for large df (known reference: t=1.96, df=1e6 ~ p=0.05)', () => {
    const p = tDistributionTwoSidedPValue(1.96, 1_000_000);
    expect(p).toBeCloseTo(0.05, 2);
  });
});

describe('cohensD', () => {
  it('computes a known pooled-sd effect size', () => {
    // group1 mean=10 sd~1.5811, group2 mean=20 sd~1.5811 (both have sd of [8,9,10,11,12])
    const a = [8, 9, 10, 11, 12];
    const b = [18, 19, 20, 21, 22];
    const d = cohensD(a, b);
    expect(d).toBeCloseTo(-6.3246, 3);
  });
  it('returns null below n=2', () => {
    expect(cohensD([1], [1, 2])).toBeNull();
  });
});
