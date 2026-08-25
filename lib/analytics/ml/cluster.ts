/**
 * k-means over per-day feature vectors, to surface recurring "day
 * patterns" (e.g. "higher glucose, little logged activity" days) rather
 * than making the user read a wall of daily numbers.
 *
 * Determinism: tests need clustering that produces the same answer every
 * run. We seed a small PRNG (mulberry32 — a standard, fast, well-tested
 * 32-bit generator) and use it for both k-means++ seeding and the
 * iteration, so `cluster(..., { seed: 1 })` is exactly reproducible.
 */
import { dayKeyInZone } from '@/lib/domain/time';
import { coefficientOfVariation, mean, stdDev, sum } from '../stats';
import type { AnalyticsInput } from '../types';

export interface DayFeatureVector {
  dayKey: string;
  meanGlucoseMgdl: number;
  glucoseCv: number;
  carbsG: number;
  activityMinutes: number;
  sleepHours: number;
}

const FEATURE_KEYS = ['meanGlucoseMgdl', 'glucoseCv', 'carbsG', 'activityMinutes', 'sleepHours'] as const;
type FeatureKey = (typeof FEATURE_KEYS)[number];

const FEATURE_DESCRIPTIONS: Record<FeatureKey, { high: string; low: string }> = {
  meanGlucoseMgdl: { high: 'higher average glucose', low: 'lower average glucose' },
  glucoseCv: { high: 'more variable glucose', low: 'more stable glucose' },
  carbsG: { high: 'more carbs logged', low: 'fewer carbs logged' },
  activityMinutes: { high: 'more logged activity', low: 'little logged activity' },
  sleepHours: { high: 'more sleep logged', low: 'less sleep logged' },
};

/** Builds one feature vector per local calendar day that has at least one glucose reading. */
export function buildDayFeatureVectors(input: AnalyticsInput): DayFeatureVector[] {
  const glucoseByDay = new Map<string, number[]>();
  for (const g of input.glucose) {
    const key = dayKeyInZone(g.takenAt, input.timezone);
    const arr = glucoseByDay.get(key) ?? [];
    arr.push(g.valueMgdl);
    glucoseByDay.set(key, arr);
  }

  const carbsByDay = new Map<string, number>();
  for (const m of input.meals) {
    if (m.carbsG === null) continue;
    const key = dayKeyInZone(m.takenAt, input.timezone);
    carbsByDay.set(key, (carbsByDay.get(key) ?? 0) + m.carbsG);
  }

  const activityByDay = new Map<string, number>();
  for (const e of input.exercise) {
    const key = dayKeyInZone(e.takenAt, input.timezone);
    activityByDay.set(key, (activityByDay.get(key) ?? 0) + e.durationMin);
  }

  const sleepByDay = new Map<string, number>();
  for (const s of input.sleep) {
    const key = dayKeyInZone(s.endedAt, input.timezone);
    sleepByDay.set(key, (sleepByDay.get(key) ?? 0) + s.durationMin / 60);
  }

  const vectors: DayFeatureVector[] = [];
  for (const [dayKey, values] of glucoseByDay) {
    const avg = mean(values);
    if (avg === null) continue;
    vectors.push({
      dayKey,
      meanGlucoseMgdl: avg,
      glucoseCv: coefficientOfVariation(values) ?? 0,
      carbsG: carbsByDay.get(dayKey) ?? 0,
      activityMinutes: activityByDay.get(dayKey) ?? 0,
      sleepHours: sleepByDay.get(dayKey) ?? 0,
    });
  }
  vectors.sort((a, b) => (a.dayKey < b.dayKey ? -1 : a.dayKey > b.dayKey ? 1 : 0));
  return vectors;
}

/** mulberry32: a small, fast, deterministic 32-bit PRNG. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function squaredDistance(a: readonly number[], b: readonly number[]): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) {
    const av = a[i] ?? 0;
    const bv = b[i] ?? 0;
    d += (av - bv) ** 2;
  }
  return d;
}

function kmeansPlusPlusInit(points: readonly number[][], k: number, rng: () => number): number[][] {
  const centers: number[][] = [];
  const first = points[Math.floor(rng() * points.length)];
  if (first) centers.push(first);
  while (centers.length < k) {
    const distances = points.map((p) => Math.min(...centers.map((c) => squaredDistance(p, c))));
    const total = sum(distances);
    if (total === 0) {
      // All remaining points coincide with a chosen center; fall back to picking arbitrarily.
      const remaining = points.find((p) => !centers.some((c) => squaredDistance(p, c) === 0));
      centers.push(remaining ?? points[0] ?? []);
      continue;
    }
    let r = rng() * total;
    let chosen = points[points.length - 1] ?? [];
    for (let i = 0; i < points.length; i++) {
      const d = distances[i] ?? 0;
      r -= d;
      if (r <= 0) {
        chosen = points[i] ?? chosen;
        break;
      }
    }
    centers.push(chosen);
  }
  return centers;
}

export interface KMeansResult {
  assignments: number[];
  centroids: number[][];
  /** Sum of squared distances of each point to its assigned centroid. */
  inertia: number;
}

export function kmeans(points: readonly number[][], k: number, options: { seed?: number; maxIterations?: number } = {}): KMeansResult | null {
  if (points.length === 0 || k < 1) return null;
  const effectiveK = Math.min(k, points.length);
  const rng = mulberry32(options.seed ?? 42);
  const maxIterations = options.maxIterations ?? 100;

  let centroids = kmeansPlusPlusInit(points, effectiveK, rng);
  let assignments = new Array<number>(points.length).fill(0);

  for (let iter = 0; iter < maxIterations; iter++) {
    let changed = false;
    const newAssignments = points.map((p) => {
      let best = 0;
      let bestDist = Infinity;
      for (let c = 0; c < centroids.length; c++) {
        const centroid = centroids[c];
        if (!centroid) continue;
        const d = squaredDistance(p, centroid);
        if (d < bestDist) {
          bestDist = d;
          best = c;
        }
      }
      return best;
    });
    for (let i = 0; i < newAssignments.length; i++) {
      if (newAssignments[i] !== assignments[i]) changed = true;
    }
    assignments = newAssignments;

    const dims = points[0]?.length ?? 0;
    const sums: number[][] = Array.from({ length: effectiveK }, () => new Array(dims).fill(0));
    const counts = new Array(effectiveK).fill(0);
    for (let i = 0; i < points.length; i++) {
      const c = assignments[i] ?? 0;
      const p = points[i];
      if (!p) continue;
      counts[c]++;
      for (let d = 0; d < dims; d++) sums[c]![d] = (sums[c]![d] ?? 0) + (p[d] ?? 0);
    }
    centroids = sums.map((s, c) => ((counts[c] ?? 0) > 0 ? s.map((v) => v / (counts[c] ?? 1)) : centroids[c] ?? s));

    if (!changed && iter > 0) break;
  }

  let inertia = 0;
  for (let i = 0; i < points.length; i++) {
    const c = assignments[i] ?? 0;
    const centroid = centroids[c];
    const p = points[i];
    if (centroid && p) inertia += squaredDistance(p, centroid);
  }

  return { assignments, centroids, inertia };
}

export interface DayCluster {
  clusterId: number;
  dayKeys: string[];
  size: number;
  /** Centroid in original (unstandardised) feature units. */
  centroid: Record<FeatureKey, number>;
  /** Plain-language description derived from the centroid relative to the whole-population mean. */
  label: string;
}

/**
 * Clusters days into `k` groups using standardised features (each feature
 * z-scored across days before clustering, so that e.g. glucose in mg/dL
 * doesn't dominate distance purely because its numbers are larger than
 * sleep hours). Returns null when there isn't enough data to cluster
 * meaningfully (fewer days than 2 * k).
 */
export function clusterDayPatterns(input: AnalyticsInput, k = 3, seed = 42): DayCluster[] | null {
  const vectors = buildDayFeatureVectors(input);
  if (vectors.length < Math.max(4, k * 2)) return null;

  const means: Record<FeatureKey, number> = {} as Record<FeatureKey, number>;
  const sds: Record<FeatureKey, number> = {} as Record<FeatureKey, number>;
  for (const key of FEATURE_KEYS) {
    const values = vectors.map((v) => v[key]);
    means[key] = mean(values) ?? 0;
    sds[key] = stdDev(values) || 1; // avoid divide-by-zero when a feature is constant
  }

  const standardized = vectors.map((v) => FEATURE_KEYS.map((key) => (v[key] - means[key]) / sds[key]));
  const result = kmeans(standardized, k, { seed });
  if (!result) return null;

  const clusters: DayCluster[] = [];
  for (let c = 0; c < result.centroids.length; c++) {
    const dayKeys = vectors.filter((_, i) => result.assignments[i] === c).map((v) => v.dayKey);
    if (dayKeys.length === 0) continue;
    const standardizedCentroid = result.centroids[c] ?? [];
    const centroid: Record<FeatureKey, number> = {} as Record<FeatureKey, number>;
    FEATURE_KEYS.forEach((key, idx) => {
      centroid[key] = (standardizedCentroid[idx] ?? 0) * sds[key] + means[key];
    });

    // Label from the two most salient (largest |z|) standardised dimensions.
    const salience = FEATURE_KEYS.map((key, idx) => ({ key, z: standardizedCentroid[idx] ?? 0 }))
      .filter((s) => Math.abs(s.z) > 0.3)
      .sort((a, b) => Math.abs(b.z) - Math.abs(a.z))
      .slice(0, 2);
    const label =
      salience.length > 0
        ? salience.map((s) => (s.z > 0 ? FEATURE_DESCRIPTIONS[s.key].high : FEATURE_DESCRIPTIONS[s.key].low)).join(', ')
        : 'a typical day for you';

    clusters.push({ clusterId: c, dayKeys, size: dayKeys.length, centroid, label });
  }

  clusters.sort((a, b) => b.size - a.size);
  return clusters;
}
