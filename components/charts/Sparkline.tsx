import { linearScale } from './primitives';

/**
 * Tiny trend indicator. Purely decorative reinforcement of a number that is
 * always stated in text next to it, so it is hidden from assistive technology.
 */
export function Sparkline({ values, width = 120, height = 34 }: { values: number[]; width?: number; height?: number }) {
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const x = linearScale([0, values.length - 1], [1, width - 1]);
  const y = linearScale([min === max ? min - 1 : min, max === min ? max + 1 : max], [height - 2, 2]);
  const d = values.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} aria-hidden="true" focusable="false">
      <path d={d} fill="none" stroke="var(--color-brand)" strokeWidth={2} strokeLinecap="round" />
    </svg>
  );
}
