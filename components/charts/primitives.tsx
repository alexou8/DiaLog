/**
 * Chart primitives.
 *
 * Charts are hand-built SVG rather than a charting library so that we control
 * the accessibility contract completely: every chart is `role="img"` with a
 * descriptive label, and every chart ships a real <table> alternative that
 * screen-reader and keyboard users can read. Charts never encode meaning in
 * colour alone — the target band is drawn as a labelled region and out-of-range
 * points use a different shape as well as a different colour.
 */
import type { ReactNode } from 'react';

export interface Scale {
  (value: number): number;
}

export function linearScale(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (value: number) => r0 + ((value - d0) / span) * (r1 - r0);
}

/** Rounded "nice" tick values for an axis. */
export function ticks(min: number, max: number, count = 4): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return [min];
  const raw = (max - min) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) ?? magnitude * 10;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step)
    out.push(Number(v.toFixed(6)));
  return out;
}

/**
 * Wrapper providing the accessible contract shared by every chart: a caption,
 * a text summary announced to assistive technology, and a collapsible data
 * table containing the same numbers the chart draws.
 */
export function ChartFrame({
  title,
  summary,
  children,
  table,
  footer,
}: {
  title: string;
  /** One or two sentences describing what the chart shows, in plain language. */
  summary: string;
  children: ReactNode;
  table: { caption: string; head: string[]; rows: (string | number)[][] };
  footer?: ReactNode;
}) {
  return (
    <figure className="m-0">
      <figcaption className="sr-only">{title}</figcaption>
      <div role="img" aria-label={`${title}. ${summary}`} className="w-full">
        {children}
      </div>
      <p className="mt-2 text-sm text-ink-muted">{summary}</p>
      {footer}
      <details className="mt-2">
        <summary className="cursor-pointer text-sm font-semibold text-brand-ink">
          View this chart as a table
        </summary>
        <div className="mt-2 max-h-80 overflow-auto rounded-lg border border-line">
          <table className="w-full border-collapse text-sm">
            <caption className="sr-only">{table.caption}</caption>
            <thead className="sticky top-0 bg-surface-sunken">
              <tr>
                {table.head.map((h) => (
                  <th
                    key={h}
                    scope="col"
                    className="border-b border-line px-3 py-2 text-left font-semibold"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.rows.map((row, i) => (
                <tr key={i} className="odd:bg-surface-sunken/50">
                  {row.map((cell, j) => (
                    <td key={j} className="border-b border-line px-3 py-2 tabular-nums">
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
