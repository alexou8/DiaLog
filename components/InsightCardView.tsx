import type { InsightCard } from '@/lib/analytics/insights';
import { EVIDENCE_LABELS } from '@/lib/domain/evidence';
import { Badge, Card, WhyThis } from '@/components/ui';

const EVIDENCE_TONE = {
  INSUFFICIENT: 'neutral',
  EARLY: 'info',
  EMERGING: 'brand',
  CONSISTENT: 'positive',
} as const;

const SOURCE_LABEL: Record<InsightCard['source'], string> = {
  STATISTICAL: 'Calculated from your records',
  ML: 'Pattern detection on your own history',
  REFERENCE: 'General reference information',
};

/** Format an evidence value for the "why am I seeing this" table. */
function renderValue(value: unknown): string {
  if (value == null) return 'Not available';
  if (typeof value === 'number') return String(Math.round(value * 100) / 100);
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (Array.isArray(value)) return value.map(renderValue).join(', ');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

const HUMAN_KEYS: Record<string, string> = {
  sampleSize: 'Records compared',
  basis: 'What was compared',
  metrics: 'Measured values',
  caveats: 'Things to keep in mind',
  source: 'Where this came from',
  kind: 'Analysis',
};

export function InsightCardView({ insight }: { insight: InsightCard }) {
  const evidence = EVIDENCE_LABELS[insight.evidenceLevel];

  return (
    <Card as="li">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <h3 className="text-lg font-semibold">{insight.title}</h3>
        <Badge tone={EVIDENCE_TONE[insight.evidenceLevel]}>{evidence.label}</Badge>
      </div>

      <p className="mt-2 text-ink-muted">{insight.summary}</p>
      {insight.detail ? <p className="mt-2 text-ink-muted">{insight.detail}</p> : null}

      <WhyThis>
        <dl className="space-y-2">
          <div>
            <dt className="font-semibold text-ink">How much data this is based on</dt>
            <dd>
              {insight.sampleSize} {insight.sampleSize === 1 ? 'record' : 'records'}.{' '}
              {evidence.description}
            </dd>
          </div>
          <div>
            <dt className="font-semibold text-ink">{HUMAN_KEYS.source}</dt>
            <dd>{SOURCE_LABEL[insight.source]}</dd>
          </div>
          {Object.entries(insight.evidence).map(([key, value]) => (
            <div key={key}>
              <dt className="font-semibold text-ink">{HUMAN_KEYS[key] ?? key}</dt>
              <dd>{renderValue(value)}</dd>
            </div>
          ))}
        </dl>
      </WhyThis>
    </Card>
  );
}
