import { EXPORT_RECORD_TYPES, EXPORT_ROW_LIMIT, exportLabel } from '@/lib/services/export-service';
import { ButtonLink, Card, Icon } from '@/components/ui';

/**
 * Server component: it only renders links to the authenticated /api/export
 * route, so it needs no client state or form. The browser's normal
 * navigation-to-download flow handles the file, no JavaScript required.
 */
export function DataExport({ totalRecords }: { totalRecords: number }) {
  return (
    <Card>
      <p className="max-w-prose text-ink-muted">
        Your export contains every reading, meal, activity, sleep, medication, weight, blood
        pressure, mood and note you have logged, plus your profile settings and the history of any
        files you imported. It never includes your password. Each file is generated fresh when you
        download it and is capped at {EXPORT_ROW_LIMIT.toLocaleString()} rows per record type, the
        file tells you if a type was cut off.
      </p>

      <div className="mt-5">
        <h3 className="text-base font-semibold">Everything, as one file</h3>
        <p className="mt-1 text-sm text-ink-muted">
          A single JSON file with all of your data together, including both mg/dL and mmol/L for
          every glucose reading so it is usable outside DiaLog too.
        </p>
        <ButtonLink href="/api/export?format=json" className="mt-3" variant="secondary">
          <Icon name="download" /> Download everything (JSON)
        </ButtonLink>
      </div>

      <div className="mt-6">
        <h3 className="text-base font-semibold">One record type at a time</h3>
        <p className="mt-1 text-sm text-ink-muted">
          A plain spreadsheet-friendly CSV file, one record type per download.
        </p>
        {totalRecords === 0 ? (
          <p className="mt-3 text-sm text-ink-muted">
            There is nothing to export yet. Once you log something, it will be available here.
          </p>
        ) : (
          <ul className="mt-3 grid gap-2 sm:grid-cols-2">
            {EXPORT_RECORD_TYPES.map((type) => (
              <li key={type}>
                <a
                  href={`/api/export?format=csv&type=${type}`}
                  className="dl-target flex min-h-11 items-center gap-2 rounded-[var(--radius-control)] border-2 border-line-strong px-4 py-2.5 text-sm font-semibold hover:border-brand hover:text-brand-ink"
                >
                  <Icon name="download" /> {exportLabel(type)} (CSV)
                </a>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Card>
  );
}
