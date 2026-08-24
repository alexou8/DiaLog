'use client';

import { useActionState, useRef, useState } from 'react';
import { useFormStatus } from 'react-dom';
import { analyzeImportAction, commitImportAction, type ImportState } from '@/lib/actions/import';
import { Button, Callout, Card, CardHeader } from '@/components/ui';
import { FormStatus } from '@/components/ui/form';

const ISSUE_EXPLANATIONS: Record<string, string> = {
  MISSING_TIMESTAMP: 'The row had no date or time, so DiaLog could not place it on your timeline.',
  INVALID_TIMESTAMP: 'The date could not be understood.',
  FUTURE_TIMESTAMP: 'The date was in the future, which usually means the device clock was wrong.',
  MISSING_VALUE: 'The row had no reading in it.',
  INVALID_VALUE: 'The value was not a number.',
  OUT_OF_RANGE: 'The value was outside the range a real reading can be.',
  UNKNOWN_UNIT: 'DiaLog could not tell whether the value was mmol/L or mg/dL.',
  UNSUPPORTED_ROW: 'This kind of row is not something DiaLog stores.',
  PARSE_ERROR: 'The row could not be read at all.',
};

function AnalyzeSubmit() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" disabled={pending} className="w-full">
      {pending ? 'Reading your file…' : 'Check this file'}
    </Button>
  );
}

function CommitSubmit({ count }: { count: number }) {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" disabled={pending}>
      {pending ? 'Saving…' : `Import ${count} ${count === 1 ? 'record' : 'records'}`}
    </Button>
  );
}

/**
 * Two-stage import. The file is analysed and reported first and saved only on a
 * second, explicit confirmation. The browser keeps the chosen file between the
 * stages, so nothing has to be parked on the server in the meantime.
 */
export function ImportPanel() {
  const [analysis, analyze] = useActionState<ImportState | null, FormData>(analyzeImportAction, null);
  const [commit, doCommit] = useActionState<ImportState | null, FormData>(commitImportAction, null);
  const [filename, setFilename] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const state = commit?.stage === 'committed' ? commit : analysis;
  const summary = state?.summary;
  const readyToImport = analysis?.stage === 'reviewed' && commit?.stage !== 'committed';

  /** Carry the already-chosen file into the confirmation submission. */
  function confirmImport(formData: FormData) {
    const file = fileInput.current?.files?.[0];
    if (file) formData.set('file', file);
    doCommit(formData);
  }

  return (
    <div className="space-y-6">
      <Card>
        <form action={analyze}>
          <FormStatus
            status={analysis && !analysis.ok && analysis.message ? { ok: false, message: analysis.message } : null}
          />
          <div className="mb-5">
            <label htmlFor="import-file" className="mb-1.5 block text-base font-semibold">
              Choose a file <span className="font-normal text-ink-muted">(required)</span>
            </label>
            <p id="import-file-hint" className="mb-2 text-sm text-ink-muted">
              CSV, Excel, JSON or XML, up to 100 MB. Nothing is saved until you confirm.
            </p>
            <input
              ref={fileInput}
              id="import-file"
              name="file"
              type="file"
              required
              accept=".csv,.tsv,.txt,.json,.xml,.xlsx,.xls"
              aria-describedby="import-file-hint"
              onChange={(event) => setFilename(event.target.files?.[0]?.name ?? null)}
              className="w-full rounded-xl border-2 border-dashed border-line-strong bg-surface-sunken p-4 text-base file:mr-4 file:rounded-lg file:border-0 file:bg-brand file:px-4 file:py-2 file:font-semibold file:text-white"
            />
            {filename ? (
              <p className="mt-2 text-sm" aria-live="polite">
                Selected: <strong>{filename}</strong>
              </p>
            ) : null}
          </div>
          <AnalyzeSubmit />
        </form>
      </Card>

      <div aria-live="polite" className="space-y-6">
        {state?.ok && summary ? (
          <Card>
            <CardHeader
              title={commit?.stage === 'committed' ? 'Import complete' : 'Here is what DiaLog found'}
              description={`Read as: ${state.connectorName}`}
            />

            <ul className="space-y-2">
              <li className="flex items-baseline justify-between gap-3 border-b border-line pb-2">
                <span>Rows in the file</span>
                <strong className="tabular-nums">{summary.rowsTotal}</strong>
              </li>
              <li className="flex items-baseline justify-between gap-3 border-b border-line pb-2">
                <span>
                  <span aria-hidden="true">✓ </span>
                  {commit?.stage === 'committed' ? 'Added to your history' : 'Ready to add'}
                </span>
                <strong className="tabular-nums">
                  {commit?.stage === 'committed' ? (commit.imported ?? 0) : summary.rowsImported}
                </strong>
              </li>
              <li className="flex items-baseline justify-between gap-3 border-b border-line pb-2">
                <span>
                  <span aria-hidden="true">⊘ </span>Already in DiaLog, so skipped
                </span>
                <strong className="tabular-nums">{summary.rowsDuplicate}</strong>
              </li>
              <li className="flex items-baseline justify-between gap-3">
                <span>
                  <span aria-hidden="true">⚠ </span>Could not be read
                </span>
                <strong className="tabular-nums">{summary.rowsRejected}</strong>
              </li>
            </ul>

            {summary.warnings.length > 0 ? (
              <div className="mt-4">
                <Callout tone="notice" icon="ⓘ" title="Assumptions DiaLog had to make">
                  <ul className="list-disc space-y-1 pl-5">
                    {summary.warnings.map((warning, index) => (
                      <li key={index}>{warning}</li>
                    ))}
                  </ul>
                </Callout>
              </div>
            ) : null}

            {summary.issueGroups.length > 0 ? (
              <details className="mt-4">
                <summary className="cursor-pointer font-semibold text-brand-ink">
                  See the rows that were not imported
                </summary>
                <div className="mt-3 space-y-3">
                  {summary.issueGroups.map((group) => (
                    <div key={group.code} className="rounded-lg border border-line p-3">
                      <p className="font-medium">
                        {group.count} {group.count === 1 ? 'row' : 'rows'}:{' '}
                        {ISSUE_EXPLANATIONS[group.code] ?? group.code}
                      </p>
                      <ul className="mt-2 space-y-1 text-sm text-ink-muted">
                        {group.examples.slice(0, 5).map((issue, index) => (
                          <li key={index}>
                            Row {issue.rowNumber}: {issue.message}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </details>
            ) : null}

            {state.preview && state.preview.length > 0 && commit?.stage !== 'committed' ? (
              <div className="mt-4">
                <h3 className="font-semibold">A few of the records that would be added</h3>
                <ul className="mt-2 space-y-1 text-sm text-ink-muted">
                  {state.preview.map((item, index) => (
                    <li key={index}>
                      {item.when} — {item.detail}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}

            {readyToImport ? (
              <form action={confirmImport} className="mt-6 flex flex-wrap items-center gap-3">
                <CommitSubmit count={summary.rowsImported} />
                <p className="text-sm text-ink-muted">
                  Nothing has been saved yet. Duplicates will be skipped automatically.
                </p>
              </form>
            ) : null}

            {commit?.stage === 'committed' ? (
              <div className="mt-6">
                <Callout tone="positive" icon="✓" title="Saved">
                  Your readings are now in DiaLog. If this was the wrong file, you can undo the whole
                  import below.
                </Callout>
              </div>
            ) : null}

            <FormStatus status={commit && !commit.ok && commit.message ? { ok: false, message: commit.message } : null} />
          </Card>
        ) : null}
      </div>
    </div>
  );
}
