'use client';

import { useActionState, useEffect, useRef, useState } from 'react';
import { useFormStatus } from 'react-dom';
import { analyzeImportAction, commitImportAction, type ImportState } from '@/lib/actions/import';
import { fileTooLargeMessage, MAX_IMPORT_BYTES, MAX_IMPORT_LABEL } from '@/lib/import/limits';
import { Button, Callout, Card, CardHeader, Icon } from '@/components/ui';
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

/**
 * An action result stamped with the file it was produced for and a sequence
 * number. Both exist so the panel can tell, at render time, whether a result
 * still describes what the person is looking at (see `ImportPanel`).
 */
type Stamped = ImportState & { fileKey: string; run: number };

/** Identifies one chosen file well enough to notice that it was swapped. */
function fileKey(file: File): string {
  return `${file.name}:${file.size}:${file.lastModified}`;
}

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

function describeSize(bytes: number): string {
  const formatter = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });
  return bytes >= 1024 * 1024
    ? `${formatter.format(bytes / (1024 * 1024))} MB`
    : `${formatter.format(Math.max(bytes / 1024, 0.1))} KB`;
}

/**
 * Two-stage import. The file is analysed and reported first and saved only on a
 * second, explicit confirmation. The browser keeps the chosen file between the
 * stages, so nothing has to be parked on the server in the meantime.
 *
 * What is on screen is always about the file that is currently chosen. Each
 * result carries the key of the file it was produced for, and a result whose
 * key no longer matches is not shown. Without that, choosing a different file
 * after "Check this file" left the old review on screen with its "Import N
 * records" button, and pressing it committed the new, never-reviewed file.
 */
export function ImportPanel() {
  // The chosen File is held in state rather than read back from the input at
  // confirm time: submitting the analyze form re-renders it and clears the
  // uncontrolled input's FileList, so by the second stage the input is empty.
  // The File object itself stays readable.
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  // Orders results across both stages: a commit result only counts if it is
  // newer than the latest analysis, so checking a file again (even the same
  // one) after importing it does not keep showing "Import complete".
  const runs = useRef(0);

  const [analysisRaw, analyze] = useActionState<Stamped | null, FormData>(
    async (previous, formData) => {
      const run = ++runs.current;
      const chosen = formData.get('file');
      const result = await analyzeImportAction(previous, formData);
      return { ...result, fileKey: chosen instanceof File ? fileKey(chosen) : '', run };
    },
    null,
  );
  const [commitRaw, doCommit] = useActionState<Stamped | null, FormData>(
    async (previous, formData) => {
      const run = ++runs.current;
      const chosen = formData.get('file');
      const result = await commitImportAction(previous, formData);
      return { ...result, fileKey: chosen instanceof File ? fileKey(chosen) : '', run };
    },
    null,
  );

  const currentKey = file ? fileKey(file) : null;
  const analysis = analysisRaw && analysisRaw.fileKey === currentKey ? analysisRaw : null;
  const commit =
    analysis && commitRaw && commitRaw.fileKey === analysis.fileKey && commitRaw.run > analysis.run
      ? commitRaw
      : null;

  const state = commit?.stage === 'committed' ? commit : analysis;
  const summary = state?.summary;
  const reviewed = analysis?.stage === 'reviewed' && commit?.stage !== 'committed';
  // Offering "Import 0 records" when every row is already stored is a button
  // that does nothing; say so instead.
  const readyToImport = reviewed && (analysis?.summary?.rowsImported ?? 0) > 0;
  const nothingNewToImport = reviewed && (analysis?.summary?.rowsImported ?? 0) === 0;

  /**
   * Records the file picked in `input`. Shared by the change handler and the
   * hydration check below, so a file chosen before the page's JavaScript ran
   * is treated exactly like one chosen after.
   */
  function choose(input: HTMLInputElement) {
    const picked = input.files?.[0] ?? null;
    if (picked && picked.size > MAX_IMPORT_BYTES) {
      // Refused here, before upload: the request would otherwise be sent only
      // to be rejected by the server, which looks like a crash. Clearing the
      // input keeps the oversize file from being submitted by the form.
      input.value = '';
      setFile(null);
      setFileError(`${fileTooLargeMessage()} The file you chose is ${describeSize(picked.size)}.`);
      return;
    }
    setFileError(null);
    setFile(picked);
  }

  function onFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    choose(event.target);
  }

  // A file picked while the page was still loading fires its `change` event
  // before React is listening, so `file` stays empty even though the input
  // holds one. Read the input once on hydration so "Selected: …" and the
  // stale-result check below both see it.
  const inputRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (inputRef.current?.files?.length) choose(inputRef.current);
    // Mount only: afterwards every pick arrives through onFileChange.
  }, []);

  /**
   * "Check this file". The native input is empty again after a first check
   * (see above), so a second press falls back to the File held in state rather
   * than failing with the browser's "select a file" message.
   */
  function checkFile(formData: FormData) {
    const submitted = formData.get('file');
    const picked = submitted instanceof File && submitted.size > 0 ? submitted : file;
    if (!picked) {
      setFileError('Please choose a file to check.');
      return;
    }
    if (picked.size > MAX_IMPORT_BYTES) {
      setFileError(`${fileTooLargeMessage()} The file you chose is ${describeSize(picked.size)}.`);
      return;
    }
    setFileError(null);
    // The checked file becomes the chosen one. Results are shown only for the
    // chosen file (see `fileKey`), so without this a file the change handler
    // never saw would be analysed and its result then hidden.
    if (picked !== file) setFile(picked);
    formData.set('file', picked);
    analyze(formData);
  }

  /** Carry the already-chosen file into the confirmation submission. */
  function confirmImport(formData: FormData) {
    // Refuse to commit anything but the file that was reviewed.
    if (!file || !analysis || fileKey(file) !== analysis.fileKey) return;
    formData.set('file', file);
    doCommit(formData);
  }

  return (
    <div className="space-y-6">
      <Card>
        {/* noValidate: the browser's own "select a file" bubble is neither
            announced consistently nor available once React has cleared the
            input after a check. The panel validates and reports for itself. */}
        <form action={checkFile} noValidate>
          <FormStatus
            status={
              analysis && !analysis.ok && analysis.message
                ? { ok: false, message: analysis.message }
                : null
            }
          />
          <div className="mb-5">
            <label htmlFor="import-file" className="mb-1.5 block text-base font-semibold">
              Choose a file <span className="font-normal text-ink-muted">(required)</span>
            </label>
            <p id="import-file-hint" className="mb-2 text-sm text-ink-muted">
              CSV, Excel, JSON or XML, up to {MAX_IMPORT_LABEL}. Nothing is saved until you confirm.
            </p>
            <input
              ref={inputRef}
              id="import-file"
              name="file"
              type="file"
              required
              accept=".csv,.tsv,.txt,.json,.xml,.xlsx,.xls"
              aria-describedby={
                fileError ? 'import-file-hint import-file-error' : 'import-file-hint'
              }
              aria-invalid={fileError ? true : undefined}
              onChange={onFileChange}
              className="w-full rounded-[var(--radius-card)] border-2 border-dashed border-line-strong bg-surface-sunken p-4 text-base file:mr-4 file:rounded-[var(--radius-control)] file:border-0 file:bg-brand file:px-4 file:py-2 file:font-semibold file:text-primary-foreground"
            />
            {/* Always rendered, so a screen reader announces the text that
                appears in it; a live region inserted together with its text is
                often not announced at all. */}
            <p className="mt-2 min-h-5 text-sm" aria-live="polite">
              {file ? (
                <>
                  Selected: <strong>{file.name}</strong> ({describeSize(file.size)})
                </>
              ) : null}
            </p>
            {fileError ? (
              <p
                id="import-file-error"
                role="alert"
                className="mt-2 flex items-start gap-2 rounded-[var(--radius-control)] border border-critical/40 bg-critical-soft p-3 text-sm font-medium text-critical"
              >
                <Icon name="caution" className="mt-0.5 shrink-0" />
                {fileError}
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
              title={
                commit?.stage === 'committed' ? 'Import complete' : 'Here is what DiaLog found'
              }
              description={`Read as: ${state.connectorName}`}
            />

            <ul className="space-y-2">
              <li className="flex items-baseline justify-between gap-3 border-b border-line pb-2">
                <span>Rows in the file</span>
                <strong className="tabular-nums">{summary.rowsTotal}</strong>
              </li>
              <li className="flex items-baseline justify-between gap-3 border-b border-line pb-2">
                <span>
                  <Icon name="ok" className="shrink-0" />
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
                  <Icon name="caution" className="shrink-0" /> Could not be read
                </span>
                <strong className="tabular-nums">{summary.rowsRejected}</strong>
              </li>
            </ul>

            {summary.warnings.length > 0 ? (
              <div className="mt-4">
                <Callout tone="notice" icon="info" title="Assumptions DiaLog had to make">
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
                      {item.when}: {item.detail}
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

            {nothingNewToImport ? (
              <div className="mt-6">
                <Callout tone="info" icon="info" title="Nothing new to add">
                  Every record in this file is already in DiaLog, so there is nothing to import.
                  Your existing records have been left exactly as they were.
                </Callout>
              </div>
            ) : null}

            {commit?.stage === 'committed' ? (
              <div className="mt-6">
                <Callout tone="positive" icon="ok" title="Saved">
                  Your readings are now in DiaLog. If this was the wrong file, you can undo the
                  whole import below.
                </Callout>
              </div>
            ) : null}

            <FormStatus
              status={
                commit && !commit.ok && commit.message
                  ? { ok: false, message: commit.message }
                  : null
              }
            />
          </Card>
        ) : null}
      </div>
    </div>
  );
}
