import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { prisma } from '@/lib/db/prisma';
import { CONNECTORS } from '@/lib/import/connectors/registry';
import { undoImportAction } from '@/lib/actions/import';
import { Button, Card, CardHeader, EmptyState, PageHeader, WhyThis } from '@/components/ui';
import { ImportPanel } from './ImportPanel';

export const metadata: Metadata = { title: 'Import data' };
export const dynamic = 'force-dynamic';

export default async function ImportPage() {
  const user = await requireOnboardedUser();
  const batches = await prisma.importBatch.findMany({
    where: { userId: user.id },
    orderBy: { createdAt: 'desc' },
    take: 10,
    select: {
      id: true,
      filename: true,
      connectorName: true,
      createdAt: true,
      rowsImported: true,
      rowsDuplicate: true,
      rowsRejected: true,
      status: true,
    },
  });

  const dateFmt = new Intl.DateTimeFormat(user.profile.locale, {
    timeZone: user.profile.timezone,
    dateStyle: 'medium',
    timeStyle: 'short',
  });

  return (
    <div className="space-y-8">
      <PageHeader
        title="Import your data"
        description="Bring in the export file from your meter's software, a spreadsheet you keep, or your phone's health app. DiaLog shows you exactly what it found before anything is saved."
      />

      <ImportPanel />

      <section aria-labelledby="formats">
        <Card>
          <CardHeader
            id="formats"
            title="What DiaLog can read"
            description="These are the formats DiaLog genuinely understands today. Where a manufacturer does not offer a way to export or connect, that is stated plainly rather than hidden behind a button that would not work."
          />
          <ul className="space-y-4">
            {CONNECTORS.map((connector) => (
              <li key={connector.id} className="border-b border-line pb-4 last:border-0 last:pb-0">
                <h3 className="font-semibold">{connector.name}</h3>
                <p className="mt-1 text-sm text-ink-muted">{connector.description}</p>
                {connector.howToExport.length > 0 ? (
                  <details className="mt-2">
                    <summary className="cursor-pointer text-sm font-semibold text-brand-ink">
                      How to get this file
                    </summary>
                    <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm text-ink-muted">
                      {connector.howToExport.map((step, index) => (
                        <li key={index}>{step}</li>
                      ))}
                    </ol>
                  </details>
                ) : null}
                <p className="mt-2 text-xs text-ink-muted">
                  Accepts: {connector.acceptedExtensions.join(', ')}
                </p>
              </li>
            ))}
          </ul>

          <WhyThis label="Why can't DiaLog connect to my meter directly?">
            <p>
              For most home glucose meters there is no public interface a website is allowed to use.
              Some manufacturers offer no developer access at all; some expose data only through their
              own cloud app; and browser-to-device communication over Bluetooth is not supported on
              iPhones and is not implemented by most meters. Rather than shipping a
              &ldquo;Connect&rdquo; button that could not work, DiaLog invests in reading the export
              files that manufacturers do provide. The full research notes, with sources, are in the
              project documentation under <code>docs/DEVICE_INTEGRATIONS.md</code>.
            </p>
          </WhyThis>
        </Card>
      </section>

      <section aria-labelledby="history">
        <Card>
          <CardHeader
            id="history"
            title="Your imports"
            description="Every import is recorded, so you can always see where a reading came from — and undo a whole import if it went wrong."
          />
          {batches.length === 0 ? (
            <EmptyState title="No imports yet" icon="📥">
              <p>When you import a file, it will be listed here with what it added.</p>
            </EmptyState>
          ) : (
            <ul className="space-y-3">
              {batches.map((batch) => (
                <li key={batch.id} className="rounded-xl border border-line p-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <p className="font-semibold">{batch.filename}</p>
                      <p className="text-sm text-ink-muted">
                        {batch.connectorName} · {dateFmt.format(batch.createdAt)}
                      </p>
                      <p className="mt-1 text-sm">
                        {batch.rowsImported} added
                        {batch.rowsDuplicate > 0 ? `, ${batch.rowsDuplicate} skipped as duplicates` : ''}
                        {batch.rowsRejected > 0 ? `, ${batch.rowsRejected} could not be read` : ''}
                      </p>
                    </div>
                    <details>
                      <summary className="dl-target cursor-pointer rounded-xl border-2 border-critical px-4 py-2 text-sm font-semibold text-critical">
                        Undo this import
                      </summary>
                      <form action={undoImportAction} className="mt-3 max-w-xs rounded-lg border border-line p-3">
                        <input type="hidden" name="batchId" value={batch.id} />
                        <p className="mb-3 text-sm">
                          This permanently removes the {batch.rowsImported} records this file added.
                          Records you entered by hand are not affected.
                        </p>
                        <Button type="submit" variant="danger" className="w-full text-sm">
                          Yes, remove these records
                        </Button>
                      </form>
                    </details>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </section>
    </div>
  );
}
