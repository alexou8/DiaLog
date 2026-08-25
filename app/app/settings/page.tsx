import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { getProvider } from '@/lib/ai/provider';
import { recordCounts } from '@/lib/db/health-records';
import { PageHeader } from '@/components/ui';
import { ThemeToggle } from '@/components/ThemeToggle';
import { UnitsForm } from './UnitsForm';
import { RegionForm } from './RegionForm';
import { DisplayForm } from './DisplayForm';
import { AssistantForm } from './AssistantForm';
import { DataExport } from './DataExport';
import {
  ChangePasswordForm,
  SignOutEverywhereForm,
  DeleteAllRecordsForm,
  DeleteAccountForm,
} from './AccountForms';

export const metadata: Metadata = { title: 'Settings' };
export const dynamic = 'force-dynamic';

export default async function SettingsPage() {
  const user = await requireOnboardedUser();
  const { profile } = user;
  const counts = await recordCounts(user.id);
  const totalRecords = Object.values(counts).reduce((a, b) => a + b, 0);

  const provider = getProvider();

  return (
    <div className="space-y-10 pb-16">
      <PageHeader
        title="Settings"
        description="Control how DiaLog shows your numbers, who can see them, and what happens to your data."
      />

      {/* --------------------------------------------------- Units & targets */}
      <section aria-labelledby="units-heading" className="space-y-4">
        <h2 id="units-heading" className="text-xl font-semibold sm:text-2xl">
          Units and targets
        </h2>
        <UnitsForm profile={profile} />
      </section>

      {/* --------------------------------------------------- Language & region */}
      <section aria-labelledby="region-heading" className="space-y-4">
        <h2 id="region-heading" className="text-xl font-semibold sm:text-2xl">
          Language and region
        </h2>
        <RegionForm profile={profile} />
      </section>

      {/* ------------------------------------------------ Display & accessibility */}
      <section aria-labelledby="display-heading" className="space-y-4">
        <h2 id="display-heading" className="text-xl font-semibold sm:text-2xl">
          Display and accessibility
        </h2>
        <div className="rounded-[var(--radius-card)] border border-line bg-surface p-5 sm:p-6">
          <ThemeToggle />
        </div>
        <DisplayForm profile={profile} />
      </section>

      {/* --------------------------------------------------------- The assistant */}
      <section aria-labelledby="assistant-heading" className="space-y-4">
        <h2 id="assistant-heading" className="text-xl font-semibold sm:text-2xl">
          The assistant
        </h2>
        <AssistantForm
          profile={profile}
          providerName={provider.name}
          providerIsExternal={provider.isExternal}
        />
      </section>

      {/* -------------------------------------------------------------- Your data */}
      <section aria-labelledby="data-heading" className="space-y-4">
        <h2 id="data-heading" className="text-xl font-semibold sm:text-2xl">
          Your data
        </h2>
        <DataExport totalRecords={totalRecords} />
      </section>

      {/* ----------------------------------------------------------------- Account */}
      <section aria-labelledby="account-heading" className="space-y-4">
        <h2 id="account-heading" className="text-xl font-semibold sm:text-2xl">
          Account
        </h2>

        <div className="space-y-4">
          <ChangePasswordForm />
          <SignOutEverywhereForm />
        </div>

        <div className="rounded-[var(--radius-card)] border-2 border-critical/50 bg-critical-soft/40 p-5 sm:p-6">
          <h3 className="text-lg font-semibold text-critical">
            <span aria-hidden="true">⚠ </span>
            Permanently delete
          </h3>
          <p className="mt-1 max-w-prose text-sm text-ink">
            These actions cannot be undone. Nothing is kept in a recoverable state — once you
            confirm, the data is gone.
          </p>
          <div className="mt-5 space-y-6">
            <DeleteAllRecordsForm totalRecords={totalRecords} userEmail={user.email} />
            <hr className="border-critical/30" />
            <DeleteAccountForm userEmail={user.email} />
          </div>
        </div>
      </section>
    </div>
  );
}
