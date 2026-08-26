import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { getProvider } from '@/lib/ai/provider';
import { recordCounts } from '@/lib/db/health-records';
import { Icon, PageHeader } from '@/components/ui';
import { ThemeToggle } from '@/components/ThemeToggle';
import { UnitsForm } from './UnitsForm';
import { RegionForm } from './RegionForm';
import { DisplayForm } from './DisplayForm';
import { AssistantForm } from './AssistantForm';
import { DataExport } from './DataExport';
import { ConnectedAccounts } from './ConnectedAccounts';
import { prisma } from '@/lib/db/prisma';
import { isGoogleEnabled } from '@/lib/auth/oauth/google';
import { oauthMessage } from '@/lib/auth/oauth/link';
import {
  ChangePasswordForm,
  SignOutEverywhereForm,
  DeleteAllRecordsForm,
  DeleteAccountForm,
  type AccountFeedback,
} from './AccountForms';
import { PASSWORD_POLICY_MESSAGES } from '@/lib/auth/password';
import type { PasswordOutcome } from '@/app/api/auth/password/route';
import type { SessionsOutcome } from '@/app/api/auth/sessions/revoke/route';

export const metadata: Metadata = { title: 'Settings' };
export const dynamic = 'force-dynamic';

/**
 * Confirmations that arrive as a query parameter rather than as returned action
 * state, because the handler that produced them answers with a redirect the
 * browser follows itself — see lib/auth/route-form.ts for why every
 * credential-changing form on this page works that way.
 */
const NOTICES = {
  linked: {
    google: 'Your Google account is now connected. You can sign in with it next time.',
    google_already: 'That Google account was already connected to your DiaLog account.',
  },
  unlinked: {
    google: 'Google has been disconnected. Sign in with your email and password from now on.',
    absent: 'Your account is not connected to Google.',
    blocked:
      'Google is currently the only way to sign in to this account. Set a password above first, then you can disconnect Google.',
  },
} satisfies Record<string, Record<string, string>>;

/** Looks up one notice, tolerating a hand-edited query string. */
function notice(group: Record<string, string>, key: string | undefined): string | null {
  return key ? (group[key] ?? null) : null;
}

/**
 * Every outcome app/api/auth/password/route.ts can report, as the sentence and
 * the field it belongs to. Typing this by `PasswordOutcome` rather than by
 * `string` is what makes a new code a compile error here instead of a silently
 * blank confirmation in front of a user.
 */
const PASSWORD_FEEDBACK: Record<PasswordOutcome, AccountFeedback> = {
  changed: {
    ok: true,
    message: 'Your password has been changed. You have been signed out of any other devices.',
  },
  set: {
    ok: true,
    message:
      'Your password has been set. You can now sign in with your email and password as well as with Google.',
  },
  missing_current: {
    ok: false,
    field: 'currentPassword',
    message: 'Please enter your current password.',
  },
  wrong_current: {
    ok: false,
    field: 'currentPassword',
    message: 'That is not your current password. Please try again.',
  },
  mismatch: {
    ok: false,
    field: 'confirmPassword',
    message: 'The two new passwords do not match.',
  },
  too_short: { ok: false, field: 'newPassword', message: PASSWORD_POLICY_MESSAGES.too_short },
  too_long: { ok: false, field: 'newPassword', message: PASSWORD_POLICY_MESSAGES.too_long },
  too_common: { ok: false, field: 'newPassword', message: PASSWORD_POLICY_MESSAGES.too_common },
  rate_limited: { ok: false, message: 'Please wait a moment and try again.' },
};

const SESSIONS_FEEDBACK: Record<SessionsOutcome, AccountFeedback> = {
  revoked: {
    ok: true,
    message: 'Every other device has been signed out. This device stays signed in.',
  },
  rate_limited: { ok: false, message: 'Please wait a moment and try again.' },
};

/** As `notice()`, but for the structured feedback the account forms render. */
function feedback<K extends string>(
  group: Record<K, AccountFeedback>,
  key: string | undefined,
): AccountFeedback | null {
  return key ? ((group as Record<string, AccountFeedback>)[key] ?? null) : null;
}

export default async function SettingsPage({
  searchParams,
}: {
  searchParams: Promise<{
    error?: string;
    linked?: string;
    unlinked?: string;
    password?: string;
    sessions?: string;
  }>;
}) {
  const params = await searchParams;
  const user = await requireOnboardedUser();
  const hasPassword = user.passwordHash !== null;
  const { profile } = user;
  const counts = await recordCounts(user.id);
  const totalRecords = Object.values(counts).reduce((a, b) => a + b, 0);

  const provider = getProvider();

  const googleIdentity = isGoogleEnabled()
    ? await prisma.authIdentity.findUnique({
        where: { userId_provider: { userId: user.id, provider: 'google' } },
        select: { email: true },
      })
    : null;

  const unlinkNotice = notice(NOTICES.unlinked, params.unlinked);
  const linkNotice = notice(NOTICES.linked, params.linked) ?? unlinkNotice;
  // Only a completed disconnect is good news; the other two explain a refusal.
  const linkNoticeIsGood = params.unlinked ? params.unlinked === 'google' : true;
  const linkError = oauthMessage(params.error);

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
          <ChangePasswordForm
            hasPassword={hasPassword}
            feedback={feedback(PASSWORD_FEEDBACK, params.password)}
          />
          {isGoogleEnabled() ? (
            <ConnectedAccounts
              googleEmail={googleIdentity?.email ?? null}
              hasPassword={hasPassword}
              notice={
                linkError
                  ? { ok: false, message: linkError }
                  : linkNotice
                    ? { ok: linkNoticeIsGood, message: linkNotice }
                    : null
              }
            />
          ) : null}
          <SignOutEverywhereForm feedback={feedback(SESSIONS_FEEDBACK, params.sessions)} />
        </div>

        <div className="rounded-[var(--radius-card)] border-2 border-critical/50 bg-critical-soft/40 p-5 sm:p-6">
          <h3 className="text-lg font-semibold text-critical">
            <Icon name="caution" className="shrink-0" />
            Permanently delete
          </h3>
          <p className="mt-1 max-w-prose text-sm text-ink">
            These actions cannot be undone. Nothing is kept in a recoverable state. Once you
            confirm, the data is gone.
          </p>
          <div className="mt-5 space-y-6">
            <DeleteAllRecordsForm
              totalRecords={totalRecords}
              userEmail={user.email}
              hasPassword={hasPassword}
            />
            <hr className="border-critical/30" />
            <DeleteAccountForm userEmail={user.email} hasPassword={hasPassword} />
          </div>
        </div>
      </section>
    </div>
  );
}
