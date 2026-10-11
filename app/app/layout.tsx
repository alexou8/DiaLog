import Link from 'next/link';
import type { Metadata } from 'next';
import { requireUser } from '@/lib/auth/current-user';
import { getDictionary } from '@/lib/i18n/dictionaries';
import { signOutAction } from '@/lib/actions/auth';
import { AddEntryButton, BottomNav, SideNav } from '@/components/AppNav';
import { AccountPreferences } from '@/components/AccountPreferences';
import { Icon, LogoMark } from '@/components/ui';

export const metadata: Metadata = {
  robots: { index: false, follow: false, nocache: true },
};

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const user = await requireUser();
  const dict = getDictionary(user.profile.locale);
  // Only the chrome strings come from the dictionary; see SideNav's dictLang.
  const chromeLang = user.profile.locale.startsWith('fr') ? 'fr-CA' : undefined;

  return (
    // `data-app-shell` is what the scroll-padding rule in globals.css keys
    // on, so focused controls are never scrolled under the sticky header or
    // the fixed bottom bar.
    <div className="min-h-dvh" data-app-shell="">
      <AccountPreferences
        largeText={user.profile.largeText}
        reduceMotion={user.profile.reduceMotion}
      />
      <a href="#main" className="dl-skip-link" lang={chromeLang}>
        {dict.nav.skipToContent}
      </a>

      <header className="dl-safe-top sticky top-0 z-30 border-b border-line bg-surface/95 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <Link
            href="/app"
            className="dl-target flex items-center gap-2 text-lg font-semibold tracking-tight"
          >
            <LogoMark size={30} />
            DiaLog
          </Link>
          <div className="flex items-center gap-2" lang={chromeLang}>
            <AddEntryButton label={dict.nav.add} />
            <form action={signOutAction}>
              <button
                type="submit"
                className="dl-target inline-flex items-center gap-2 rounded-[var(--radius-control)] px-3 py-2 text-sm font-semibold text-ink-muted hover:bg-surface-sunken hover:text-ink"
              >
                <Icon name="signOut" />
                {dict.nav.signOut}
              </button>
            </form>
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-6xl gap-8 px-4 pb-28 pt-6 sm:px-6 lg:pb-12">
        <aside className="hidden w-56 shrink-0 lg:block">
          <SideNav dict={dict} />
        </aside>
        <main id="main" tabIndex={-1} className="min-w-0 flex-1">
          {children}
        </main>
      </div>

      <BottomNav dict={dict} />
    </div>
  );
}
