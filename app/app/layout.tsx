import Link from 'next/link';
import Image from 'next/image';
import type { Metadata } from 'next';
import { requireUser } from '@/lib/auth/current-user';
import { getDictionary } from '@/lib/i18n/dictionaries';
import { signOutAction } from '@/lib/actions/auth';
import { BottomNav, SideNav } from '@/components/AppNav';
import { ButtonLink, Icon } from '@/components/ui';

export const metadata: Metadata = {
  robots: { index: false, follow: false, nocache: true },
};

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const user = await requireUser();
  const dict = getDictionary(user.profile.locale);

  return (
    <div className="min-h-dvh">
      <a href="#main" className="dl-skip-link">
        {dict.nav.skipToContent}
      </a>

      <header className="dl-safe-top sticky top-0 z-30 border-b border-line bg-surface/95 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <Link
            href="/app"
            className="dl-target flex items-center gap-2 text-lg font-semibold tracking-tight"
          >
            <Image
              src="/icons/icon-192.png"
              alt=""
              width={30}
              height={30}
              className="rounded-lg"
              priority
            />
            DiaLog
          </Link>
          <div className="flex items-center gap-2">
            <ButtonLink href="/app/glucose/new" className="px-4 py-2 text-sm sm:text-base">
              <Icon name="add" /> {dict.nav.add}
            </ButtonLink>
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
