'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import clsx from 'clsx';
import type { Dictionary } from '@/lib/i18n/dictionaries';
import { Icon, type IconName } from '@/components/ui/icon';

export interface NavItem {
  href: string;
  labelKey: keyof Dictionary['nav'];
  icon: IconName;
  /** Shown in the compact mobile bar. Five items maximum. */
  primary?: boolean;
}

export const NAV_ITEMS: NavItem[] = [
  { href: '/app', labelKey: 'home', icon: 'home', primary: true },
  { href: '/app/insights', labelKey: 'insights', icon: 'insights', primary: true },
  { href: '/app/glucose', labelKey: 'glucose', icon: 'glucose', primary: true },
  { href: '/app/meals', labelKey: 'meals', icon: 'meals' },
  { href: '/app/activity', labelKey: 'activity', icon: 'activity' },
  { href: '/app/health', labelKey: 'health', icon: 'health' },
  { href: '/app/assistant', labelKey: 'assistant', icon: 'assistant', primary: true },
  { href: '/app/reports', labelKey: 'reports', icon: 'reports' },
  { href: '/app/history', labelKey: 'history', icon: 'history' },
  { href: '/app/import', labelKey: 'import', icon: 'import' },
  { href: '/app/settings', labelKey: 'settings', icon: 'settings', primary: true },
];

function isActive(pathname: string, href: string): boolean {
  return href === '/app' ? pathname === '/app' : pathname.startsWith(href);
}

/** Desktop/tablet sidebar. */
export function SideNav({ dict }: { dict: Dictionary }) {
  const pathname = usePathname();
  return (
    <nav aria-label={dict.nav.mainNavigation} className="hidden lg:block">
      <ul className="space-y-1">
        {NAV_ITEMS.map((item) => {
          const active = isActive(pathname, item.href);
          return (
            <li key={item.href}>
              <Link
                href={item.href}
                aria-current={active ? 'page' : undefined}
                className={clsx(
                  // The active item is marked by a left rule as well as by
                  // colour and `aria-current`, so the current location is
                  // still obvious without colour perception.
                  'dl-target flex items-center gap-3 border-l-2 py-2.5 pl-3 pr-3 text-base',
                  active
                    ? 'border-brand bg-brand-soft font-semibold text-brand-ink'
                    : 'border-transparent font-medium text-ink-muted hover:border-line-strong hover:bg-surface-sunken hover:text-ink',
                )}
              >
                <Icon name={item.icon} className="shrink-0" size="1.25em" />
                {dict.nav[item.labelKey]}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

/**
 * Mobile bottom bar. Only five destinations, because a phone-sized tab bar with
 * more than five targets stops being tappable. Everything else lives behind
 * "More" on the Home screen rather than in a nested menu.
 */
export function BottomNav({ dict }: { dict: Dictionary }) {
  const pathname = usePathname();
  const items = NAV_ITEMS.filter((item) => item.primary);
  return (
    <nav
      aria-label={dict.nav.mainNavigation}
      className="dl-safe-bottom fixed inset-x-0 bottom-0 z-40 border-t border-line bg-surface lg:hidden"
    >
      <ul className="mx-auto flex max-w-lg">
        {items.map((item) => {
          const active = isActive(pathname, item.href);
          return (
            <li key={item.href} className="flex-1">
              <Link
                href={item.href}
                aria-current={active ? 'page' : undefined}
                className={clsx(
                  // A top rule repeats what the colour says, for the same
                  // reason as the sidebar.
                  'flex flex-col items-center gap-1 border-t-2 px-1 pb-2 pt-2 text-xs font-medium',
                  active ? 'border-brand text-brand-ink' : 'border-transparent text-ink-muted',
                )}
              >
                <Icon name={item.icon} size="1.5em" />
                <span className={clsx(active && 'font-semibold')}>{dict.nav[item.labelKey]}</span>
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
