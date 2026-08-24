'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import clsx from 'clsx';
import type { Dictionary } from '@/lib/i18n/dictionaries';

export interface NavItem {
  href: string;
  labelKey: keyof Dictionary['nav'];
  icon: string;
  /** Shown in the compact mobile bar. Five items maximum. */
  primary?: boolean;
}

export const NAV_ITEMS: NavItem[] = [
  { href: '/app', labelKey: 'home', icon: '🏠', primary: true },
  { href: '/app/insights', labelKey: 'insights', icon: '💡', primary: true },
  { href: '/app/glucose', labelKey: 'glucose', icon: '💧', primary: true },
  { href: '/app/meals', labelKey: 'meals', icon: '🍽️' },
  { href: '/app/activity', labelKey: 'activity', icon: '🚶' },
  { href: '/app/health', labelKey: 'health', icon: '❤️' },
  { href: '/app/assistant', labelKey: 'assistant', icon: '💬', primary: true },
  { href: '/app/reports', labelKey: 'reports', icon: '📄' },
  { href: '/app/history', labelKey: 'history', icon: '🗂️' },
  { href: '/app/import', labelKey: 'import', icon: '📥' },
  { href: '/app/settings', labelKey: 'settings', icon: '⚙️', primary: true },
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
                  'dl-target flex items-center gap-3 rounded-xl px-3 py-2.5 text-base font-medium',
                  active ? 'bg-brand-soft font-semibold text-brand-ink' : 'hover:bg-surface-sunken',
                )}
              >
                <span aria-hidden="true" className="text-lg">
                  {item.icon}
                </span>
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
                  'flex flex-col items-center gap-0.5 px-1 py-2 text-xs font-medium',
                  active ? 'text-brand-ink' : 'text-ink-muted',
                )}
              >
                <span aria-hidden="true" className="text-xl leading-none">
                  {item.icon}
                </span>
                <span className={clsx(active && 'font-bold')}>{dict.nav[item.labelKey]}</span>
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
