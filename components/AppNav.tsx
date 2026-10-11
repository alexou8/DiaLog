'use client';

import { useRef } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import clsx from 'clsx';
import type { Dictionary } from '@/lib/i18n/dictionaries';
import { Icon, type IconName } from '@/components/ui/icon';
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';

export interface NavItem {
  href: string;
  labelKey: keyof Dictionary['nav'];
  icon: IconName;
  /** Has its own tab in the mobile bar; everything else is under "More". */
  primary?: boolean;
}

export const NAV_ITEMS: NavItem[] = [
  { href: '/app', labelKey: 'home', icon: 'home', primary: true },
  { href: '/app/insights', labelKey: 'insights', icon: 'insights', primary: true },
  { href: '/app/glucose', labelKey: 'glucose', icon: 'glucose', primary: true },
  { href: '/app/meals', labelKey: 'meals', icon: 'meals' },
  { href: '/app/activity', labelKey: 'activity', icon: 'activity' },
  { href: '/app/health', labelKey: 'health', icon: 'health' },
  { href: '/app/assistant', labelKey: 'assistant', icon: 'assistant' },
  { href: '/app/reports', labelKey: 'reports', icon: 'reports' },
  { href: '/app/history', labelKey: 'history', icon: 'history' },
  { href: '/app/import', labelKey: 'import', icon: 'import' },
  { href: '/app/settings', labelKey: 'settings', icon: 'settings' },
];

/**
 * Every kind of entry the product can record, in one place. The header's
 * "Add" used to go straight to the glucose form, which was the only way it
 * said what "Add" meant; the other ten forms had no entry point on a phone.
 */
export const LOG_ENTRIES: { href: string; label: string; icon: IconName; hint?: string }[] = [
  { href: '/app/glucose/new', label: 'Glucose reading', icon: 'glucose' },
  { href: '/app/meals/new', label: 'Meal', icon: 'meals' },
  { href: '/app/activity/new', label: 'Activity', icon: 'activity' },
  { href: '/app/health/weight/new', label: 'Weight', icon: 'weight' },
  { href: '/app/health/blood-pressure/new', label: 'Blood pressure', icon: 'bloodPressure' },
  { href: '/app/health/sleep/new', label: 'Sleep', icon: 'sleep' },
  { href: '/app/health/mood/new', label: 'Mood', icon: 'mood' },
  { href: '/app/health/symptom/new', label: 'Symptom', icon: 'symptom' },
  { href: '/app/health/medication/new', label: 'Medication', icon: 'medication' },
  { href: '/app/health/hydration/new', label: 'Water', icon: 'hydration' },
  {
    href: '/app/quick-log',
    label: 'Describe your day',
    icon: 'quickLog',
    hint: 'Type it in your own words',
  },
];

const MORE_ITEMS = NAV_ITEMS.filter((item) => !item.primary);

function isActive(pathname: string, href: string): boolean {
  return href === '/app' ? pathname === '/app' : pathname.startsWith(href);
}

/**
 * Only the navigation chrome is translated (lib/i18n/dictionaries.ts), so it
 * declares its own language while the page around it stays `en-CA`. Without
 * this a French label would be read with an English voice, which is the same
 * 3.1.2 problem in reverse.
 */
function dictLang(dict: Dictionary): string | undefined {
  return dict.nav.home === 'Home' ? undefined : 'fr-CA';
}

/** Desktop/tablet sidebar. */
export function SideNav({ dict }: { dict: Dictionary }) {
  const pathname = usePathname();
  const lang = dictLang(dict);
  return (
    <nav aria-label={dict.nav.mainNavigation} lang={lang} className="hidden lg:block">
      <ul className="flex flex-col gap-1">
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
                  'dl-target flex w-full items-center gap-3 border-l-2 py-2.5 pl-3 pr-3 text-base',
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
 * Where focus goes when a sheet closes. Radix returns it to the trigger, which
 * is right after Escape or the close button. After following a link it is
 * not: the page has changed, and a keyboard or screen-reader user would land
 * back on "More" or "Log" in the bar instead of on what they asked to see. So
 * a followed link sends focus to the new page's main region instead.
 */
function useSheetLinkFocus() {
  const followed = useRef(false);
  return {
    onFollow: () => {
      followed.current = true;
    },
    onCloseAutoFocus: (event: Event) => {
      if (!followed.current) return;
      followed.current = false;
      event.preventDefault();
      document.getElementById('main')?.focus();
    },
  };
}

/** A list of links inside a sheet; following one closes the sheet. */
function SheetLinks({
  items,
  pathname,
  lang,
  onFollow,
}: {
  items: { href: string; label: string; icon: IconName; hint?: string }[];
  pathname: string;
  lang?: string;
  onFollow: () => void;
}) {
  return (
    <ul className="grid grid-cols-1 gap-2 px-4 pb-4 min-[22rem]:grid-cols-2">
      {items.map((item) => {
        const active = pathname === item.href;
        return (
          <li key={item.href}>
            <SheetClose asChild>
              <Link
                href={item.href}
                lang={lang}
                onClick={onFollow}
                aria-current={active ? 'page' : undefined}
                className={clsx(
                  'dl-target flex h-full w-full items-center gap-3 rounded-[var(--radius-control)] border-2 bg-surface px-3 py-3 font-semibold text-ink hover:border-brand',
                  active ? 'border-brand bg-brand-soft' : 'border-line-strong',
                )}
              >
                <Icon name={item.icon} className="shrink-0 text-brand-ink" size="1.35em" />
                <span>
                  <span className="block leading-snug">{item.label}</span>
                  {item.hint ? (
                    <span className="block text-sm font-normal text-ink-muted">{item.hint}</span>
                  ) : null}
                </span>
              </Link>
            </SheetClose>
          </li>
        );
      })}
    </ul>
  );
}

/**
 * The "what would you like to record?" panel, shared by the header's Add
 * button and the bottom bar's Log tab so the two can never disagree.
 */
function LogSheetContent({ side, pathname }: { side: 'right' | 'bottom'; pathname: string }) {
  const { onFollow, onCloseAutoFocus } = useSheetLinkFocus();
  return (
    <SheetContent
      side={side}
      onCloseAutoFocus={onCloseAutoFocus}
      className={clsx(
        'overflow-y-auto',
        side === 'bottom' &&
          'max-h-[85dvh] rounded-t-[var(--radius-card)] pb-[max(env(safe-area-inset-bottom),0.5rem)]',
      )}
    >
      <SheetHeader className="pr-16">
        <SheetTitle className="text-xl">What would you like to log?</SheetTitle>
        <SheetDescription>
          Each one opens a short form. Nothing is saved until you do.
        </SheetDescription>
      </SheetHeader>
      <SheetLinks items={LOG_ENTRIES} pathname={pathname} onFollow={onFollow} />
    </SheetContent>
  );
}

/**
 * Header action that opens the log panel. A real <button> (Radix gives it
 * `aria-haspopup="dialog"` and `aria-expanded`), so "Add" is announced as
 * something that opens a choice rather than as a link to one form.
 */
export function AddEntryButton({ label }: { label: string }) {
  const pathname = usePathname();
  return (
    <Sheet>
      <SheetTrigger className="inline-flex min-h-11 items-center gap-2 rounded-[var(--radius-control)] bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground transition-colors hover:bg-brand-strong sm:text-base">
        <Icon name="add" />
        {label}
      </SheetTrigger>
      <LogSheetContent side="right" pathname={pathname} />
    </Sheet>
  );
}

const TAB_CLASS =
  'flex w-full flex-col items-center gap-1 border-t-2 px-1 pb-2 pt-2 text-xs font-medium';

/**
 * Mobile bottom bar: Home · Glucose · Log · Insights · More.
 *
 * Five targets, because a phone-sized tab bar with more stops being
 * tappable. The previous bar held five destinations and said everything else
 * lived behind "More" on the Home screen, but Home had no such section:
 * Meals, Activity, Health, Reports, History and Import had no route in at all
 * below 1024px. "More" is now a real panel listing every other destination,
 * and "Log" opens the same panel as the header's Add.
 */
export function BottomNav({ dict }: { dict: Dictionary }) {
  const pathname = usePathname();
  const lang = dictLang(dict);
  const moreActive = MORE_ITEMS.some((item) => isActive(pathname, item.href));
  const moreFocus = useSheetLinkFocus();

  const tab = (item: NavItem) => {
    const active = isActive(pathname, item.href);
    return (
      <li key={item.href} className="flex-1">
        <Link
          href={item.href}
          aria-current={active ? 'page' : undefined}
          className={clsx(
            TAB_CLASS,
            // A top rule repeats what the colour says, for the same reason
            // as the sidebar.
            active
              ? 'border-brand font-semibold text-brand-ink'
              : 'border-transparent text-ink-muted',
          )}
        >
          <Icon name={item.icon} size="1.5em" />
          <span lang={lang}>{dict.nav[item.labelKey]}</span>
        </Link>
      </li>
    );
  };

  const [home, insights, glucose] = NAV_ITEMS;

  return (
    <nav
      aria-label={dict.nav.mainNavigation}
      className="dl-safe-bottom fixed inset-x-0 bottom-0 z-40 border-t border-line bg-surface lg:hidden"
    >
      <ul className="mx-auto flex max-w-lg items-stretch">
        {tab(home!)}
        {tab(glucose!)}
        <li className="flex-1">
          <Sheet>
            <SheetTrigger className={clsx(TAB_CLASS, 'border-transparent text-brand-ink')}>
              {/* The one filled control in the bar: logging is what the app
                  is for, and the fill marks it as an action, not a place. */}
              <span className="-mt-1 inline-flex size-8 items-center justify-center rounded-full bg-primary text-primary-foreground">
                <Icon name="add" size="1.25em" />
              </span>
              <span className="font-semibold">Log</span>
            </SheetTrigger>
            <LogSheetContent side="bottom" pathname={pathname} />
          </Sheet>
        </li>
        {tab(insights!)}
        <li className="flex-1">
          <Sheet>
            <SheetTrigger
              className={clsx(
                TAB_CLASS,
                moreActive
                  ? 'border-brand font-semibold text-brand-ink'
                  : 'border-transparent text-ink-muted',
              )}
            >
              <Icon name="more" size="1.5em" />
              <span>More</span>
            </SheetTrigger>
            <SheetContent
              side="bottom"
              onCloseAutoFocus={moreFocus.onCloseAutoFocus}
              className="max-h-[85dvh] overflow-y-auto rounded-t-[var(--radius-card)] pb-[max(env(safe-area-inset-bottom),0.5rem)]"
            >
              <SheetHeader className="pr-16">
                <SheetTitle className="text-xl">More of DiaLog</SheetTitle>
                <SheetDescription>Everything that does not fit in the bar below.</SheetDescription>
              </SheetHeader>
              <SheetLinks
                lang={lang}
                pathname={pathname}
                onFollow={moreFocus.onFollow}
                items={MORE_ITEMS.map((item) => ({
                  href: item.href,
                  label: dict.nav[item.labelKey],
                  icon: item.icon,
                }))}
              />
            </SheetContent>
          </Sheet>
        </li>
      </ul>
    </nav>
  );
}
