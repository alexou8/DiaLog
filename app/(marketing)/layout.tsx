import Link from 'next/link';
import { ButtonLink, LogoMark, MedicalDisclaimer } from '@/components/ui';

const FOOTER_LINKS = [
  { href: '/about', label: 'About' },
  { href: '/privacy', label: 'Privacy' },
  { href: '/security', label: 'Security' },
  { href: '/accessibility', label: 'Accessibility' },
  { href: '/terms', label: 'Terms' },
  { href: '/help', label: 'Help' },
];

export default function MarketingLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-dvh flex-col">
      <a href="#main" className="dl-skip-link">
        Skip to main content
      </a>
      <header className="dl-safe-top border-b border-line bg-surface">
        {/* Wraps, and the buttons tighten, below ~400px. Without both, the
            logo plus the two account buttons measured 445px wide and the whole
            marketing site scrolled sideways on a 360px phone. The min-h-11
            floor is untouched, so the 44px touch target survives the smaller
            padding. */}
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-x-4 gap-y-2 px-5 py-3">
          <Link href="/" className="dl-target flex items-center gap-2 text-lg font-bold">
            <LogoMark size={32} />
            DiaLog
          </Link>
          <nav aria-label="Account">
            <ul className="flex items-center gap-2">
              <li>
                <ButtonLink
                  href="/sign-in"
                  variant="ghost"
                  className="px-3 text-sm sm:px-5 sm:text-base"
                >
                  Sign in
                </ButtonLink>
              </li>
              <li>
                <ButtonLink href="/sign-up" className="px-3 text-sm sm:px-5 sm:text-base">
                  Create account
                </ButtonLink>
              </li>
            </ul>
          </nav>
        </div>
      </header>

      <main id="main" tabIndex={-1} className="flex-1">
        {children}
      </main>

      <footer className="border-t border-line bg-surface">
        <div className="mx-auto max-w-5xl px-5 py-8">
          <nav aria-label="Site information">
            <ul className="flex flex-wrap gap-x-6 gap-y-2">
              {FOOTER_LINKS.map((link) => (
                <li key={link.href}>
                  <Link href={link.href} className="dl-target underline underline-offset-4">
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
          <div className="mt-5 max-w-prose">
            <MedicalDisclaimer />
          </div>
          <p className="mt-4 text-sm text-ink-muted">
            DiaLog is an open-source personal project. It is not affiliated with, or endorsed by,
            any device manufacturer.
          </p>
        </div>
      </footer>
    </div>
  );
}
