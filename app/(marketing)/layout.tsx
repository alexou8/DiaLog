import Link from 'next/link';
import Image from 'next/image';
import { ButtonLink, MedicalDisclaimer } from '@/components/ui';

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
        <div className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-5 py-3">
          <Link href="/" className="dl-target flex items-center gap-2 text-lg font-bold">
            <Image
              src="/icons/icon-192.png"
              alt=""
              width={32}
              height={32}
              className="rounded-lg"
              priority
            />
            DiaLog
          </Link>
          <nav aria-label="Account">
            <ul className="flex items-center gap-2">
              <li>
                <ButtonLink href="/sign-in" variant="ghost">
                  Sign in
                </ButtonLink>
              </li>
              <li>
                <ButtonLink href="/sign-up">Create account</ButtonLink>
              </li>
            </ul>
          </nav>
        </div>
      </header>

      <main id="main" className="flex-1">
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
