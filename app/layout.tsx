import type { Metadata, Viewport } from 'next';
import './globals.css';
import { PreferencesScript } from '@/components/PreferencesScript';
import { ServiceWorkerRegistration } from '@/components/ServiceWorkerRegistration';
import { THEME_COLORS } from '@/components/theme';

// Resolve the canonical origin. `NEXT_PUBLIC_APP_URL` may be present but empty
// on Vercel, so fall back through the deployment URL before localhost.
function resolveAppUrl(): string {
  const explicit = process.env.NEXT_PUBLIC_APP_URL?.trim();
  if (explicit) return explicit;
  const vercel =
    process.env.VERCEL_PROJECT_PRODUCTION_URL?.trim() || process.env.VERCEL_URL?.trim();
  if (vercel) return `https://${vercel}`;
  return 'http://localhost:3000';
}

export const metadata: Metadata = {
  metadataBase: new URL(resolveAppUrl()),
  title: {
    default: 'DiaLog: understand your glucose data',
    template: '%s · DiaLog',
  },
  description:
    'DiaLog brings your glucose readings, meals, activity and sleep together and explains, in plain language, what your own data shows.',
  applicationName: 'DiaLog',
  manifest: '/manifest.webmanifest',
  appleWebApp: { capable: true, title: 'DiaLog', statusBarStyle: 'default' },
  // SVG first so the tab icon stays sharp on any DPI and in dark tab bars.
  // The ICO is a fallback, not a duplicate: Safari only gained SVG favicon
  // support in 17, and every browser still probes /favicon.ico regardless of
  // what is declared here.
  icons: {
    icon: [
      { url: '/favicon.svg', type: 'image/svg+xml' },
      { url: '/favicon.ico', sizes: '32x32' },
    ],
    apple: '/icons/icon-180.png',
  },
  openGraph: {
    type: 'website',
    siteName: 'DiaLog',
    title: 'DiaLog: understand your glucose data',
    description: 'Calm, accessible tracking for glucose and the everyday things that affect it.',
  },
};

export const viewport: Viewport = {
  themeColor: [
    // Initial values for the OS scheme. THEME_SCRIPT rewrites both to the
    // in-app choice before paint, so a Dark choice on a light-mode phone
    // still gets a dark browser bar.
    { media: '(prefers-color-scheme: light)', color: THEME_COLORS.light },
    { media: '(prefers-color-scheme: dark)', color: THEME_COLORS.dark },
  ],
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
};

/**
 * The root layout reads no session, on purpose. It used to call
 * `getCurrentUser()` for the language and the text-size and motion
 * preferences, and reading cookies here made every route dynamic: each
 * anonymous visit to `/`, `/privacy` or `/help` cost a database query and
 * could not be served from a CDN. Account preferences are now applied by the
 * signed-in shell (app/app/layout.tsx → AccountPreferences), the only place
 * they were ever needed.
 *
 * `lang` is the language the interface actually renders, not the locale
 * preference. Only the navigation chrome has a French translation (see
 * lib/i18n/dictionaries.ts); declaring `fr-CA` on pages that are about 95%
 * English made screen readers read every English sentence with a French
 * voice (WCAG 3.1.1). The translated chrome marks itself `lang="fr-CA"`
 * instead, and the locale preference still drives number and date formatting.
 */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en-CA" suppressHydrationWarning>
      <body className="min-h-dvh antialiased">
        <PreferencesScript />
        {children}
        <ServiceWorkerRegistration />
      </body>
    </html>
  );
}
