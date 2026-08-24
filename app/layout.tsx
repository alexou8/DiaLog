import type { Metadata, Viewport } from 'next';
import './globals.css';
import { getCurrentUser } from '@/lib/auth/current-user';
import { PreferencesScript } from '@/components/PreferencesScript';
import { ServiceWorkerRegistration } from '@/components/ServiceWorkerRegistration';

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_APP_URL ?? 'http://localhost:3000'),
  title: {
    default: 'DiaLog — understand your glucose data',
    template: '%s · DiaLog',
  },
  description:
    'DiaLog brings your glucose readings, meals, activity and sleep together and explains, in plain language, what your own data shows.',
  applicationName: 'DiaLog',
  manifest: '/manifest.webmanifest',
  appleWebApp: { capable: true, title: 'DiaLog', statusBarStyle: 'default' },
  icons: { icon: '/icons/icon-192.png', apple: '/icons/icon-180.png' },
  openGraph: {
    type: 'website',
    siteName: 'DiaLog',
    title: 'DiaLog — understand your glucose data',
    description: 'Calm, accessible tracking for glucose and the everyday things that affect it.',
  },
};

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#f7f6f2' },
    { media: '(prefers-color-scheme: dark)', color: '#14181f' },
  ],
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
};

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const user = await getCurrentUser();
  const profile = user?.profile;

  return (
    <html
      lang={profile?.locale.startsWith('fr') ? 'fr-CA' : 'en-CA'}
      data-text={profile?.largeText ? 'large' : undefined}
      data-motion={profile?.reduceMotion ? 'reduced' : undefined}
      suppressHydrationWarning
    >
      <body className="min-h-dvh antialiased">
        <PreferencesScript />
        {children}
        <ServiceWorkerRegistration />
      </body>
    </html>
  );
}
