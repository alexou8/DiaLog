import Link from 'next/link';
import Image from 'next/image';
import { MedicalDisclaimer } from '@/components/ui';

export default function AuthLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-dvh flex-col">
      <a href="#main" className="dl-skip-link">
        Skip to main content
      </a>
      <header className="dl-safe-top px-5 py-4">
        <Link href="/" className="dl-target inline-flex items-center gap-2 text-lg font-bold">
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
      </header>
      <main id="main" tabIndex={-1} className="flex flex-1 items-start justify-center px-5 py-8">
        <div className="w-full max-w-md">{children}</div>
      </main>
      <footer className="mx-auto max-w-md px-5 pb-10">
        <MedicalDisclaimer compact />
      </footer>
    </div>
  );
}
