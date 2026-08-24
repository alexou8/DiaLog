import type { Metadata } from 'next';
import Link from 'next/link';
import { ButtonLink, Card } from '@/components/ui';

export const metadata: Metadata = {
  title: 'DiaLog — understand your glucose data',
  description:
    'Bring your glucose readings, meals, activity and sleep together in one calm place, and see in plain language what your own data actually shows.',
};

const WHAT_IT_DOES = [
  {
    icon: '📥',
    title: 'Get your readings in without retyping them',
    body: 'Import the CSV, Excel, JSON or XML files your meter software already exports — or add readings by hand in about ten seconds. Every import tells you exactly what was added, what was skipped as a duplicate, and what could not be read.',
  },
  {
    icon: '🔎',
    title: 'See patterns, not just numbers',
    body: 'DiaLog compares your readings against your own history — mornings against evenings, days you walked against days you did not — and only says something when there is enough of your data to back it up.',
  },
  {
    icon: '💬',
    title: 'Ask questions in your own words',
    body: 'Ask "why was my glucose higher this week?" and get an answer built from your own numbers, with the evidence behind it shown alongside. When the data cannot answer, DiaLog says so.',
  },
  {
    icon: '📄',
    title: 'Bring something useful to your appointment',
    body: 'Weekly and monthly summaries written in plain language, including questions worth raising with your healthcare professional.',
  },
];

export default function LandingPage() {
  return (
    <>
      <section className="mx-auto max-w-5xl px-5 py-14 sm:py-20">
        <p className="mb-3 inline-block rounded-full border border-brand/30 bg-brand-soft px-3 py-1 text-sm font-semibold text-brand-ink">
          For prediabetes, type 2 diabetes, and anyone tracking their glucose
        </p>
        <h1 className="max-w-3xl text-4xl font-bold tracking-tight sm:text-5xl">
          Your glucose data, finally in plain language.
        </h1>
        <p className="mt-5 max-w-2xl text-lg text-ink-muted sm:text-xl">
          DiaLog collects your readings, meals, activity and sleep in one place, works out what is
          actually going on, and explains it the way a person would — without jargon, alarms, or
          twenty charts you have to interpret yourself.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <ButtonLink href="/sign-up" className="text-lg">
            Create your account
          </ButtonLink>
          <ButtonLink href="/about" variant="secondary" className="text-lg">
            See how it works
          </ButtonLink>
        </div>
        <p className="mt-5 text-sm text-ink-muted">
          Free and open source. Works in mmol/L or mg/dL. Installs to your phone&apos;s home screen.
        </p>
      </section>

      <section aria-labelledby="what" className="border-y border-line bg-surface">
        <div className="mx-auto max-w-5xl px-5 py-14">
          <h2 id="what" className="text-2xl font-bold sm:text-3xl">
            What DiaLog actually does
          </h2>
          <ul className="mt-8 grid gap-5 sm:grid-cols-2">
            {WHAT_IT_DOES.map((item) => (
              <Card as="li" key={item.title} className="bg-canvas">
                <p aria-hidden="true" className="text-2xl">
                  {item.icon}
                </p>
                <h3 className="mt-3 text-lg font-semibold">{item.title}</h3>
                <p className="mt-2 text-ink-muted">{item.body}</p>
              </Card>
            ))}
          </ul>
        </div>
      </section>

      <section aria-labelledby="honest" className="mx-auto max-w-5xl px-5 py-14">
        <h2 id="honest" className="text-2xl font-bold sm:text-3xl">
          What DiaLog will never do
        </h2>
        <div className="mt-6 grid gap-5 sm:grid-cols-3">
          <Card>
            <h3 className="font-semibold">It won&apos;t practise medicine</h3>
            <p className="mt-2 text-ink-muted">
              No diagnoses, no dose calculations, no instructions about your medication. DiaLog
              describes your data and helps you take good questions to your healthcare professional.
            </p>
          </Card>
          <Card>
            <h3 className="font-semibold">It won&apos;t guess</h3>
            <p className="mt-2 text-ink-muted">
              Every observation carries how many of your records it is based on. When there is not
              enough, DiaLog says &ldquo;not enough data yet&rdquo; instead of inventing a pattern.
            </p>
          </Card>
          <Card>
            <h3 className="font-semibold">It won&apos;t alarm you</h3>
            <p className="mt-2 text-ink-muted">
              One unusual reading is not an emergency. DiaLog explains context calmly and reserves
              stronger wording for the few situations that genuinely warrant it.
            </p>
          </Card>
        </div>
      </section>

      <section aria-labelledby="access" className="border-t border-line bg-surface-sunken">
        <div className="mx-auto grid max-w-5xl gap-8 px-5 py-14 sm:grid-cols-2">
          <div>
            <h2 id="access" className="text-2xl font-bold sm:text-3xl">
              Built to be readable
            </h2>
            <p className="mt-4 text-ink-muted">
              Large type, generous spacing, big buttons and short forms. Full keyboard and screen
              reader support, and every chart comes with the same information as text and as a table.
              Status is never shown by colour alone — there is always a label and an icon too.
            </p>
            <p className="mt-4">
              <Link href="/accessibility" className="font-semibold underline underline-offset-4">
                Read our accessibility commitment
              </Link>
            </p>
          </div>
          <div>
            <h2 className="text-2xl font-bold sm:text-3xl">Your data stays yours</h2>
            <p className="mt-4 text-ink-muted">
              Health records are stored in your own account and never sold or used for advertising.
              The assistant runs on a local, no-network explanation engine by default — nothing is
              sent to an external AI provider unless you turn that on yourself.
            </p>
            <p className="mt-4">
              <Link href="/privacy" className="font-semibold underline underline-offset-4">
                Read the privacy notice
              </Link>
            </p>
          </div>
        </div>
      </section>
    </>
  );
}
