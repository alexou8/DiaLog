import type { Metadata } from 'next';
import { Prose } from '../_prose';

export const metadata: Metadata = {
  title: 'Privacy',
  description: 'What DiaLog stores, what it never stores, and the choices you control.',
};

export default function PrivacyPage() {
  return (
    <Prose title="Privacy notice" updated="August 2026">
      <p>
        Health data is among the most sensitive information about a person. This page describes what
        DiaLog does with yours in specific terms rather than reassuring generalities.
      </p>

      <h2>What is stored</h2>
      <ul>
        <li>Your email address and a one-way hash of your password. The password itself is never stored.</li>
        <li>The health records you enter or import, with their timestamps and their source.</li>
        <li>Your preferences: units, target range, timezone, language, display and assistant settings.</li>
        <li>A security log of sign-ins, imports, exports and deletions. It contains no health values.</li>
      </ul>

      <h2>What is never stored or logged</h2>
      <ul>
        <li>Your password in a readable form.</li>
        <li>Health values in application logs or error reports.</li>
        <li>Advertising or third-party analytics identifiers. DiaLog carries no tracking scripts.</li>
      </ul>

      <h2>The assistant and external AI providers</h2>
      <p>
        By default the assistant uses a local explanation engine that runs on the same server as the
        rest of DiaLog. In that mode <strong>no health information leaves the deployment</strong>.
      </p>
      <p>
        If an external AI provider is configured and you switch it on, DiaLog sends only the
        structured findings produced by the analysis — figures such as &ldquo;average post-dinner
        reading across 14 comparable days&rdquo; — never your raw records, and never your free-text
        notes unless you separately consent to that. You can turn the assistant off entirely in
        Settings, and everything else keeps working.
      </p>

      <h2>Your control</h2>
      <ul>
        <li>Export everything you have stored, at any time, as JSON or CSV.</li>
        <li>Correct or delete any individual record.</li>
        <li>Delete your account and all associated records. Deletion is permanent and confirmed first.</li>
        <li>See where any imported record came from, and remove an entire import batch.</li>
      </ul>

      <h2>Storage and transport</h2>
      <p>
        Data is held in a PostgreSQL database and transported over TLS. On a managed platform the
        database is encrypted at rest by the provider. Session cookies are HTTP-only, same-site and
        signed; changing your password invalidates every existing session.
      </p>

      <h2>Retention</h2>
      <p>
        Your records are kept until you delete them or your account. Security log entries are kept
        for the same period so that you can audit access to your own account.
      </p>
    </Prose>
  );
}
