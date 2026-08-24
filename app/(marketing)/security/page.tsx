import type { Metadata } from 'next';
import { Prose } from '../_prose';

export const metadata: Metadata = {
  title: 'Security',
  description: 'How DiaLog protects accounts and health records.',
};

export default function SecurityPage() {
  return (
    <Prose title="Security">
      <h2>Accounts</h2>
      <ul>
        <li>Passwords are hashed with bcrypt at cost factor 12 and never stored or logged in the clear.</li>
        <li>
          A minimum length of ten characters is required, and passwords appearing on common breach
          lists are rejected. Length is favoured over symbol requirements.
        </li>
        <li>
          Sessions are signed, HTTP-only, same-site cookies. Every session carries a version number,
          so changing your password signs out all other devices immediately.
        </li>
        <li>Sign-in and account creation are rate limited to blunt automated guessing.</li>
      </ul>

      <h2>Authorisation</h2>
      <p>
        Every query that touches health data is scoped by the signed-in user&apos;s id at the data
        layer, not merely filtered in the interface. A record belonging to another account cannot be
        read, edited or deleted even with a valid session and a guessed record id — the request
        simply does not match. This is covered by automated tests that attempt exactly that.
      </p>

      <h2>Input and uploads</h2>
      <ul>
        <li>All request bodies are validated against strict schemas before reaching business logic.</li>
        <li>
          Uploaded files are size-capped and parsed defensively. Nothing from a file is ever executed
          or interpolated into a query; database access goes through a parameterised query builder.
        </li>
        <li>
          The application renders text as text — no HTML from user input or from an AI response is
          ever injected into the page.
        </li>
        <li>
          A restrictive Content Security Policy, frame denial and MIME-sniffing protection are sent
          with every response.
        </li>
      </ul>

      <h2>The assistant</h2>
      <p>
        Model responses are validated against a schema, checked against the evidence they claim to
        cite, and screened for medical-advice and dosing language before anything is displayed. A
        response that fails those checks is replaced, not shown.
      </p>

      <h2>Reporting a problem</h2>
      <p>
        If you believe you have found a security issue, please open a private report on the project
        repository rather than a public issue.
      </p>
    </Prose>
  );
}
