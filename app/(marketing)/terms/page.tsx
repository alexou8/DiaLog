import type { Metadata } from 'next';
import { Prose } from '../_prose';

export const metadata: Metadata = { title: 'Terms of use' };

export default function TermsPage() {
  return (
    <Prose title="Terms of use" updated="August 2026">
      <h2>What DiaLog is</h2>
      <p>
        DiaLog is a personal health-data organisation tool provided for informational and educational
        purposes. It is not a medical device, has not been reviewed or approved by Health Canada, the
        FDA or any other regulator, and makes no claim of clinical validity.
      </p>

      <h2>Not medical advice</h2>
      <p>
        Nothing in DiaLog — including observations, summaries and anything the assistant writes — is
        medical advice, diagnosis or treatment. Do not use DiaLog to make decisions about medication,
        insulin, diet or treatment. Always consult a qualified healthcare professional. In an
        emergency, contact your local emergency services.
      </p>

      <h2>Accuracy</h2>
      <p>
        DiaLog reports what your records contain. If a reading was entered incorrectly, imported from
        a device with a wrong clock, or is missing entirely, the analysis will reflect that. Observations
        are statistical associations found in your own data; association is not causation.
      </p>

      <h2>Your account</h2>
      <p>
        You are responsible for keeping your password secure. You retain ownership of the data you
        enter, and may export or delete it at any time.
      </p>

      <h2>Availability and liability</h2>
      <p>
        The service is provided &ldquo;as is&rdquo;, without warranty of any kind. To the maximum
        extent permitted by law, the authors are not liable for any loss arising from the use of, or
        inability to use, DiaLog.
      </p>
    </Prose>
  );
}
