import type { Metadata } from 'next';
import { Prose } from '../_prose';

export const metadata: Metadata = {
  title: 'About DiaLog',
  description: 'How DiaLog turns your own health records into plain-language observations.',
};

export default function AboutPage() {
  return (
    <Prose title="About DiaLog">
      <p>
        Most diabetes software is either a spreadsheet with charts bolted on, or a clinical tool
        designed for the person reading the report rather than the person living the data. DiaLog is
        built for the second person.
      </p>

      <h2>How it works</h2>
      <p>The pipeline is deliberately boring in the places where being boring is safer:</p>
      <ul>
        <li>
          <strong>Your records are normalised.</strong> Whatever the source — a meter export, a
          spreadsheet, a form you filled in — everything becomes the same kind of timestamped event,
          with its origin recorded so you can always trace a reading back to where it came from.
        </li>
        <li>
          <strong>Statistics do the measuring.</strong> Averages, medians, spread, the share of
          readings in your target range, week-over-week comparisons and correlations are ordinary
          arithmetic, not AI. They are computed the same way every time and you can check them.
        </li>
        <li>
          <strong>Machine learning does the pattern-finding.</strong> Detecting readings that are
          unusual <em>for you</em>, grouping your days into recurring patterns, and ranking which of
          your logged factors move together with your glucose.
        </li>
        <li>
          <strong>Language models only explain.</strong> The assistant never sees your raw records.
          It receives a structured set of findings — each with its sample size and confidence — and
          its job is to put them into readable sentences. It cannot invent a number that the
          analysis did not produce.
        </li>
      </ul>

      <h2>Why the sample sizes are everywhere</h2>
      <p>
        A pattern found in eight readings is a coincidence with good marketing. DiaLog grades every
        observation as <em>not enough data</em>, <em>early signal</em>, <em>emerging pattern</em> or{' '}
        <em>consistent pattern</em>, based on thresholds set per type of analysis, and it shows you
        which one applies. Being told &ldquo;there isn&apos;t enough data to answer that yet&rdquo; is
        a feature, not a failure.
      </p>

      <h2>What it is not</h2>
      <p>
        DiaLog is not a medical device and has not been assessed by any regulator. It does not
        diagnose, does not calculate or recommend doses, and does not tell you to change anything
        about your treatment. Medication logging exists so you can see your own timing next to your
        own readings — that is medication <em>tracking</em>, which is a different thing from
        medication management.
      </p>

      <h2>Devices</h2>
      <p>
        Rather than claiming integrations that do not exist, DiaLog documents honestly what each
        manufacturer actually makes available, and builds excellent file import for the exports you
        can really get today. See the device integration notes in the project documentation.
      </p>
    </Prose>
  );
}
