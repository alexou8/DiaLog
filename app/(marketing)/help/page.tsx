import type { Metadata } from 'next';
import { Prose } from '../_prose';

export const metadata: Metadata = { title: 'Help', description: 'Getting started with DiaLog.' };

export default function HelpPage() {
  return (
    <Prose title="Help and getting started">
      <h2>Getting your readings in</h2>
      <p>There are three ways, and you can mix them freely:</p>
      <ul>
        <li>
          <strong>Type them in.</strong> Home → Add a reading. Value, time and (optionally) whether
          it was fasting or after a meal. That is the whole form.
        </li>
        <li>
          <strong>Import a file.</strong> Import → drag in the CSV, Excel, JSON or XML file your
          meter&apos;s software or your phone&apos;s health app exports. DiaLog works out the format,
          shows you what it found, and asks before saving anything.
        </li>
        <li>
          <strong>Describe your day.</strong> Type &ldquo;had a burger and fries around 7, walked 20
          minutes after&rdquo; and DiaLog proposes structured entries for you to check and edit
          before they are saved. Nothing is stored until you confirm it.
        </li>
      </ul>

      <h2>Why does it say &ldquo;not enough data yet&rdquo;?</h2>
      <p>
        Because it is true. Each kind of comparison needs a minimum number of your records before the
        result means anything. The message tells you how many you have and how many are needed, so you
        know what to log next.
      </p>

      <h2>Changing units</h2>
      <p>
        Settings → Units. Switching between mmol/L and mg/dL converts everything you see instantly;
        your stored readings are untouched, because DiaLog keeps one canonical unit internally and
        converts only for display.
      </p>

      <h2>Duplicate imports</h2>
      <p>
        Importing the same export twice is safe. DiaLog fingerprints each record and skips ones you
        already have, then tells you how many it skipped.
      </p>

      <h2>Installing on your phone</h2>
      <p>
        Open DiaLog in your phone&apos;s browser and choose &ldquo;Add to Home Screen&rdquo;. It then
        opens like an app, full screen. For your privacy it does not keep your readings on the device,
        so it needs a connection to show them.
      </p>

      <h2>Something looks wrong</h2>
      <p>
        Every record can be edited or deleted from History, and every imported record shows which file
        it came from. If an import went wrong, you can remove the whole batch in one step.
      </p>
    </Prose>
  );
}
