import type { Metadata } from 'next';
import { Prose } from '../_prose';

export const metadata: Metadata = {
  title: 'Accessibility',
  description: 'DiaLog’s accessibility commitment and the specific things it does.',
};

export default function AccessibilityPage() {
  return (
    <Prose title="Accessibility">
      <p>
        DiaLog targets WCAG 2.2 level AA. Accessibility was part of the design rather than a pass at
        the end, because the people most likely to be managing glucose every day are also the people
        most likely to be doing it with imperfect eyesight, unsteady hands, or a screen reader.
      </p>

      <h2>What that means in practice</h2>
      <ul>
        <li>Large default type, with a larger-text setting that scales the whole interface.</li>
        <li>
          Every control reachable and operable by keyboard, with a focus ring that is never removed.
        </li>
        <li>
          Semantic headings, landmarks and labels, so screen reader navigation works properly.
        </li>
        <li>
          Status is never conveyed by colour alone. A reading outside your range always carries a
          word (&ldquo;Above your target range&rdquo;) and a shape or icon as well as a colour.
        </li>
        <li>
          Every chart has a text summary and a data table containing the same numbers, so no
          information exists only as a picture.
        </li>
        <li>Touch targets of at least 44 pixels, and forms kept short with sensible defaults.</li>
        <li>
          Motion is minimal, and the reduced-motion setting is respected at both OS and app level.
        </li>
        <li>Errors are written in plain language and are announced to assistive technology.</li>
      </ul>

      <h2>Known limitations</h2>
      <ul>
        <li>
          The French translation currently covers navigation and shared interface text only; page
          content falls back to English.
        </li>
        <li>
          Automated checks (axe) run against the main journeys, but automated testing catches only
          part of what matters. Reports from people using assistive technology are very welcome.
        </li>
      </ul>

      <h2>Feedback</h2>
      <p>
        If something here does not work for you, please open an issue on the project repository
        describing what you were trying to do and what got in the way.
      </p>
    </Prose>
  );
}
