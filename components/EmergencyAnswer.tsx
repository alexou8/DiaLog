import type { AssistantAnswer } from '@/lib/ai/schemas';
import { Alert, AlertDescription, AlertTitle, Icon } from '@/components/ui';

/**
 * The fixed safety template shown when someone's words describe a possible
 * emergency, in the assistant and in quick-log alike, so the two can never
 * disagree.
 *
 * A calm but unmistakable alert: an icon and a heading say what it is in
 * words, so the meaning does not rest on the red border. `role="alert"`
 * announces it once when it appears. Focus is deliberately left where it was:
 * moving it onto a heading inside the alert made screen readers read the
 * safety message twice, once as the alert and again as the focused heading.
 */
export function EmergencyAnswer({
  answer,
  headingId,
  title,
}: {
  answer: AssistantAnswer;
  headingId: string;
  title: string;
}) {
  return (
    <section aria-labelledby={headingId}>
      <Alert tone="critical" role="alert" className="p-5">
        <Icon name="alert" />
        <AlertTitle>
          <h2 id={headingId} className="text-lg">
            {title}
          </h2>
        </AlertTitle>
        <AlertDescription>
          <p className="text-base font-semibold text-ink">{answer.shortAnswer}</p>
          {answer.detail.map((paragraph, index) => (
            <p key={index} className="mt-2 text-ink">
              {paragraph}
            </p>
          ))}
        </AlertDescription>
      </Alert>
    </section>
  );
}
