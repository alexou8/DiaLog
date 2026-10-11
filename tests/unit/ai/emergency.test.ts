import { describe, expect, it } from 'vitest';
import { detectEmergency, emergencyAnswer } from '@/lib/ai/emergency';
import { AssistantAnswerSchema } from '@/lib/ai/schemas';

const EMERGENCIES: string[] = [
  "My sugar is 45 and I'm shaking, what do I do?",
  'glucose 2.8 mmol/L right now',
  'Got a reading of 38 mg/dL',
  'my cgm says 50',
  "I'm at 40 and sweating",
  'my sugar dropped from 180 to 45',
  'blood sugar 2,6',
  'I keep passing out after lunch',
  'my husband is unconscious',
  'she is having a seizure',
  "I can't wake him up",
  'I think I have DKA',
  'my ketones are high',
  'I have chest pain',
  "I'm vomiting and can't keep anything down",
  'I took an overdose of insulin',
  'I injected too much insulin',
  'meter reads LO',
  'il est inconscient',
];

const NOT_EMERGENCIES: string[] = [
  'Why are my lows at 2am?',
  'What happened after 30 minutes of walking?',
  'Is my sugar 140 after dinner normal?',
  'My average was 5.8 mmol/L last week',
  'I ate 45 g of carbs at lunch',
  'What happens to my glucose after breakfast?',
  'How did my readings look over the last 14 days?',
  'Do I spike more after 60g carb meals?',
  'My glucose was 7.2 this morning',
  'I take 25 mg of something for blood pressure',
];

describe('detectEmergency', () => {
  it.each(EMERGENCIES)('flags %j', (question) => {
    expect(detectEmergency(question).emergency).toBe(true);
  });

  it.each(NOT_EMERGENCIES)('does not flag %j', (question) => {
    expect(detectEmergency(question).emergency).toBe(false);
  });
});

describe('emergencyAnswer', () => {
  it('is schema-valid, cites nothing and points to emergency services and the care plan', () => {
    const answer = emergencyAnswer();
    expect(AssistantAnswerSchema.safeParse(answer).success).toBe(true);
    expect(answer.citedFindingIds).toEqual([]);
    expect(answer.shortAnswer).toMatch(/emergency number/i);
    expect(answer.detail.join(' ')).toMatch(/care team gave you/i);
  });

  it('contains no amounts of anything', () => {
    const answer = emergencyAnswer();
    expect([answer.shortAnswer, ...answer.detail].join(' ')).not.toMatch(/\d/);
  });
});
