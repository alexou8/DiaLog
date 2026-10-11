import { describe, expect, it } from 'vitest';
import {
  AssistantAnswerJsonSchema,
  AssistantAnswerSchema,
  MealParseJsonSchema,
  NoteStructureJsonSchema,
  NoteStructureSchema,
  WeeklyNarrativeJsonSchema,
} from '@/lib/ai/schemas';
import {
  adaptSchema,
  ANTHROPIC_UNSUPPORTED_KEYWORDS,
  OPENAI_UNSUPPORTED_KEYWORDS,
} from '@/lib/ai/providers/schema-compat';

const ALL_SCHEMAS = [
  AssistantAnswerJsonSchema,
  MealParseJsonSchema,
  WeeklyNarrativeJsonSchema,
  NoteStructureJsonSchema,
];

/** Every object node in a JSON Schema, found by walking the schema keywords. */
function objectNodes(node: unknown): Record<string, unknown>[] {
  if (Array.isArray(node)) return node.flatMap(objectNodes);
  if (!node || typeof node !== 'object') return [];
  const record = node as Record<string, unknown>;
  const own = record.type === 'object' ? [record] : [];
  return [...own, ...Object.values(record).flatMap(objectNodes)];
}

function keywordsIn(node: unknown): string[] {
  if (Array.isArray(node)) return node.flatMap(keywordsIn);
  if (!node || typeof node !== 'object') return [];
  return Object.entries(node as Record<string, unknown>).flatMap(([k, v]) => [k, ...keywordsIn(v)]);
}

describe('structured-output schemas', () => {
  it.each(ALL_SCHEMAS.map((s, i) => [i, s] as const))(
    'schema %i: every object is closed and requires every property (OpenAI strict mode)',
    (_i, schema) => {
      for (const node of objectNodes(schema)) {
        expect(node.additionalProperties).toBe(false);
        const props = Object.keys((node.properties as object) ?? {});
        expect([...((node.required as string[]) ?? [])].sort()).toEqual(props.sort());
      }
    },
  );

  it('strips the keywords Anthropic structured outputs reject, and type arrays', () => {
    for (const schema of ALL_SCHEMAS) {
      const adapted = adaptSchema(schema, {
        strip: ANTHROPIC_UNSUPPORTED_KEYWORDS,
        typeArraysToAnyOf: true,
      });
      const keywords = keywordsIn(adapted);
      for (const banned of ANTHROPIC_UNSUPPORTED_KEYWORDS) expect(keywords).not.toContain(banned);
      expect(JSON.stringify(adapted)).not.toMatch(/"type":\[/);
    }
  });

  it('strips only string-length keywords for OpenAI', () => {
    const adapted = adaptSchema(MealParseJsonSchema, { strip: OPENAI_UNSUPPORTED_KEYWORDS });
    const keywords = keywordsIn(adapted);
    expect(keywords).not.toContain('maxLength');
    expect(keywords).toContain('maxItems');
    expect(keywords).toContain('minimum');
  });

  it('never strips a property that happens to share a keyword name', () => {
    const schema = {
      type: 'object',
      additionalProperties: false,
      properties: { pattern: { type: 'string', maxLength: 5 } },
      required: ['pattern'],
    };
    const adapted = adaptSchema(schema, { strip: ANTHROPIC_UNSUPPORTED_KEYWORDS }) as {
      properties: Record<string, unknown>;
    };
    expect(adapted.properties.pattern).toEqual({ type: 'string' });
  });
});

describe('schema caps', () => {
  const answer = {
    shortAnswer: 'Your data shows a mild rise after breakfast.',
    detail: ['detail line'],
    citedFindingIds: ['f1'],
    confidence: 'moderate',
    suggestedQuestionsForClinician: [],
    notEnoughData: false,
  };

  it('rejects an empty shortAnswer', () => {
    expect(AssistantAnswerSchema.safeParse({ ...answer, shortAnswer: '' }).success).toBe(false);
  });

  it('rejects an unbounded detail list', () => {
    const detail = Array.from({ length: 50 }, () => 'line');
    expect(AssistantAnswerSchema.safeParse({ ...answer, detail }).success).toBe(false);
  });

  it('rejects a free-form fields record on a note candidate', () => {
    const bad = {
      candidates: [{ type: 'meal', summary: 'toast', fields: { carbs: 20 }, confidence: 'low' }],
      unparsed: [],
    };
    expect(NoteStructureSchema.safeParse(bad).success).toBe(false);
  });
});
