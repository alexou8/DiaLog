/**
 * Adapts the JSON Schemas in `lib/ai/schemas.ts` to what each provider's
 * structured-output mode accepts.
 *
 * The schemas carry length, count and range caps because zod enforces them
 * on the way back (`schemas.ts`), but a provider's constrained-decoding mode
 * may reject a keyword it does not support with a 400, which the pipeline
 * turns into a silent local fallback. Anthropic structured outputs reject
 * numeric, string-length and array-count constraints and `pattern`; OpenAI
 * strict mode rejects `minLength`/`maxLength`. Stripping them here keeps one
 * source schema while every cap is still enforced by zod.
 *
 * Only schema keywords are touched: a property that happens to be *named*
 * "pattern" or "minimum" is left alone, because `properties` maps are walked
 * by value, not filtered by key.
 */

type JsonValue = unknown;

export interface SchemaCompatOptions {
  /** Keywords to remove wherever they appear as schema keywords. */
  strip: readonly string[];
  /** Rewrite `type: ['string', 'null']` as `anyOf` (the form both providers document). */
  typeArraysToAnyOf?: boolean;
}

export const ANTHROPIC_UNSUPPORTED_KEYWORDS = [
  'minimum',
  'maximum',
  'exclusiveMinimum',
  'exclusiveMaximum',
  'multipleOf',
  'minLength',
  'maxLength',
  'minItems',
  'maxItems',
  'pattern',
] as const;

export const OPENAI_UNSUPPORTED_KEYWORDS = ['minLength', 'maxLength'] as const;

const SUBSCHEMA_MAP_KEYS = new Set(['properties', '$defs', 'definitions']);
const SUBSCHEMA_LIST_KEYS = new Set(['anyOf', 'allOf', 'oneOf', 'prefixItems']);
const SUBSCHEMA_KEYS = new Set(['items', 'additionalProperties', 'not']);

export function adaptSchema(schema: JsonValue, options: SchemaCompatOptions): JsonValue {
  const strip = new Set(options.strip);

  function walk(node: JsonValue): JsonValue {
    if (Array.isArray(node)) return node.map(walk);
    if (node === null || typeof node !== 'object') return node;

    const out: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(node as Record<string, unknown>)) {
      if (strip.has(key)) continue;
      if (SUBSCHEMA_MAP_KEYS.has(key) && value && typeof value === 'object') {
        out[key] = Object.fromEntries(
          Object.entries(value as Record<string, unknown>).map(([k, v]) => [k, walk(v)]),
        );
      } else if (SUBSCHEMA_LIST_KEYS.has(key) && Array.isArray(value)) {
        out[key] = value.map(walk);
      } else if (SUBSCHEMA_KEYS.has(key)) {
        out[key] = walk(value);
      } else {
        out[key] = value;
      }
    }

    if (options.typeArraysToAnyOf && Array.isArray(out.type)) {
      const { type, ...rest } = out;
      return { ...rest, anyOf: (type as unknown[]).map((t) => ({ type: t })) };
    }
    return out;
  }

  return walk(schema);
}
