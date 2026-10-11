/**
 * Action-level consent tests (review C1/C2, 2026-10): with AI_PROVIDER set to
 * an external provider and a key configured, a user who has not consented
 * must be answered locally, and `fetch` must never be called. Everything
 * the server actions touch besides `lib/ai` is mocked.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { makeBundle } from './fixtures';

const state = vi.hoisted(() => ({
  consentAt: null as Date | null,
  email: 'person@example.com',
}));

vi.mock('@/lib/auth/current-user', () => ({
  requireOnboardedUser: async () => ({
    id: 'user-1',
    email: state.email,
    profile: {
      aiEnabled: true,
      detailLevel: 'STANDARD',
      externalAiConsentAt: state.consentAt,
    },
  }),
}));
vi.mock('@/lib/auth/rate-limit', () => ({
  RATE_LIMITS: { ai: { limit: 100, windowMs: 60_000 } },
  rateLimit: () => ({ ok: true }),
}));
vi.mock('@/lib/auth/audit', () => ({ audit: vi.fn(async () => undefined) }));
vi.mock('@/lib/db/prisma', () => ({
  prisma: {
    aIConversation: {
      findFirst: vi.fn(async () => null),
      create: vi.fn(async () => ({ id: 'conversation-1' })),
    },
    aIMessage: { createMany: vi.fn(async () => ({ count: 2 })) },
  },
}));
vi.mock('@/lib/services/analytics-service', () => ({
  defaultWindow: () => ({ start: new Date(0), end: new Date(0) }),
  analyzeUser: async () => ({ result: {} }),
  toEvidenceBundle: () => makeBundle(),
}));

import { askAssistantAction, proposeQuickLogAction } from '@/lib/actions/assistant';

const ENV_KEYS = ['AI_PROVIDER', 'ANTHROPIC_API_KEY', 'ANTHROPIC_MODEL'] as const;
const savedEnv: Partial<Record<(typeof ENV_KEYS)[number], string | undefined>> = {};
const ORIGINAL_FETCH = global.fetch;
let fetchMock: ReturnType<typeof vi.fn>;

function anthropicReply(payload: unknown): Response {
  return new Response(
    JSON.stringify({
      stop_reason: 'end_turn',
      content: [{ type: 'text', text: JSON.stringify(payload) }],
    }),
    { status: 200 },
  );
}

beforeEach(() => {
  for (const key of ENV_KEYS) savedEnv[key] = process.env[key];
  process.env.AI_PROVIDER = 'anthropic';
  process.env.ANTHROPIC_API_KEY = 'sk-ant-test-key';
  delete process.env.ANTHROPIC_MODEL;
  vi.spyOn(console, 'info').mockImplementation(() => {});
  fetchMock = vi.fn(async (_url: unknown, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body)) as {
      output_config: { format: { schema: { properties: object } } };
    };
    const props = Object.keys(body.output_config.format.schema.properties);
    return props.includes('meals')
      ? anthropicReply({ meals: [], exercise: [], unparsed: ['toast and jam'] })
      : anthropicReply({
          shortAnswer: 'Your data shows a mild rise after breakfast.',
          detail: ['Your data shows a mild rise after breakfast.'],
          citedFindingIds: ['finding-1'],
          confidence: 'moderate',
          suggestedQuestionsForClinician: [],
          notEnoughData: false,
        });
  });
  global.fetch = fetchMock as unknown as typeof fetch;
});

afterEach(() => {
  for (const key of ENV_KEYS) {
    if (savedEnv[key] === undefined) delete process.env[key];
    else process.env[key] = savedEnv[key];
  }
  global.fetch = ORIGINAL_FETCH;
  state.consentAt = null;
  state.email = 'person@example.com';
  vi.restoreAllMocks();
});

function form(fields: Record<string, string>): FormData {
  const data = new FormData();
  for (const [k, v] of Object.entries(fields)) data.set(k, v);
  return data;
}

describe('askAssistantAction consent', () => {
  it('answers locally and never calls fetch without consent', async () => {
    const result = await askAssistantAction(
      null,
      form({ question: 'I took 20u Lantus, what happens after breakfast?' }),
    );
    expect(result.ok).toBe(true);
    expect(result.providerId).toBe('local');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('calls the external provider once the user has consented', async () => {
    state.consentAt = new Date('2026-01-01T00:00:00Z');
    const result = await askAssistantAction(
      null,
      form({ question: 'What happens after breakfast?' }),
    );
    expect(result.ok).toBe(true);
    expect(result.providerId).toBe('anthropic');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('answers an emergency question with the safety template and no provider call', async () => {
    state.consentAt = new Date('2026-01-01T00:00:00Z');
    const result = await askAssistantAction(
      null,
      form({ question: 'my sugar is 40 and I feel faint' }),
    );
    expect(result.emergency).toBe(true);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe('proposeQuickLogAction consent', () => {
  it('parses locally and never calls fetch without consent', async () => {
    const result = await proposeQuickLogAction(null, form({ text: 'toast and jam' }));
    expect(result.ok).toBe(true);
    expect(result.providerId).toBe('local');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('calls the external provider once the user has consented', async () => {
    state.consentAt = new Date('2026-01-01T00:00:00Z');
    const result = await proposeQuickLogAction(null, form({ text: 'toast and jam' }));
    expect(result.providerId).toBe('anthropic');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe('proposeQuickLogAction emergency triage', () => {
  it('surfaces the safety template and never calls the provider', async () => {
    state.consentAt = new Date('2026-01-01T00:00:00Z');
    const result = await proposeQuickLogAction(
      null,
      form({ text: 'BG 2.8 and shaking, had juice' }),
    );
    expect(result.ok).toBe(true);
    expect(result.emergency).toBe(true);
    expect(result.emergencyAnswer?.shortAnswer).toMatch(/emergency/i);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe('the shared demo account', () => {
  // One visitor's stored consent must not cover the next visitor's free text.
  it('never reaches an external provider, even with consent on its row', async () => {
    state.email = 'demo@dialog.health';
    state.consentAt = new Date('2026-01-01T00:00:00Z');
    const asked = await askAssistantAction(null, form({ question: 'What happens after lunch?' }));
    expect(asked.providerId).toBe('local');
    const logged = await proposeQuickLogAction(null, form({ text: 'toast and jam' }));
    expect(logged.providerId).toBe('local');
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
