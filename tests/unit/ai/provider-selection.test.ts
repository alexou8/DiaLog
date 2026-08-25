import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { getProvider } from '@/lib/ai/provider';

const ENV_KEYS = ['AI_PROVIDER', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY'] as const;
type EnvKey = (typeof ENV_KEYS)[number];

let savedEnv: Partial<Record<EnvKey, string | undefined>>;

beforeEach(() => {
  savedEnv = {};
  for (const key of ENV_KEYS) {
    savedEnv[key] = process.env[key];
    delete process.env[key];
  }
});

afterEach(() => {
  for (const key of ENV_KEYS) {
    if (savedEnv[key] === undefined) delete process.env[key];
    else process.env[key] = savedEnv[key];
  }
});

describe('getProvider', () => {
  it('defaults to local when nothing is configured', () => {
    const provider = getProvider();
    expect(provider.id).toBe('local');
  });

  it('falls back to local when AI_PROVIDER=anthropic but no API key is set', () => {
    process.env.AI_PROVIDER = 'anthropic';
    const provider = getProvider();
    expect(provider.id).toBe('local');
  });

  it('falls back to local when AI_PROVIDER=openai but no API key is set', () => {
    process.env.AI_PROVIDER = 'openai';
    const provider = getProvider();
    expect(provider.id).toBe('local');
  });

  it('selects anthropic when AI_PROVIDER=anthropic and a key is present', () => {
    process.env.AI_PROVIDER = 'anthropic';
    process.env.ANTHROPIC_API_KEY = 'sk-ant-test';
    const provider = getProvider();
    expect(provider.id).toBe('anthropic');
    expect(provider.isExternal).toBe(true);
  });

  it('selects openai when AI_PROVIDER=openai and a key is present', () => {
    process.env.AI_PROVIDER = 'openai';
    process.env.OPENAI_API_KEY = 'sk-oa-test';
    const provider = getProvider();
    expect(provider.id).toBe('openai');
    expect(provider.isExternal).toBe(true);
  });

  it('an explicit preferred argument overrides the env var', () => {
    process.env.AI_PROVIDER = 'local';
    process.env.ANTHROPIC_API_KEY = 'sk-ant-test';
    const provider = getProvider('anthropic');
    expect(provider.id).toBe('anthropic');
  });

  it('falls back to local for an unknown preferred id', () => {
    const provider = getProvider('made-up-provider');
    expect(provider.id).toBe('local');
  });

  it('falls back to local when preferred is a known id but unavailable', () => {
    const provider = getProvider('openai');
    expect(provider.id).toBe('local');
  });
});
