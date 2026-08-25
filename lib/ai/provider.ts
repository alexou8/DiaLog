/**
 * Provider-independent abstraction over "something that turns a prompt plus
 * a JSON Schema into structured JSON". Every concrete provider (Anthropic,
 * OpenAI, or the offline local provider) implements this same interface so
 * the rest of `lib/ai` never branches on which one is active.
 */
import { AnthropicProvider } from './providers/anthropic';
import { OpenAIProvider } from './providers/openai';
import { LocalProvider } from './providers/local';

export interface CompletionRequestMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface CompletionRequest {
  system: string;
  messages: CompletionRequestMessage[];
  /** JSON Schema the model's structured output must satisfy. */
  responseSchema: object;
  maxTokens: number;
  temperature?: number;
}

export interface CompletionResult {
  json: unknown;
  providerId: string;
  raw?: string;
}

export interface AIProvider {
  id: string;
  name: string;
  /** true when data leaves this deployment (i.e. a hosted third-party API). */
  isExternal: boolean;
  /** Whether this provider is currently usable (e.g. has an API key configured). */
  available(): boolean;
  complete(req: CompletionRequest): Promise<CompletionResult>;
}

export type AIProviderErrorKind =
  | 'timeout'
  | 'http_error'
  | 'malformed_json'
  | 'schema_rejected'
  | 'unavailable'
  | 'unknown';

/** Typed error thrown by provider implementations. Never carries request/response bodies. */
export class AIProviderError extends Error {
  readonly providerId: string;
  readonly kind: AIProviderErrorKind;
  readonly status?: number;

  constructor(providerId: string, kind: AIProviderErrorKind, message: string, status?: number) {
    super(message);
    this.name = 'AIProviderError';
    this.providerId = providerId;
    this.kind = kind;
    this.status = status;
  }
}

export const KNOWN_PROVIDER_IDS = ['anthropic', 'openai', 'local'] as const;
export type KnownProviderId = (typeof KNOWN_PROVIDER_IDS)[number];

function getRegistry(): Record<KnownProviderId, () => AIProvider> {
  return {
    anthropic: () => new AnthropicProvider(),
    openai: () => new OpenAIProvider(),
    local: () => new LocalProvider(),
  };
}

function isKnownProviderId(id: string | undefined): id is KnownProviderId {
  return !!id && (KNOWN_PROVIDER_IDS as readonly string[]).includes(id);
}

/**
 * Select an AI provider. Order of resolution:
 *   1. `preferred` argument, if given and known.
 *   2. `AI_PROVIDER` env var, if set and known.
 *   3. `local` as the ultimate default.
 *
 * Whatever is selected, if it reports itself unavailable (e.g. no API key)
 * we fall back to the local provider, which always works offline and keeps
 * health data on-server.
 */
export function getProvider(preferred?: string): AIProvider {
  const registry = getRegistry();
  const requestedId: KnownProviderId = isKnownProviderId(preferred)
    ? preferred
    : isKnownProviderId(process.env.AI_PROVIDER)
      ? (process.env.AI_PROVIDER as KnownProviderId)
      : 'local';

  const requested = registry[requestedId]();
  if (requested.available()) return requested;
  if (requestedId === 'local') return requested; // local is always "available"; defensive
  return registry.local();
}
