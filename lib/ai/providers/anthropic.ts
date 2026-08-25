/**
 * Anthropic provider, implemented with plain `fetch` — no `@anthropic-ai/sdk`
 * dependency (per project constraints: no new npm packages).
 *
 * Structured JSON output is obtained by giving Claude a single tool whose
 * `input_schema` is the caller's JSON Schema, and forcing its use with
 * `tool_choice: { type: 'tool', name: ... }`. The tool's `input` is then the
 * structured JSON we want — this is the documented pattern for constrained
 * output on the Messages API.
 *
 * Never logs request or response bodies (they contain health data) — only
 * status codes and durations.
 */
import {
  AIProviderError,
  type AIProvider,
  type CompletionRequest,
  type CompletionResult,
} from '../provider';

const ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages';
const ANTHROPIC_VERSION = '2023-06-01';
const REQUEST_TIMEOUT_MS = 30_000;
const STRUCTURED_TOOL_NAME = 'emit_structured_output';

export class AnthropicProvider implements AIProvider {
  readonly id = 'anthropic';
  readonly name = 'Anthropic Claude';
  readonly isExternal = true;

  private apiKey(): string | undefined {
    return process.env.ANTHROPIC_API_KEY;
  }

  private model(): string {
    return process.env.ANTHROPIC_MODEL ?? 'claude-sonnet-5';
  }

  available(): boolean {
    return !!this.apiKey();
  }

  async complete(req: CompletionRequest): Promise<CompletionResult> {
    const apiKey = this.apiKey();
    if (!apiKey) {
      throw new AIProviderError(this.id, 'unavailable', 'ANTHROPIC_API_KEY is not configured');
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    const startedAt = Date.now();

    let response: Response;
    try {
      response = await fetch(ANTHROPIC_API_URL, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': ANTHROPIC_VERSION,
        },
        body: JSON.stringify({
          model: this.model(),
          max_tokens: req.maxTokens,
          temperature: req.temperature,
          system: req.system,
          messages: req.messages.map((m) => ({ role: m.role, content: m.content })),
          tools: [
            {
              name: STRUCTURED_TOOL_NAME,
              description: 'Emit the structured JSON response. This tool must always be used.',
              input_schema: req.responseSchema,
            },
          ],
          tool_choice: { type: 'tool', name: STRUCTURED_TOOL_NAME },
        }),
        signal: controller.signal,
      });
    } catch (err) {
      const durationMs = Date.now() - startedAt;
      if (err instanceof Error && err.name === 'AbortError') {
        console.error(`[ai:anthropic] request timed out after ${durationMs}ms`);
        throw new AIProviderError(this.id, 'timeout', 'Anthropic request timed out');
      }
      console.error(`[ai:anthropic] request failed after ${durationMs}ms`);
      throw new AIProviderError(this.id, 'unknown', 'Anthropic request failed');
    } finally {
      clearTimeout(timeout);
    }

    const durationMs = Date.now() - startedAt;
    console.info(`[ai:anthropic] status=${response.status} durationMs=${durationMs}`);

    if (!response.ok) {
      throw new AIProviderError(
        this.id,
        'http_error',
        `Anthropic API returned status ${response.status}`,
        response.status,
      );
    }

    let body: unknown;
    try {
      body = await response.json();
    } catch {
      throw new AIProviderError(this.id, 'malformed_json', 'Anthropic response was not valid JSON');
    }

    const toolInput = extractToolInput(body);
    if (toolInput === undefined) {
      throw new AIProviderError(
        this.id,
        'malformed_json',
        'Anthropic response did not contain a structured tool call',
      );
    }

    return { json: toolInput, providerId: this.id };
  }
}

interface AnthropicContentBlock {
  type: string;
  input?: unknown;
  name?: string;
}

interface AnthropicMessageResponse {
  content?: AnthropicContentBlock[];
}

function extractToolInput(body: unknown): unknown {
  if (!body || typeof body !== 'object') return undefined;
  const content = (body as AnthropicMessageResponse).content;
  if (!Array.isArray(content)) return undefined;
  const toolUse = content.find(
    (block) => block.type === 'tool_use' && block.name === STRUCTURED_TOOL_NAME,
  );
  return toolUse?.input;
}
