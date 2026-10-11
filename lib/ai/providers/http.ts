/**
 * The one HTTP call shape both hosted providers share: POST a JSON body and
 * read a JSON body, all under a single timeout.
 *
 * The timeout covers the response body, not just the headers. It used to be
 * cleared in a `finally` as soon as `fetch` resolved, so a server that sent
 * headers and then stalled the body left `response.json()` unbounded and hung
 * the server action (review L1).
 *
 * Logs only the status code and duration. Request and response bodies carry
 * health data and user free text, so they are never logged or put in an error.
 */
import { AIProviderError } from '../provider';

export interface PostJsonResult {
  status: number;
  ok: boolean;
  /** The parsed body, or undefined when it was empty or not JSON (see `parsed`). */
  body: unknown;
  parsed: boolean;
}

export async function postJson(options: {
  providerId: string;
  /** Short name used in log lines and error messages, e.g. "Anthropic". */
  displayName: string;
  url: string;
  headers: Record<string, string>;
  payload: unknown;
  timeoutMs: number;
}): Promise<PostJsonResult> {
  const { providerId, displayName, url, headers, payload, timeoutMs } = options;
  const logTag = `[ai:${providerId}]`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const startedAt = Date.now();

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    // Still inside the timeout: an aborted body read rejects with AbortError.
    const text = await response.text();
    const durationMs = Date.now() - startedAt;
    console.info(`${logTag} status=${response.status} durationMs=${durationMs}`);

    let body: unknown;
    let parsed = false;
    if (text.length > 0) {
      try {
        body = JSON.parse(text);
        parsed = true;
      } catch {
        body = undefined;
      }
    }
    return { status: response.status, ok: response.ok, body, parsed };
  } catch (err) {
    const durationMs = Date.now() - startedAt;
    if ((err as { name?: unknown } | null)?.name === 'AbortError') {
      console.error(`${logTag} request timed out after ${durationMs}ms`);
      throw new AIProviderError(providerId, 'timeout', `${displayName} request timed out`);
    }
    console.error(`${logTag} request failed after ${durationMs}ms`);
    throw new AIProviderError(providerId, 'unknown', `${displayName} request failed`);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * The provider's machine-readable error type (e.g. "invalid_request_error"),
 * safe to log because it is an enum-like code, never message text. A
 * systematic 400 used to be invisible: the pipeline fell back to local and the
 * only trace was a status code (review H1).
 */
export function safeErrorType(body: unknown): string | null {
  if (!body || typeof body !== 'object') return null;
  const error = (body as { error?: unknown }).error;
  const type = error && typeof error === 'object' ? (error as { type?: unknown }).type : undefined;
  return typeof type === 'string' && /^[a-z_]{1,64}$/.test(type) ? type : null;
}
