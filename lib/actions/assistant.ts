'use server';

import { prisma } from '@/lib/db/prisma';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { audit } from '@/lib/auth/audit';
import { RATE_LIMITS, rateLimit } from '@/lib/auth/rate-limit';
import { analyzeUser, defaultWindow, toEvidenceBundle } from '@/lib/services/analytics-service';
import { answerQuestion, parseMealText } from '@/lib/ai/pipeline';
import { getProvider } from '@/lib/ai/provider';
import { redactForProvider } from '@/lib/ai/redact';
import type { AssistantAnswer, MealParse } from '@/lib/ai/schemas';
import { assistantQuestionSchema, fieldErrors, quickLogSchema } from '@/lib/validation';

export interface AssistantState {
  ok: boolean;
  message?: string;
  errors?: Record<string, string>;
  question?: string;
  answer?: AssistantAnswer;
  providerId?: string;
  usedFallback?: boolean;
  /** Findings the answer is allowed to cite, so the UI can show the evidence. */
  citedFindings?: {
    id: string;
    statement: string;
    sampleSize: number;
    basis: string;
    evidenceLevel: string;
  }[];
}

const DETAIL_BY_PROFILE = { SIMPLE: 'simple', STANDARD: 'standard', DETAILED: 'detailed' } as const;

export async function askAssistantAction(
  _prev: AssistantState | null,
  formData: FormData,
): Promise<AssistantState> {
  const user = await requireOnboardedUser();

  if (!user.profile.aiEnabled) {
    return {
      ok: false,
      message:
        'The assistant is switched off for your account. You can turn it back on in Settings.',
    };
  }

  const limit = rateLimit(`ai:${user.id}`, RATE_LIMITS.ai.limit, RATE_LIMITS.ai.windowMs);
  if (!limit.ok) {
    return {
      ok: false,
      message: 'You have asked a lot of questions in a short time. Please try again shortly.',
    };
  }

  const parsed = assistantQuestionSchema.safeParse({
    question: formData.get('question'),
    detailLevel: formData.get('detailLevel') ?? DETAIL_BY_PROFILE[user.profile.detailLevel],
  });
  if (!parsed.success) return { ok: false, errors: fieldErrors(parsed.error) };

  const window = defaultWindow(90);
  const { result } = await analyzeUser(user.id, user.profile, window);
  const bundle = toEvidenceBundle(result, user.profile, window);

  const provider = getProvider(process.env.AI_PROVIDER);
  // Free-text is stripped before anything reaches an external provider unless
  // the user has explicitly consented.
  const redaction = redactForProvider(
    bundle,
    provider.isExternal,
    Boolean(user.profile.externalAiConsentAt),
  );

  const outcome = await answerQuestion({
    question: parsed.data.question,
    bundle: redaction.bundle,
    detailLevel: parsed.data.detailLevel,
    provider,
  });

  const cited = outcome.answer.citedFindingIds
    .map((id) => bundle.findings.find((finding) => finding.id === id))
    .filter((finding): finding is NonNullable<typeof finding> => finding != null)
    .map((finding) => ({
      id: finding.id,
      statement: finding.statement,
      sampleSize: finding.sampleSize,
      basis: finding.basis,
      evidenceLevel: finding.evidenceLevel,
    }));

  // Conversations are stored so the user can look back at what they asked.
  // The evidence used is stored alongside, so an old answer can still be audited.
  const conversation = await prisma.aIConversation
    .upsert({
      where: { id: formData.get('conversationId')?.toString() ?? '__none__' },
      create: { userId: user.id, title: parsed.data.question.slice(0, 80) },
      update: {},
    })
    .catch(() =>
      prisma.aIConversation.create({
        data: { userId: user.id, title: parsed.data.question.slice(0, 80) },
      }),
    );

  await prisma.aIMessage.createMany({
    data: [
      { conversationId: conversation.id, role: 'user', content: parsed.data.question },
      {
        conversationId: conversation.id,
        role: 'assistant',
        content: outcome.answer.shortAnswer,
        evidence: { findings: cited },
        providerId: outcome.providerId,
      },
    ],
  });

  await audit({ userId: user.id, action: 'ai.question', detail: outcome.providerId });

  return {
    ok: true,
    question: parsed.data.question,
    answer: outcome.answer,
    providerId: outcome.providerId,
    usedFallback: outcome.usedFallback,
    citedFindings: cited,
  };
}

export interface QuickLogState {
  ok: boolean;
  message?: string;
  errors?: Record<string, string>;
  original?: string;
  proposal?: MealParse;
  providerId?: string;
}

/**
 * Turn a sentence into proposed records. Nothing is saved here — the proposal
 * is returned for the user to check and edit first.
 */
export async function proposeQuickLogAction(
  _prev: QuickLogState | null,
  formData: FormData,
): Promise<QuickLogState> {
  const user = await requireOnboardedUser();
  if (!user.profile.aiEnabled) {
    return {
      ok: false,
      message: 'Quick logging uses the assistant, which is switched off for your account.',
    };
  }

  const limit = rateLimit(`ai:${user.id}`, RATE_LIMITS.ai.limit, RATE_LIMITS.ai.windowMs);
  if (!limit.ok) return { ok: false, message: 'Please wait a moment before trying again.' };

  const parsed = quickLogSchema.safeParse({ text: formData.get('text') });
  if (!parsed.success) return { ok: false, errors: fieldErrors(parsed.error) };

  const provider = getProvider(process.env.AI_PROVIDER);
  const outcome = await parseMealText({ text: parsed.data.text, provider });

  await audit({ userId: user.id, action: 'ai.quick_log_parse', detail: outcome.providerId });

  return {
    ok: true,
    original: parsed.data.text,
    proposal: outcome.parse,
    providerId: outcome.providerId,
  };
}
