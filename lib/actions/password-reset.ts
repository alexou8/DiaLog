'use server';

import { headers } from 'next/headers';
import type { ActionState } from './auth';
import { requestPasswordReset } from '@/lib/auth/password-reset';

export async function forgotPasswordAction(
  _previous: ActionState | null,
  form: FormData,
): Promise<ActionState> {
  const h = await headers();
  const ip = h.get('x-forwarded-for')?.split(',')[0]?.trim() ?? h.get('x-real-ip') ?? 'unknown';
  return requestPasswordReset(Object.fromEntries(form), ip);
}
