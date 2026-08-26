import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Class composer used by the shadcn primitives. `twMerge` is what lets a call
 * site override a variant's utility (say, padding) without having to know
 * which class the variant already set — the later class wins instead of both
 * landing in the string and the cascade deciding arbitrarily.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
