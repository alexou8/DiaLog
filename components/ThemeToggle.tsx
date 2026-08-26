'use client';

import { useEffect, useId, useState } from 'react';

import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

type Theme = 'light' | 'dark' | 'system';

/**
 * Colour scheme is a device-level choice, so it lives in localStorage rather
 * than the account profile. The initial value is applied before paint by
 * PreferencesScript.
 *
 * This is one of the few controls that can safely be shadcn's Radix Select
 * rather than a native <select>: it changes local state on the client and
 * never posts a form, so there is no value to lose to a submit that beats
 * hydration. The account-level preferences in Settings use the native control
 * for exactly that reason.
 */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>('system');
  const id = useId();

  useEffect(() => {
    const stored = localStorage.getItem('dialog-theme');
    if (stored === 'light' || stored === 'dark' || stored === 'system') setTheme(stored);
  }, []);

  function apply(next: Theme) {
    setTheme(next);
    localStorage.setItem('dialog-theme', next);
    const resolved =
      next === 'system'
        ? window.matchMedia('(prefers-color-scheme: dark)').matches
          ? 'dark'
          : 'light'
        : next;
    document.documentElement.dataset.theme = resolved;
  }

  return (
    <div className="flex items-center gap-2">
      <Label htmlFor={id} className="text-sm">
        Appearance
      </Label>
      <Select value={theme} onValueChange={(next) => apply(next as Theme)}>
        <SelectTrigger id={id} className="w-auto min-w-44">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="system">Match my device</SelectItem>
          <SelectItem value="light">Light</SelectItem>
          <SelectItem value="dark">Dark</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
}
