'use client';

import { useEffect, useState } from 'react';

type Theme = 'light' | 'dark' | 'system';

/**
 * Colour scheme is a device-level choice, so it lives in localStorage rather
 * than the account profile. The initial value is applied before paint by
 * PreferencesScript.
 */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>('system');

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
      <label htmlFor="dl-theme" className="text-sm font-medium">
        Appearance
      </label>
      <select
        id="dl-theme"
        value={theme}
        onChange={(event) => apply(event.target.value as Theme)}
        className="rounded-lg border-2 border-line-strong bg-surface px-3 py-2 text-sm"
      >
        <option value="system">Match my device</option>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
    </div>
  );
}
