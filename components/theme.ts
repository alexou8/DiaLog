/**
 * Device-level colour scheme, shared by the pre-paint script and the picker.
 *
 * The theme-colour values are the two canvases (light and dark), so the
 * browser chrome and the installed PWA's title bar match the page behind
 * them. They must stay in step with `--color-canvas` in globals.css and with
 * `viewport.themeColor` in app/layout.tsx.
 */
export const THEME_STORAGE_KEY = 'dialog-theme';
export const THEME_COLORS = { light: '#f7f6f2', dark: '#14181f' } as const;

export type ThemePreference = 'light' | 'dark' | 'system';

/**
 * Inline, dependency-free script that runs before first paint (see
 * PreferencesScript). It also owns "Match my device": a `matchMedia` change
 * listener re-applies the theme whenever the OS switches, on every page, for
 * as long as the stored preference is "system" or unset. Before this the
 * page kept whatever scheme it loaded with until a reload.
 *
 * Every storage access is wrapped: Safari private mode and blocked site data
 * throw on `localStorage`, and a throw here would leave the page unthemed.
 */
export const THEME_SCRIPT = `(function(){var C=${JSON.stringify(THEME_COLORS)};var K=${JSON.stringify(THEME_STORAGE_KEY)};var mq=window.matchMedia('(prefers-color-scheme: dark)');function pref(){try{return localStorage.getItem(K)}catch(e){return null}}function apply(){var p=pref();var t=p==='light'||p==='dark'?p:mq.matches?'dark':'light';document.documentElement.dataset.theme=t;var m=document.querySelectorAll('meta[name="theme-color"]');for(var i=0;i<m.length;i++)m[i].setAttribute('content',C[t]);}try{apply()}catch(e){}try{mq.addEventListener('change',function(){try{apply()}catch(e){}})}catch(e){}})();`;

export function readThemePreference(): ThemePreference {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    return stored === 'light' || stored === 'dark' || stored === 'system' ? stored : 'system';
  } catch {
    return 'system';
  }
}

/** Persists the choice (best effort) and applies it to the page and browser chrome. */
export function applyThemePreference(next: ThemePreference): void {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, next);
  } catch {
    // Storage unavailable: the choice still applies to this page view.
  }
  const resolved =
    next === 'system'
      ? window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light'
      : next;
  document.documentElement.dataset.theme = resolved;
  // The theme-color metas are media-keyed to the OS scheme, so without this
  // a user who picked Dark on a light-mode phone got a light browser bar
  // above a dark page.
  document
    .querySelectorAll('meta[name="theme-color"]')
    .forEach((meta) => meta.setAttribute('content', THEME_COLORS[resolved]));
}
