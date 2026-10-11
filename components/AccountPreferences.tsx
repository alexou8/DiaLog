'use client';

import { useLayoutEffect } from 'react';

/**
 * Applies the account-level display preferences (larger text, reduced motion)
 * to <html>, where `--dl-font-scale` and the motion rules in globals.css read
 * them.
 *
 * Rendered by the signed-in shell rather than the root layout so that the
 * public pages stay static (see app/layout.tsx). Two halves:
 *
 * - The inline script runs while the server HTML is being parsed, before the
 *   shell's content paints, so a large-text user does not see the page jump.
 * - The layout effect covers what a script cannot: client-side navigation
 *   into the app (an inserted <script> never runs) and a preference saved in
 *   Settings, which re-renders this with new props. Its cleanup removes the
 *   attributes when the shell unmounts, e.g. after signing out.
 */
export function AccountPreferences({
  largeText,
  reduceMotion,
}: {
  largeText: boolean;
  reduceMotion: boolean;
}) {
  useLayoutEffect(() => {
    const root = document.documentElement;
    if (largeText) root.dataset.text = 'large';
    else delete root.dataset.text;
    if (reduceMotion) root.dataset.motion = 'reduced';
    else delete root.dataset.motion;
    return () => {
      delete root.dataset.text;
      delete root.dataset.motion;
    };
  }, [largeText, reduceMotion]);

  const script = `(function(){var r=document.documentElement;${largeText ? "r.dataset.text='large';" : ''}${reduceMotion ? "r.dataset.motion='reduced';" : ''}})();`;
  return <script dangerouslySetInnerHTML={{ __html: script }} />;
}
