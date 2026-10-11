import { THEME_SCRIPT } from './theme';

/**
 * Applies the saved colour-scheme preference before first paint so that a
 * dark-theme user never sees a flash of the light theme, and keeps "Match my
 * device" following the OS live (see THEME_SCRIPT). Text-size and
 * reduced-motion are account preferences and are applied by
 * AccountPreferences inside the signed-in shell; this only handles theme,
 * which is a device-level choice rather than an account one.
 */
export function PreferencesScript() {
  return <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />;
}
