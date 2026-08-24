/**
 * Applies the saved colour-scheme preference before first paint so that a
 * dark-theme user never sees a flash of the light theme. Text-size and
 * reduced-motion preferences are server-rendered from the profile; this only
 * handles theme, which is a device-level choice rather than an account one.
 */
export function PreferencesScript() {
  const script = `(function(){try{var t=localStorage.getItem('dialog-theme');if(!t||t==='system'){t=window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';}document.documentElement.dataset.theme=t;}catch(e){}})();`;
  return <script dangerouslySetInnerHTML={{ __html: script }} />;
}
