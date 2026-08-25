/**
 * Carries a boolean preference the current form does not expose as a
 * visible control, so a save from any one settings section still submits a
 * value for every field `preferencesSchema` expects.
 *
 * Mirrors how `Checkbox` submits: present (any value) means true, absent
 * means false — never render this with the string "false", since
 * `z.coerce.boolean()` treats any non-empty string as true.
 */
export function HiddenBool({ name, value }: { name: string; value: boolean }) {
  if (!value) return null;
  return <input type="hidden" name={name} value="true" />;
}
