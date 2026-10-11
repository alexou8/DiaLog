'use client';

import { Button } from '@/components/ui';

/**
 * Opens the browser's print dialog, which is also how "save as PDF" works. The
 * report is something to bring to an appointment, and a button that says so is
 * easier to find than knowing the keyboard shortcut. It does nothing without
 * JavaScript, which is why it is a button and not a link.
 */
export function PrintButton() {
  return (
    <Button type="button" variant="secondary" onClick={() => window.print()}>
      Print or save as PDF
    </Button>
  );
}
