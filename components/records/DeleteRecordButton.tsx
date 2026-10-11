'use client';

import type { DeletableRecordType } from '@/lib/db/health-records';
import { deleteRecordAction } from '@/lib/actions/records';
import { Button } from '@/components/ui';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';

/**
 * Two-step delete confirmation, on shadcn/ui's AlertDialog. The one copy used
 * by Meals, Activity, Health and History (it used to be four byte-identical
 * files).
 *
 * The record is identified by three pieces the page formats with its own
 * helpers: what kind of record it is (`noun`), when it was taken (`when`) and
 * its value (`detail`). The trigger's accessible name and the dialog title are
 * both "Delete {noun}, {when}, {detail}". A list of "Delete this record"
 * buttons is indistinguishable in a screen reader's buttons list, and a
 * confirmation that does not say what it will remove invites deleting the
 * wrong row. The visible text stays "Delete", so the accessible name begins
 * with the visible label (WCAG 2.5.3, Label in Name). This component never
 * converts or formats a health value itself: units go through
 * `lib/domain/units.ts` in the caller.
 *
 * This replaced a <details>/<summary> popover. The popover worked without
 * JavaScript, which the dialog does not, and that trade was made deliberately:
 * deleting a health record is exactly the case the alertdialog role exists
 * for. The dialog traps focus, restores it to the trigger on close, closes on
 * Escape, names itself through aria-labelledby/aria-describedby, and marks the
 * page behind it inert. The popover did none of that, so a screen-reader user
 * could tab straight past the confirmation into the page behind it and never
 * know the prompt was open.
 *
 * The destructive action is still a real <form> posting to a Server Action, so
 * the deletion itself is a normal submit rather than a fetch the client has to
 * get right. The action re-checks ownership (`where: { id, userId }`); nothing
 * here is trusted for that.
 */
export function DeleteRecordButton({
  type,
  id,
  noun,
  when,
  detail,
}: {
  type: DeletableRecordType;
  id: string;
  /** "glucose reading", "meal", "sleep entry". */
  noun: string;
  /** The record's date and time, already formatted for the user's locale and zone. */
  when: string;
  /** The record's value or description, already formatted. Omit if there is none. */
  detail?: string;
}) {
  const label = [noun, when, detail].filter(Boolean).join(', ');
  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button variant="secondary" size="sm" aria-label={`Delete ${label}`}>
          Delete
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete {label}?</AlertDialogTitle>
          <AlertDialogDescription>
            This removes the record from your history. It cannot be undone.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Keep it</AlertDialogCancel>
          <form action={deleteRecordAction}>
            <input type="hidden" name="type" value={type} />
            <input type="hidden" name="id" value={id} />
            <AlertDialogAction type="submit" variant="danger">
              Yes, delete
            </AlertDialogAction>
          </form>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
