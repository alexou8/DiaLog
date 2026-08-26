'use client';

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
 * Two-step delete confirmation, on shadcn/ui's AlertDialog.
 *
 * This replaced a <details>/<summary> popover. The popover worked without
 * JavaScript, which the dialog does not, and that trade was made deliberately:
 * deleting a health record is exactly the case the alertdialog role exists for.
 * The dialog traps focus, restores it to the trigger on close, closes on
 * Escape, names itself through aria-labelledby/aria-describedby, and marks the
 * page behind it inert. The popover did none of that, so a screen-reader user
 * could tab straight past the confirmation into the page behind it and never
 * know the prompt was open.
 *
 * The destructive action is still a real <form> posting to a Server Action, so
 * the deletion itself is a normal submit rather than a fetch the client has to
 * get right.
 */
export function DeleteRecordButton({
  type,
  id,
  label,
}: {
  type: string;
  id: string;
  label: string;
}) {
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
