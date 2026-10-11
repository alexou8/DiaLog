'use client';

import { useSearchParams } from 'next/navigation';
import { Callout } from '@/components/ui';

/**
 * Confirmation after account deletion. `deleteAccountAction` redirects to
 * `/?deleted=1`, and the landing page used to ignore the parameter, so the
 * last thing a person saw after erasing their health data was the marketing
 * page with no word that it had worked.
 */
export function DeletedNotice() {
  const params = useSearchParams();
  if (params.get('deleted') !== '1') return null;
  return (
    <div className="mb-8">
      <Callout tone="positive" icon="ok" role="status">
        Your account and all its data have been deleted.
      </Callout>
    </div>
  );
}
