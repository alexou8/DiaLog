import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { Card, PageHeader } from '@/components/ui';
import { MedicationForm } from './MedicationForm';

export const metadata: Metadata = { title: 'Log medication' };

export default async function NewMedicationPage() {
  const user = await requireOnboardedUser();

  return (
    <div className="mx-auto max-w-xl">
      <PageHeader
        title="Log a medication"
        description="Record that you took something, and when."
      />
      <Card>
        <MedicationForm timezone={user.profile.timezone} />
      </Card>
    </div>
  );
}
