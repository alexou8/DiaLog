import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { toLocalInputValue } from '@/lib/domain/time';
import { formatGlucose, unitLabel } from '@/lib/domain/units';
import { Card, MedicalDisclaimer, PageHeader } from '@/components/ui';
import { GlucoseForm } from './GlucoseForm';

export const metadata: Metadata = { title: 'Add a glucose reading' };

export default async function NewGlucosePage() {
  const { profile } = await requireOnboardedUser();
  return (
    <div className="mx-auto max-w-xl">
      <PageHeader
        title="Add a glucose reading"
        description="Two things are needed: the number and when it was taken. Everything else is optional."
      />
      <Card>
        <GlucoseForm
          unit={profile.glucoseUnit}
          unitLabel={unitLabel(profile.glucoseUnit)}
          defaultTime={toLocalInputValue(new Date(), profile.timezone)}
          targetHint={`Your target range is ${formatGlucose(profile.targetLowMgdl, profile.glucoseUnit, profile.locale)}–${formatGlucose(profile.targetHighMgdl, profile.glucoseUnit, profile.locale)} ${unitLabel(profile.glucoseUnit)}.`}
        />
      </Card>
      <div className="mt-6">
        <MedicalDisclaimer compact />
      </div>
    </div>
  );
}
