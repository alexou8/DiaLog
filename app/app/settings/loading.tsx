import { Skeleton } from '@/components/ui';

export default function SettingsLoading() {
  return (
    <div className="space-y-10 pb-16" aria-busy="true" aria-label="Loading settings">
      <div>
        <Skeleton className="h-8 w-40" />
        <Skeleton className="mt-2 h-4 w-96 max-w-full" />
      </div>

      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="space-y-4">
          <Skeleton className="h-6 w-48" />
          <div className="rounded-[var(--radius-card)] border border-line bg-surface p-5 sm:p-6">
            <Skeleton className="h-4 w-full max-w-sm" />
            <Skeleton className="mt-3 h-11 w-full" />
            <Skeleton className="mt-3 h-11 w-full" />
            <Skeleton className="mt-4 h-11 w-40" />
          </div>
        </div>
      ))}
    </div>
  );
}
