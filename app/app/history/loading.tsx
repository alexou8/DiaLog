import { Card, Skeleton } from '@/components/ui';

export default function HistoryLoading() {
  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4">
        <div className="space-y-2">
          <Skeleton className="h-8 w-32" />
          <Skeleton className="h-4 w-56" />
        </div>
        <Skeleton className="h-11 w-40" />
      </div>
      <Skeleton className="h-11 w-full" />
      <Card>
        <Skeleton className="h-16 w-full" />
        <Skeleton className="mt-2 h-16 w-full" />
        <Skeleton className="mt-2 h-16 w-full" />
      </Card>
    </div>
  );
}
