import { Card, Skeleton } from '@/components/ui';

export default function ActivityLoading() {
  return (
    <div className="space-y-8">
      <div className="flex items-end justify-between gap-4">
        <div className="space-y-2">
          <Skeleton className="h-8 w-32" />
          <Skeleton className="h-4 w-56" />
        </div>
        <Skeleton className="h-11 w-36" />
      </div>
      <Card>
        <Skeleton className="mb-4 h-16 w-full" />
        <Skeleton className="h-40 w-full" />
      </Card>
      <div className="space-y-6">
        {[0, 1, 2].map((i) => (
          <Card key={i}>
            <Skeleton className="mb-3 h-5 w-40" />
            <Skeleton className="h-16 w-full" />
          </Card>
        ))}
      </div>
    </div>
  );
}
