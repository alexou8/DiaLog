import { Card, Skeleton } from '@/components/ui';

export default function HealthLoading() {
  return (
    <div className="space-y-8">
      <div className="space-y-2">
        <Skeleton className="h-8 w-32" />
        <Skeleton className="h-4 w-56" />
      </div>
      {[0, 1, 2, 3, 4].map((i) => (
        <Card key={i}>
          <Skeleton className="mb-4 h-6 w-32" />
          <Skeleton className="h-16 w-full" />
        </Card>
      ))}
    </div>
  );
}
