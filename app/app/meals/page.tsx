import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { prisma } from '@/lib/db/prisma';
import { dayKeyInZone } from '@/lib/domain/time';
import { Badge, ButtonLink, Card, CardHeader, EmptyState } from '@/components/ui';
import { BarChart, type Bar } from '@/components/charts/BarChart';
import { DeleteRecordButton } from './DeleteRecordButton';

export const metadata: Metadata = { title: 'Meals' };
export const dynamic = 'force-dynamic';

const MEAL_TYPE_LABELS: Record<string, string> = {
  BREAKFAST: 'Breakfast',
  LUNCH: 'Lunch',
  DINNER: 'Dinner',
  SNACK: 'Snack',
  OTHER: 'Other',
};

const MIN_SAMPLE = 3;

export default async function MealsPage() {
  const user = await requireOnboardedUser();
  const { locale, timezone } = user.profile;

  const meals = await prisma.meal.findMany({
    where: { userId: user.id },
    orderBy: { takenAt: 'desc' },
    take: 100,
    select: {
      id: true,
      takenAt: true,
      mealType: true,
      description: true,
      carbsG: true,
      proteinG: true,
      fatG: true,
      fiberG: true,
      calories: true,
      portion: true,
      note: true,
      estimateSource: true,
    },
  });

  if (meals.length === 0) {
    return (
      <div className="space-y-6">
        <header>
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Meals</h1>
        </header>
        <EmptyState
          title="No meals logged yet"
          icon="🍽️"
          action={<ButtonLink href="/app/meals/new">Log your first meal</ButtonLink>}
        >
          <p>
            Log what you eat to see it here — a short description and the time are all that is
            required. If you add carbs, DiaLog can start showing how meals relate to your readings.
          </p>
        </EmptyState>
      </div>
    );
  }

  const dayFmt = new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    weekday: 'long',
    month: 'long',
    day: 'numeric',
  });
  const timeFmt = new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    hour: 'numeric',
    minute: '2-digit',
  });

  const groups = new Map<string, typeof meals>();
  for (const meal of meals) {
    const key = dayKeyInZone(meal.takenAt, timezone);
    const bucket = groups.get(key);
    if (bucket) bucket.push(meal);
    else groups.set(key, [meal]);
  }

  // Average carbs by meal type, computed from the loaded page of meals.
  const byType = new Map<string, number[]>();
  for (const meal of meals) {
    if (meal.carbsG == null) continue;
    const arr = byType.get(meal.mealType);
    if (arr) arr.push(meal.carbsG);
    else byType.set(meal.mealType, [meal.carbsG]);
  }
  const bars: Bar[] = Object.keys(MEAL_TYPE_LABELS).map((type) => {
    const values = byType.get(type);
    const n = values?.length ?? 0;
    const avg = values && n >= MIN_SAMPLE ? values.reduce((a, b) => a + b, 0) / n : null;
    return { label: MEAL_TYPE_LABELS[type] ?? type, value: avg, n };
  });
  const hasChartData = bars.some((b) => b.value != null);

  return (
    <div className="space-y-8">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Meals</h1>
          <p className="mt-1 text-ink-muted">Your recent meals, grouped by day.</p>
        </div>
        <ButtonLink href="/app/meals/new">
          <span aria-hidden="true">＋</span> Log a meal
        </ButtonLink>
      </header>

      {hasChartData ? (
        <section aria-labelledby="meal-chart-heading">
          <Card>
            <CardHeader id="meal-chart-heading" title="Average carbs by meal type" level={2} />
            <BarChart
              bars={bars}
              valueLabel="Average carbs (g)"
              title="Average carbs by meal type"
              summary="Average carbohydrate content of your logged meals, grouped by meal type. Bars need at least three meals to show an average."
              minSample={MIN_SAMPLE}
              format={(v) => `${Math.round(v)}`}
            />
          </Card>
        </section>
      ) : null}

      <section aria-labelledby="meals-list-heading">
        <h2 id="meals-list-heading" className="sr-only">
          Meal log
        </h2>
        <div className="space-y-6">
          {[...groups.entries()].map(([dayKey, dayMeals]) => (
            <Card key={dayKey}>
              <h3 className="mb-3 text-base font-semibold text-ink-muted">
                {dayFmt.format(dayMeals[0]?.takenAt ?? new Date(`${dayKey}T00:00:00`))}
              </h3>
              <ul>
                {dayMeals.map((meal) => (
                  <li
                    key={meal.id}
                    className="flex flex-wrap items-start justify-between gap-3 border-b border-line py-3 last:border-0"
                  >
                    <div className="min-w-0">
                      <p className="flex flex-wrap items-center gap-2">
                        <span className="text-base font-semibold">{meal.description}</span>
                        {meal.estimateSource === 'AI_ESTIMATE' ? (
                          <Badge tone="info" icon="✨">
                            Estimated by the assistant
                          </Badge>
                        ) : null}
                      </p>
                      <p className="text-sm text-ink-muted">
                        {MEAL_TYPE_LABELS[meal.mealType] ?? meal.mealType} ·{' '}
                        {timeFmt.format(meal.takenAt)}
                        {meal.portion ? ` · ${meal.portion}` : ''}
                      </p>
                      <p className="mt-1 text-sm text-ink-muted">{formatMacros(meal)}</p>
                      {meal.note ? <p className="mt-1 text-sm">{meal.note}</p> : null}
                    </div>
                    <DeleteRecordButton type="meal" id={meal.id} label="this meal" />
                  </li>
                ))}
              </ul>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}

function formatMacros(meal: {
  carbsG: number | null;
  proteinG: number | null;
  fatG: number | null;
  fiberG: number | null;
  calories: number | null;
}): string {
  const parts: string[] = [];
  if (meal.carbsG != null) parts.push(`${Math.round(meal.carbsG)} g carbs`);
  if (meal.proteinG != null) parts.push(`${Math.round(meal.proteinG)} g protein`);
  if (meal.fatG != null) parts.push(`${Math.round(meal.fatG)} g fat`);
  if (meal.fiberG != null) parts.push(`${Math.round(meal.fiberG)} g fibre`);
  if (meal.calories != null) parts.push(`${Math.round(meal.calories)} kcal`);
  return parts.length > 0 ? parts.join(' · ') : 'No nutrition details recorded.';
}
