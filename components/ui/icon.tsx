/**
 * DiaLog icon registry.
 *
 * Icons are referenced by a semantic name rather than imported ad hoc, so a
 * given concept (a meal, a caution, an export) always draws the same glyph
 * everywhere in the product. This replaced an earlier emoji-based icon system:
 * emoji render differently on every platform, are read aloud unpredictably by
 * screen readers, and made a health record look like a chat message.
 *
 * Icons are decorative by default (`aria-hidden`). They never carry meaning on
 * their own — every one of them sits next to a text label.
 */
import type { ComponentType, SVGProps } from 'react';
import {
  Activity,
  ArrowDown,
  ArrowUp,
  Archive,
  Brain,
  ChartLine,
  ChartColumn,
  Check,
  CircleAlert,
  Cloud,
  Download,
  Droplet,
  FileText,
  Footprints,
  Gauge,
  GlassWater,
  Heart,
  HeartPulse,
  House,
  Info,
  Lightbulb,
  Lock,
  LogOut,
  MessageSquareText,
  Minus,
  Moon,
  NotebookPen,
  PenLine,
  Pill,
  Plus,
  Scale,
  Search,
  Settings,
  Stethoscope,
  Thermometer,
  TrendingUp,
  TriangleAlert,
  Upload,
  Utensils,
  Weight,
} from 'lucide-react';

type LucideIcon = ComponentType<SVGProps<SVGSVGElement> & { strokeWidth?: number | string }>;

const ICONS = {
  // Navigation and destinations
  home: House,
  insights: Lightbulb,
  glucose: Droplet,
  meals: Utensils,
  activity: Footprints,
  health: HeartPulse,
  assistant: MessageSquareText,
  reports: FileText,
  history: Archive,
  import: Upload,
  settings: Settings,
  signOut: LogOut,

  // Health record types
  sleep: Moon,
  medication: Pill,
  weight: Weight,
  bloodPressure: Stethoscope,
  mood: Brain,
  symptom: Thermometer,
  hydration: GlassWater,
  exercise: Activity,
  heart: Heart,

  // Actions
  add: Plus,
  edit: PenLine,
  download: Download,
  search: Search,
  quickLog: NotebookPen,

  // Status and meaning
  ok: Check,
  caution: TriangleAlert,
  alert: CircleAlert,
  info: Info,
  up: ArrowUp,
  down: ArrowDown,
  steady: Minus,
  trend: TrendingUp,
  chart: ChartLine,
  bars: ChartColumn,
  range: Gauge,
  scale: Scale,

  // Privacy
  local: Lock,
  external: Cloud,
} satisfies Record<string, LucideIcon>;

export type IconName = keyof typeof ICONS;

export function isIconName(value: unknown): value is IconName {
  return typeof value === 'string' && value in ICONS;
}

/**
 * Sizes are set in `em` so an icon tracks the text it sits beside, including
 * when the user turns on the larger-text preference.
 */
export function Icon({
  name,
  className,
  size = '1.15em',
  strokeWidth = 1.75,
  title,
}: {
  name: IconName;
  className?: string;
  size?: string | number;
  strokeWidth?: number;
  /** Only pass this when the icon is the sole content of a control. */
  title?: string;
}) {
  const Glyph = ICONS[name];
  return (
    <Glyph
      className={className}
      width={size}
      height={size}
      strokeWidth={strokeWidth}
      aria-hidden={title ? undefined : true}
      role={title ? 'img' : undefined}
      aria-label={title}
      focusable="false"
    />
  );
}
