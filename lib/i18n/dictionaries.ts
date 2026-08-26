/**
 * Internationalisation.
 *
 * UI strings live in dictionaries rather than inside business logic, so that
 * adding a language is a data change. Numbers, dates and units are formatted
 * through `Intl` with the user's locale (see lib/domain/units.ts and
 * lib/domain/time.ts), so mmol/L, 24-hour clocks and Canadian date order work
 * without any component knowing about them.
 *
 * English and French (Canada) are wired up. Only the shared chrome is
 * translated today; page bodies fall back to English, which is stated
 * honestly in the language setting rather than pretending to be complete.
 */
export const LOCALES = ['en-CA', 'fr-CA'] as const;
export type Locale = (typeof LOCALES)[number];

export const LOCALE_LABELS: Record<Locale, string> = {
  'en-CA': 'English (Canada)',
  'fr-CA': 'Français (Canada), partial translation',
};

const en = {
  nav: {
    home: 'Home',
    insights: 'Insights',
    glucose: 'Glucose',
    meals: 'Meals',
    activity: 'Activity',
    health: 'Health',
    history: 'History',
    reports: 'Reports',
    assistant: 'Assistant',
    import: 'Import',
    settings: 'Settings',
    add: 'Add',
    signOut: 'Sign out',
    skipToContent: 'Skip to main content',
    mainNavigation: 'Main navigation',
  },
  common: {
    save: 'Save',
    cancel: 'Cancel',
    delete: 'Delete',
    edit: 'Edit',
    loading: 'Loading…',
    notEnoughData: 'Not enough data yet',
    whyThis: 'Why am I seeing this?',
  },
};

const fr: DeepPartial<typeof en> = {
  nav: {
    home: 'Accueil',
    insights: 'Observations',
    glucose: 'Glycémie',
    meals: 'Repas',
    activity: 'Activité',
    health: 'Santé',
    history: 'Historique',
    reports: 'Rapports',
    assistant: 'Assistant',
    import: 'Importer',
    settings: 'Réglages',
    add: 'Ajouter',
    signOut: 'Se déconnecter',
    skipToContent: 'Aller au contenu principal',
    mainNavigation: 'Navigation principale',
  },
  common: {
    save: 'Enregistrer',
    cancel: 'Annuler',
    delete: 'Supprimer',
    edit: 'Modifier',
    loading: 'Chargement…',
    notEnoughData: 'Pas encore assez de données',
    whyThis: 'Pourquoi je vois ceci ?',
  },
};

type DeepPartial<T> = { [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K] };
export type Dictionary = typeof en;

/** Merge a partial translation over English so nothing renders as a blank. */
function merge(base: Dictionary, override: DeepPartial<Dictionary>): Dictionary {
  return {
    nav: { ...base.nav, ...(override.nav ?? {}) },
    common: { ...base.common, ...(override.common ?? {}) },
  };
}

export function getDictionary(locale: string): Dictionary {
  return locale.startsWith('fr') ? merge(en, fr) : en;
}
