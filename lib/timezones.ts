/**
 * Timezone choices offered in onboarding and settings.
 *
 * Canadian zones lead the list because that is where DiaLog is first being
 * used; the rest cover the common cases without turning the control into a
 * 400-item scroll.
 */
export const TIMEZONE_GROUPS: { label: string; zones: { id: string; label: string }[] }[] = [
  {
    label: 'Canada',
    zones: [
      { id: 'America/St_Johns', label: 'Newfoundland (St. John’s)' },
      { id: 'America/Halifax', label: 'Atlantic (Halifax)' },
      { id: 'America/Toronto', label: 'Eastern (Toronto, Ottawa, Montréal)' },
      { id: 'America/Winnipeg', label: 'Central (Winnipeg)' },
      { id: 'America/Regina', label: 'Saskatchewan (Regina)' },
      { id: 'America/Edmonton', label: 'Mountain (Edmonton, Calgary)' },
      { id: 'America/Vancouver', label: 'Pacific (Vancouver)' },
      { id: 'America/Whitehorse', label: 'Yukon (Whitehorse)' },
    ],
  },
  {
    label: 'United States',
    zones: [
      { id: 'America/New_York', label: 'Eastern (New York)' },
      { id: 'America/Chicago', label: 'Central (Chicago)' },
      { id: 'America/Denver', label: 'Mountain (Denver)' },
      { id: 'America/Phoenix', label: 'Arizona (Phoenix)' },
      { id: 'America/Los_Angeles', label: 'Pacific (Los Angeles)' },
      { id: 'America/Anchorage', label: 'Alaska (Anchorage)' },
      { id: 'Pacific/Honolulu', label: 'Hawaii (Honolulu)' },
    ],
  },
  {
    label: 'Elsewhere',
    zones: [
      { id: 'Europe/London', label: 'United Kingdom (London)' },
      { id: 'Europe/Dublin', label: 'Ireland (Dublin)' },
      { id: 'Europe/Paris', label: 'Central Europe (Paris)' },
      { id: 'Europe/Athens', label: 'Eastern Europe (Athens)' },
      { id: 'Asia/Kolkata', label: 'India (Kolkata)' },
      { id: 'Asia/Shanghai', label: 'China (Shanghai)' },
      { id: 'Asia/Tokyo', label: 'Japan (Tokyo)' },
      { id: 'Australia/Sydney', label: 'Australia (Sydney)' },
      { id: 'Pacific/Auckland', label: 'New Zealand (Auckland)' },
      { id: 'UTC', label: 'UTC' },
    ],
  },
];

export const ALL_TIMEZONE_IDS: string[] = TIMEZONE_GROUPS.flatMap((group) =>
  group.zones.map((zone) => zone.id),
);
