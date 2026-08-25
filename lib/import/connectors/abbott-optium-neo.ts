/**
 * Abbott FreeStyle Optium Neo connector.
 *
 * HONESTY NOTE: Abbott does not publish a file-format specification for the
 * CSV/report export produced by the FreeStyle Optium Neo meter's companion
 * desktop software (Abbott Diabetes Care / CoPilot-family tools), and we
 * could not obtain a verified sample export to reverse-engineer precisely
 * (see docs/DEVICE_INTEGRATIONS.md). Rather than invent a column layout we
 * cannot verify, this connector is deliberately a THIN WRAPPER around
 * `generic-csv`: it only adds (a) recognition of the Abbott header preamble
 * text so the importer can label the file as "from an Abbott meter" and
 * surface Abbott-specific export instructions, and (b) a couple of
 * additional column-name aliases reported in user-facing Abbott
 * documentation ("Result", "Glucose (mg/dL)"). All the actual parsing is
 * generic-csv's tolerant column matching. If Abbott's real export layout
 * differs, generic-csv's column detection still degrades gracefully to
 * per-row UNSUPPORTED_ROW/MISSING_VALUE issues rather than silent
 * misimport.
 */
import { parseGenericCsv } from './generic-csv';
import type { DeviceConnector, ParseOptions, ParseResult, ParsedFile } from '../types';

const PREAMBLE_MARKERS = ['freestyle optium', 'abbott diabetes care', 'optium neo'];

function textHead(sample: ParsedFile): string {
  if (sample.rows)
    return sample.rows
      .slice(0, 5)
      .map((r) => r.join(' '))
      .join(' ')
      .toLowerCase();
  return (sample.text ?? '').slice(0, 2000).toLowerCase();
}

export const abbottOptiumNeoConnector: DeviceConnector = {
  id: 'abbott-optium-neo',
  name: 'Abbott FreeStyle Optium Neo',
  vendor: 'Abbott',
  description:
    "Thin specialisation of the generic CSV importer for FreeStyle Optium Neo meter software exports. Abbott has not published this export's column layout, so this connector recognises the Abbott header preamble and falls back entirely to tolerant, column-name-driven CSV parsing rather than assuming an unverified fixed layout.",
  howToExport: [
    "Connect the FreeStyle Optium Neo meter to a computer running Abbott's meter software (e.g. via the supplied USB cable).",
    "Use the software's export/report feature to save readings as a CSV or text file.",
    'Upload that file here. If it is not recognised automatically, choose "Generic CSV" and DiaLog will still import it column-by-column.',
  ],
  acceptedExtensions: ['.csv', '.txt'],
  kind: 'GLUCOSE_METER',

  detect(sample: ParsedFile): number {
    const head = textHead(sample);
    const hasPreamble = PREAMBLE_MARKERS.some((m) => head.includes(m));
    if (!hasPreamble) return 0;
    // Slightly higher than generic-csv's flat 0.5 so the Abbott label wins when the preamble is present.
    return 0.6;
  },

  parse(file: ParsedFile, options: ParseOptions): ParseResult {
    return parseGenericCsv(file, options);
  },
};
