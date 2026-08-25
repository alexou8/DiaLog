/**
 * Connector registry: every known DeviceConnector, plus `detectConnector`
 * which picks the highest-confidence match for an uploaded file, falling
 * back to `generic-csv` for anything CSV-shaped that no specific connector
 * recognised.
 */
import { abbottLibreViewConnector } from './abbott-libreview';
import { abbottOptiumNeoConnector } from './abbott-optium-neo';
import { appleHealthConnector } from './apple-health';
import { dialogLegacyConnector } from './dialog-legacy';
import { genericCsvConnector } from './generic-csv';
import { genericJsonConnector } from './generic-json';
import { genericXmlConnector } from './generic-xml';
import { nightscoutConnector } from './nightscout';
import { omronConnector } from './omron';
import type { DeviceConnector, ParsedFile } from '../types';

/**
 * Order matters only as a tie-breaker: connectors earlier in this list win
 * when two connectors report the exact same confidence. Specific connectors
 * are listed before their generic fallbacks.
 */
export const CONNECTORS: readonly DeviceConnector[] = [
  dialogLegacyConnector,
  abbottLibreViewConnector,
  nightscoutConnector,
  appleHealthConnector,
  omronConnector,
  abbottOptiumNeoConnector,
  genericJsonConnector,
  genericXmlConnector,
  genericCsvConnector,
];

export function getConnector(id: string): DeviceConnector | undefined {
  return CONNECTORS.find((c) => c.id === id);
}

export interface DetectionResult {
  connector: DeviceConnector;
  confidence: number;
}

/**
 * Picks the best-matching connector for `file` by running every connector's
 * `detect`, taking the highest confidence (ties broken by registry order).
 * Returns `null` only when nothing — including generic-csv — thinks it can
 * parse the file (e.g. an empty file, or a format with none of rows/json/xml
 * set).
 */
export function detectConnector(file: ParsedFile): DetectionResult | null {
  let best: DetectionResult | null = null;
  for (const connector of CONNECTORS) {
    const confidence = connector.detect(file);
    if (confidence > 0 && (!best || confidence > best.confidence)) {
      best = { connector, confidence };
    }
  }
  return best;
}
