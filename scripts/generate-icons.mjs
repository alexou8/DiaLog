/**
 * Generates the PWA icon set as PNGs with no image dependencies.
 *
 * The mark is a rising-then-settling curve inside a rounded square — a calm
 * reference to a glucose trace rather than a medical cross.
 */
import { deflateSync } from 'node:zlib';
import { writeFileSync, mkdirSync } from 'node:fs';

const BRAND = [21, 94, 105];
const INK = [255, 255, 255];

function crc32(buf) {
  let c, crc = 0xffffffff;
  for (let n = 0; n < buf.length; n++) {
    c = (crc ^ buf[n]) & 0xff;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    crc = (crc >>> 8) ^ c;
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
}

function png(size, pixels) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // RGBA
  const raw = Buffer.alloc(size * (size * 4 + 1));
  for (let y = 0; y < size; y++) {
    raw[y * (size * 4 + 1)] = 0;
    pixels.copy(raw, y * (size * 4 + 1) + 1, y * size * 4, (y + 1) * size * 4);
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

/** Curve control points in unit space. */
const CURVE = [
  [0.16, 0.66], [0.3, 0.7], [0.42, 0.4], [0.55, 0.3], [0.68, 0.52], [0.84, 0.44],
];

function distanceToCurve(px, py) {
  let best = Infinity;
  for (let i = 0; i < CURVE.length - 1; i++) {
    const [x1, y1] = CURVE[i];
    const [x2, y2] = CURVE[i + 1];
    const dx = x2 - x1, dy = y2 - y1;
    const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)));
    best = Math.min(best, Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy)));
  }
  return best;
}

function render(size, { maskable }) {
  const buf = Buffer.alloc(size * size * 4);
  const radius = maskable ? size : size * 0.22;
  const inset = maskable ? 0 : 0;
  const stroke = 0.052;
  const scale = maskable ? 0.78 : 1; // keep the mark inside the maskable safe zone
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      // Rounded-square mask.
      const cx = Math.min(x, size - 1 - x), cy = Math.min(y, size - 1 - y);
      let inside = true;
      if (!maskable && cx < radius && cy < radius) {
        inside = Math.hypot(radius - cx, radius - cy) <= radius;
      }
      if (!inside) { buf[i + 3] = 0; continue; }
      buf[i] = BRAND[0]; buf[i + 1] = BRAND[1]; buf[i + 2] = BRAND[2]; buf[i + 3] = 255;
      const ux = (x / size - 0.5) / scale + 0.5;
      const uy = (y / size - 0.5) / scale + 0.5;
      const d = distanceToCurve(ux, uy);
      if (d < stroke) {
        const a = Math.min(1, (stroke - d) / (stroke * 0.25));
        buf[i] = Math.round(BRAND[0] + (INK[0] - BRAND[0]) * a);
        buf[i + 1] = Math.round(BRAND[1] + (INK[1] - BRAND[1]) * a);
        buf[i + 2] = Math.round(BRAND[2] + (INK[2] - BRAND[2]) * a);
      }
    }
  }
  return png(size, buf);
}

mkdirSync('public/icons', { recursive: true });
for (const size of [180, 192, 512]) {
  writeFileSync(`public/icons/icon-${size}.png`, render(size, { maskable: false }));
}
writeFileSync('public/icons/maskable-512.png', render(512, { maskable: true }));
writeFileSync('public/favicon.ico', render(32, { maskable: false }));
console.log('icons written');
