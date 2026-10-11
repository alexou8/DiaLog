/*
 * DiaLog service worker.
 *
 * Privacy rule: this worker caches ONLY the public application shell and
 * static build assets. It never caches API responses, never caches any page
 * under /app, and never stores health data. That means the installed app opens
 * instantly and shows a real offline page when the network is gone, without
 * leaving readings in a cache that outlives the session.
 */
// v3: the runtime cache is now an allowlist (see `isCacheablePage`). Renaming
// the cache is what makes `activate` delete every v2 entry, including any
// `/reset-password?token=…` responses the old catch-all policy stored.
const CACHE = 'dialog-shell-v3';
const SHELL = ['/', '/offline', '/manifest.webmanifest', '/icons/icon-192.png'];

// Public pages worth keeping for offline reading. Everything else that is not
// a static asset goes to the network only. An allowlist rather than a
// blocklist, because the blocklist version cached sign-in, sign-up and
// password-reset pages, query-string variants, and Next's RSC payloads (a new
// entry per `?_rsc=` hash, so the cache grew without bound).
const CACHEABLE_PAGES = new Set([
  '/',
  '/offline',
  '/about',
  '/privacy',
  '/security',
  '/accessibility',
  '/terms',
  '/help',
]);

// Upper bound on cached static assets. Hashed build output changes on every
// deploy, so without a cap old chunks pile up for as long as the app is
// installed.
const MAX_ASSET_ENTRIES = 120;

function isCacheablePage(url, request) {
  return (
    url.search === '' &&
    CACHEABLE_PAGES.has(url.pathname) &&
    // RSC payloads share the page's URL but are a different representation.
    !request.headers.has('RSC') &&
    !request.headers.has('Next-Router-Prefetch')
  );
}

/** Only complete, same-origin, explicitly cacheable 200s are ever stored. */
function isStorable(response) {
  if (!response || response.status !== 200 || response.type !== 'basic') return false;
  const cacheControl = response.headers.get('Cache-Control') || '';
  return !/no-store|private/i.test(cacheControl);
}

async function trim(cache) {
  const keys = await cache.keys();
  const assets = keys.filter((request) => {
    const path = new URL(request.url).pathname;
    return path.startsWith('/_next/static') || path.startsWith('/icons');
  });
  const excess = assets.length - MAX_ASSET_ENTRIES;
  // Cache keys come back in insertion order, so the oldest go first.
  for (let i = 0; i < excess; i++) await cache.delete(assets[i]);
}

async function precache() {
  const cache = await caches.open(CACHE);
  await cache.addAll(SHELL);
  // The offline page is useless unstyled. Its stylesheets are hashed build
  // files the shell list cannot name ahead of time, so read them out of the
  // page that was just cached.
  const offline = await cache.match('/offline');
  if (!offline) return;
  const html = await offline.clone().text();
  const sheets = Array.from(html.matchAll(/href="(\/_next\/static\/[^"]+\.css)"/g), (m) => m[1]);
  if (sheets.length > 0) await cache.addAll(sheets);
}

self.addEventListener('install', (event) => {
  event.waitUntil(precache().then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  // Never touch authenticated surfaces or the API.
  const isPrivate = url.pathname.startsWith('/app') || url.pathname.startsWith('/api');

  if (isPrivate) {
    event.respondWith(
      fetch(request).catch(() =>
        request.mode === 'navigate' ? caches.match('/offline') : Response.error(),
      ),
    );
    return;
  }

  // Immutable build output: cache-first.
  if (url.pathname.startsWith('/_next/static') || url.pathname.startsWith('/icons')) {
    event.respondWith(
      caches.match(request).then(
        (hit) =>
          hit ??
          fetch(request).then((response) => {
            if (isStorable(response)) {
              const copy = response.clone();
              event.waitUntil(
                caches
                  .open(CACHE)
                  .then((cache) => cache.put(request, copy).then(() => trim(cache))),
              );
            }
            return response;
          }),
      ),
    );
    return;
  }

  // Allowlisted public pages: network-first, falling back to the cached copy
  // when offline.
  if (isCacheablePage(url, request)) {
    event.respondWith(
      fetch(request)
        .then((response) => {
          if (isStorable(response)) {
            const copy = response.clone();
            event.waitUntil(caches.open(CACHE).then((cache) => cache.put(request, copy)));
          }
          return response;
        })
        .catch(() => caches.match(request).then((hit) => hit ?? caches.match('/offline'))),
    );
    return;
  }

  // Everything else (sign-in, sign-up, reset-password?token=…, any query
  // string, RSC payloads): network only, never stored. A navigation that
  // fails offline still gets the offline page rather than the browser's error.
  if (request.mode === 'navigate') {
    event.respondWith(fetch(request).catch(() => caches.match('/offline')));
  }
});
