/*
 * DiaLog service worker.
 *
 * Privacy rule: this worker caches ONLY the public application shell and
 * static build assets. It never caches API responses, never caches any page
 * under /app, and never stores health data. That means the installed app opens
 * instantly and shows a real offline page when the network is gone, without
 * leaving readings in a cache that outlives the session.
 */
const CACHE = 'dialog-shell-v1';
const SHELL = ['/', '/offline', '/manifest.webmanifest', '/icons/icon-192.png'];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting()));
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
            const copy = response.clone();
            caches.open(CACHE).then((cache) => cache.put(request, copy));
            return response;
          }),
      ),
    );
    return;
  }

  // Public pages: network-first, fall back to the cached shell when offline.
  event.respondWith(
    fetch(request)
      .then((response) => {
        const copy = response.clone();
        caches.open(CACHE).then((cache) => cache.put(request, copy));
        return response;
      })
      .catch(() => caches.match(request).then((hit) => hit ?? caches.match('/offline'))),
  );
});
