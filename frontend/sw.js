/* ═══════════════════════════════════════════════
   MEDEUS — Service Worker v1
   Стратегия: Cache-first для статики, Network-first для API
   ═══════════════════════════════════════════════ */

const CACHE_NAME = 'medeus-v1';

// Статические файлы для кеширования при установке
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/auth.html',
  '/register.html',
  '/cabinet.html',
  '/upload.html',
  '/indicators.html',
  '/indicator.html',
  '/analysis.html',
  '/all-analyses.html',
  '/all-recommendations.html',
  '/profile.html',
  '/global.css',
  '/nav.js',
  '/supabase.js',
  '/body_wireframe.png',
  '/body_wireframe_female.png',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
  '/icons/icon-180.png',
];

// При установке — кешируем статику
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      // Кешируем по одному, чтобы ошибка одного файла не ломала всё
      return Promise.allSettled(
        STATIC_ASSETS.map(url => cache.add(url).catch(() => {}))
      );
    }).then(() => self.skipWaiting())
  );
});

// При активации — удаляем старые кеши
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

// Fetch — стратегия по типу запроса
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // API запросы (Supabase, backend) — только сеть, не кешируем
  if (
    url.hostname.includes('supabase.co') ||
    url.hostname.includes('render.com') ||
    url.pathname.startsWith('/api/')
  ) {
    event.respondWith(fetch(request));
    return;
  }

  // Google Fonts — network-first с fallback на кеш
  if (url.hostname.includes('fonts.googleapis.com') || url.hostname.includes('fonts.gstatic.com')) {
    event.respondWith(
      fetch(request)
        .then(response => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(request, clone));
          return response;
        })
        .catch(() => caches.match(request))
    );
    return;
  }

  // Статика — cache-first, затем сеть
  event.respondWith(
    caches.match(request).then(cached => {
      if (cached) return cached;
      return fetch(request).then(response => {
        if (response && response.status === 200 && response.type !== 'opaque') {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(request, clone));
        }
        return response;
      }).catch(() => {
        // Офлайн-фолбек для HTML-страниц
        if (request.headers.get('accept')?.includes('text/html')) {
          return caches.match('/index.html');
        }
      });
    })
  );
});
