/**
 * nav.js — общий заголовок и мобильная навигация для авторизованных страниц
 * Подключать после supabase.js
 */

function renderHeader(activePage) {
  const pages = [
    { id: 'cabinet',    href: 'cabinet.html',    label: 'Кабинет',  icon: `<svg viewBox="0 0 24 24" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>` },
    { id: 'upload',     href: 'upload.html',     label: 'Анализ',   icon: `<svg viewBox="0 0 24 24" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>` },
    { id: 'indicators', href: 'indicators.html', label: 'Данные', icon: `<svg viewBox="0 0 24 24" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>` },
    { id: 'export',     href: 'export.html',     label: 'Экспорт',   icon: `<svg viewBox="0 0 24 24" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>` },
    { id: 'profile',    href: 'profile.html',    label: 'Профиль',  icon: `<svg viewBox="0 0 24 24" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>` },
  ];

  // Desktop nav items
  const desktopNavHtml = pages.map(p => `
    <a href="${p.href}" class="desk-nav-link${p.id === activePage ? ' active' : ''}">${p.label}</a>
  `).join('');

  // Mobile bottom nav
  const mobileNavHtml = pages.map(p => `
    <a href="${p.href}" class="mobile-nav-item${p.id === activePage ? ' active' : ''}">
      ${p.icon}
      ${p.label}
    </a>
  `).join('');

  document.getElementById('desktopNav').innerHTML = desktopNavHtml;
  document.getElementById('mobileNav').innerHTML  = mobileNavHtml;
}
