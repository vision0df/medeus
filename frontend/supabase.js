// ========================================
// supabase.js — общая конфигурация
// ========================================

const SUPABASE_URL    = 'https://ralfwhzqmtwcxxbjtliy.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJhbGZ3aHpxbXR3Y3h4Ymp0bGl5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzc2MzY5NDEsImV4cCI6MjA5MzIxMjk0MX0.VtDBrC6u2W22xqzkUX8DNo1Xm2zalUsFRNMrO3jIeIE';
const API_BASE = 'https://medeus.onrender.com';

// ── Singleton клиент ──
let _supabaseClient = null;
function getSupabase() {
  if (!_supabaseClient) {
    _supabaseClient = supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
      auth: { persistSession: true, autoRefreshToken: true },
    });
    // Слушаем обновление токена — сохраняем свежую сессию
    _supabaseClient.auth.onAuthStateChange((event, session) => {
      if (event === 'TOKEN_REFRESHED') {
        console.log('🔑 Token refreshed');
      }
      if (event === 'SIGNED_OUT') {
        try { localStorage.removeItem('medeus_dashboard_cache'); } catch(e) {}
        window.location.href = 'auth.html';
      }
    });
  }
  return _supabaseClient;
}

// Получить текущую сессию (Supabase SDK сам рефрешит если нужно)
async function getCurrentSession() {
  const client = getSupabase();
  const { data: { session } } = await client.auth.getSession();
  return session;
}

// Редирект если не залогинен
async function requireAuth() {
  const session = await getCurrentSession();
  if (!session) {
    window.location.href = 'auth.html';
    return null;
  }
  return session;
}

// Выйти из аккаунта
async function signOut() {
  const client = getSupabase();
  try { localStorage.removeItem('medeus_dashboard_cache'); } catch(e) {}
  await client.auth.signOut();
  window.location.href = 'auth.html';
}

// ── Wake-up ping для Render cold start ──
// Вызывать на страницах, которые будут делать запросы к API
function pingBackend() {
  fetch(`${API_BASE}/health`, { method: 'GET' }).catch(() => {});
}
