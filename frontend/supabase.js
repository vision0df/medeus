// ========================================
// supabase.js — общая конфигурация
// Замените значения на свои из Supabase Dashboard
// ========================================

const SUPABASE_URL = 'https://ralfwhzqmtwcxxbjtliy.supabase.co';
const API_BASE = 'https://medeus.onrender.com';

// Supabase SDK через CDN (подключается в каждой странице через importmap или script)
// Используем глобальный supabase из CDN-скрипта

function getSupabase() {
  return supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
}

// Получить текущего юзера и его access_token
async function getCurrentSession() {
  const client = getSupabase();
  const { data: { session } } = await client.auth.getSession();
  return session; // null если не авторизован
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
  await client.auth.signOut();
  window.location.href = 'auth.html';
}
