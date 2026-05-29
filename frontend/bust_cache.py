#!/usr/bin/env python3
"""
bust_cache.py — запускай перед каждым деплоем (или в CI/CD).

Что делает:
  1. Генерирует BUILD_HASH из содержимого всех файлов (меняется только если
     реально что-то изменилось).
  2. Заменяет ?v=... у global.css, nav.js, supabase.js во всех .html файлах.
  3. Добавляет <meta http-equiv="Cache-Control"> в <head> каждого .html.
  4. Обновляет CACHE_KEY в cabinet.html чтобы сбросить localStorage.
"""

import os
import re
import hashlib
import glob

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. Считаем хэш из содержимого всех исходников ──────────────────────────
def build_hash():
    h = hashlib.sha1()
    for ext in ("*.html", "*.css", "*.js"):
        for path in sorted(glob.glob(os.path.join(FRONTEND_DIR, ext))):
            with open(path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:10]

VERSION = build_hash()
print(f"BUILD VERSION: {VERSION}")

# ── 2. Обрабатываем каждый HTML файл ───────────────────────────────────────
HTML_FILES = glob.glob(os.path.join(FRONTEND_DIR, "*.html"))

# Файлы для версионирования (локальные статики)
VERSIONED_FILES = ["global.css", "nav.js", "supabase.js"]

META_NO_CACHE = (
    '  <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">\n'
    '  <meta http-equiv="Pragma" content="no-cache">\n'
    '  <meta http-equiv="Expires" content="0">\n'
)

for filepath in HTML_FILES:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    original = content

    # 2a. Добавить/обновить meta no-cache после <head>
    if 'http-equiv="Cache-Control"' not in content:
        content = re.sub(
            r"(<head[^>]*>)",
            r"\1\n" + META_NO_CACHE.rstrip("\n"),
            content,
            count=1,
            flags=re.IGNORECASE,
        )
    else:
        # уже есть — ничего не трогаем
        pass

    # 2b. Обновить ?v= у локальных статиков
    for filename in VERSIONED_FILES:
        # href="global.css" или href="global.css?v=abc"
        content = re.sub(
            rf'((?:href|src)="{re.escape(filename)})(?:\?v=[^"]*)?(")',
            rf'\1?v={VERSION}\2',
            content,
        )

    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  updated: {os.path.basename(filepath)}")

# ── 3. Обновить CACHE_KEY в cabinet.html ────────────────────────────────────
cabinet = os.path.join(FRONTEND_DIR, "cabinet.html")
with open(cabinet, "r", encoding="utf-8") as f:
    cab = f.read()

new_key = f"medeus_dash_{VERSION}"
cab_new = re.sub(
    r"(const CACHE_KEY\s*=\s*')[^']*(')",
    rf"\g<1>{new_key}\g<2>",
    cab,
)

if cab_new != cab:
    with open(cabinet, "w", encoding="utf-8") as f:
        f.write(cab_new)
    print(f"  cabinet.html CACHE_KEY → {new_key}")

print("Done ✓")
