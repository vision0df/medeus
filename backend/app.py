import os
import re
import json
import uuid
import hashlib
import logging
import threading
import traceback

import httpx
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ========================
# Логирование
# ========================
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ========================
# Приложение
# ========================
app = Flask(__name__)

ALLOWED_ORIGINS = [
    "https://medeus.vercel.app",
    "http://localhost:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8080",
]
CORS(app, origins=ALLOWED_ORIGINS)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Файл превышает максимальный размер 10 МБ"}), 413

# ========================
# Конфигурация
# ========================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL   = os.environ.get("SUPABASE_URL")
SUPABASE_KEY   = os.environ.get("SUPABASE_SERVICE_KEY")
STORAGE_BUCKET = "analyses-files"

log.info("GEMINI KEY:  %s", "OK" if GEMINI_API_KEY else "MISSING")
log.info("SUPABASE URL: %s", "OK" if SUPABASE_URL  else "MISSING")
log.info("SUPABASE KEY: %s", "OK" if SUPABASE_KEY  else "MISSING")

gemini = genai.Client(api_key=GEMINI_API_KEY)

# Модели для извлечения показателей (парсинг) — начинаем с лёгкой модели
GEMINI_MODELS_EXTRACT = [
    "gemini-2.5-flash-lite",  # основная для парсинга: 1000 запросов/день
    "gemini-2.5-flash",       # фолбэк
]

# Модели для анализа — начинаем с лучшей
GEMINI_MODELS_ANALYZE = [
    "gemini-2.5-flash",       # основная: лучшее качество
    "gemini-2.5-flash-lite",  # фолбэк при исчерпании лимита
]

VALID_GROUP_KEYS = {
    "blood", "hormones", "infections", "biomaterials",
    "genetics", "microbiome", "oncology", "functional",
}

ALLOWED_MIME_TYPES = {
    ".pdf":  "application/pdf",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
}


# ========================
# Утилиты
# ========================
def clean_name(raw_name: str) -> str:
    """Минимальная очистка: trim пробелов."""
    return (raw_name or "").strip()


def merge_key(name: str) -> str:
    """
    Ключ для дедупликации показателей.
    Приводит варианты одного показателя к одному ключу:
      "Лимфоциты (абс.)" / "Лимфоциты абсолютные" / "Лимфоциты абс" → "лимфоциты абс"
      "Базофилы (%)" / "Базофилы %" → "базофилы"
    """
    s = (name or "").strip().lower()
    s = re.sub(r'\s*\(\s*аб[сc]\.?\s*\)\s*', ' абс', s)
    s = re.sub(r'\s*\(\s*abs\.?\s*\)\s*', ' абс', s)
    s = re.sub(r'\s+абсолютн\w*$', ' абс', s)
    s = re.sub(r'\s*\(\s*%\s*\)\s*', ' ', s)
    s = re.sub(r'\s+%$', '', s)
    s = re.sub(r'\s*\([a-z][a-z0-9%#]{1,6}\)\s*', ' ', s)
    s = re.sub(r'\s*\([а-яёa-z][а-яёa-z\s]{1,30}\)\s*', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def get_mime_type(filename: str) -> str | None:
    ext = os.path.splitext(filename)[1].lower()
    return ALLOWED_MIME_TYPES.get(ext)


def supa_headers() -> dict:
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }


def parse_gemini_json(raw: str, expect_type: type = list):
    """
    Единая функция для парсинга JSON-ответов от Gemini.
    Убирает возможные markdown-блоки ```json ... ```.
    expect_type: list или dict — тип ожидаемого корневого объекта.
    Возвращает объект нужного типа или пустой list/dict при ошибке.
    """
    clean = raw.strip()
    if clean.startswith("```"):
        parts = clean.split("```")
        clean = parts[1] if len(parts) > 1 else clean
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()
    try:
        result = json.loads(clean)
        if isinstance(result, expect_type):
            return result
        # Попытка вытащить нужный тип из обёртки
        if expect_type is list and isinstance(result, dict):
            # Gemini иногда оборачивает массив в {"indicators": [...]}
            for v in result.values():
                if isinstance(v, list):
                    return v
        return expect_type()
    except Exception:
        return expect_type()


def validate_age(age: str) -> int:
    """Валидирует возраст. Возвращает int или кидает ValueError."""
    try:
        age_int = int(age)
        if not (0 <= age_int <= 120):
            raise ValueError()
        return age_int
    except (ValueError, TypeError):
        raise ValueError("Возраст должен быть числом от 0 до 120")


# ========================
# Supabase DB / Storage
# ========================
def db_insert(table: str, data: dict):
    resp = httpx.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=supa_headers(),
        json=data,
        timeout=10,
    )
    if resp.status_code not in (200, 201):
        raise Exception(f"DB insert error {resp.status_code}: {resp.text}")
    return resp.json()


def db_select(table: str, select: str, filters: dict) -> list:
    params = {"select": select, "order": "created_at.desc",
              **{k: f"eq.{v}" for k, v in filters.items()}}
    resp = httpx.get(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=supa_headers(),
        params=params,
        timeout=10,
    )
    if resp.status_code != 200:
        raise Exception(f"DB select error {resp.status_code}: {resp.text}")
    return resp.json()


def db_delete(table: str, filters: dict):
    params = {k: f"eq.{v}" for k, v in filters.items()}
    resp = httpx.delete(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=supa_headers(),
        params=params,
        timeout=10,
    )
    if resp.status_code not in (200, 204):
        raise Exception(f"DB delete error {resp.status_code}: {resp.text}")


def upload_file_to_storage(user_id: str, filename: str, file_bytes: bytes, mime_type: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    storage_path = f"{user_id}/{uuid.uuid4()}{ext}"
    resp = httpx.post(
        f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{storage_path}",
        headers={
            "apikey":        SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type":  mime_type,
            "x-upsert":      "false",
        },
        content=file_bytes,
        timeout=30,
    )
    if resp.status_code not in (200, 201):
        raise Exception(f"Storage upload error {resp.status_code}: {resp.text}")
    return f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{storage_path}"


def delete_file_from_storage(file_url: str):
    if not file_url:
        return
    marker = f"/object/public/{STORAGE_BUCKET}/"
    if marker not in file_url:
        return
    storage_path = file_url.split(marker)[-1]
    resp = httpx.delete(
        f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{storage_path}",
        headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
        timeout=10,
    )
    if resp.status_code not in (200, 204):
        log.warning("Storage delete warning %s: %s", resp.status_code, resp.text)


# ========================
# Auth
# ========================
def get_current_user(auth_header: str | None) -> dict:
    if not auth_header or not auth_header.startswith("Bearer "):
        raise ValueError("Требуется авторизация")
    token = auth_header.removeprefix("Bearer ").strip()
    resp = httpx.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {token}"},
        timeout=10,
    )
    if resp.status_code != 200:
        raise ValueError("Недействительный токен")
    return resp.json()


def try_get_current_user(auth_header: str | None) -> dict | None:
    """Возвращает пользователя если авторизован, иначе None (для публичных эндпоинтов)."""
    if not auth_header:
        return None
    try:
        return get_current_user(auth_header)
    except ValueError:
        return None


# ========================
# Gemini: вызов с автоматическим фолбэком при 429
# ========================
def gemini_generate(models: list, contents: list) -> str:
    """Вызывает Gemini с перебором моделей при ошибке 429 (лимит исчерпан)."""
    last_error = None
    for model in models:
        try:
            response = gemini.models.generate_content(
                model=model,
                contents=contents,
            )
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            if (
                "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()
                or "503" in err_str or "UNAVAILABLE" in err_str
            ):
                last_error = e
                continue  # пробуем следующую модель
            raise  # любая другая ошибка — пробрасываем сразу
    raise Exception(
        f"Сервис временно недоступен: все модели Gemini либо исчерпали лимит запросов, "
        f"либо перегружены. Попробуйте через несколько минут. "
        f"(последняя ошибка: {last_error})"
    )


# ========================
# Gemini: извлечение показателей
# ========================
def extract_indicators_from_file(file_bytes: bytes, filename: str) -> str:
    prompt = """
Ты — парсер медицинских документов. Твоя единственная задача — извлечь все числовые показатели из документа.

ПРАВИЛА:
1. Верни ТОЛЬКО валидный JSON-массив, без каких-либо пояснений, без markdown-блоков.
2. Каждый элемент массива: {"name": "...", "value": "...", "unit": "..."}
3. name — оригинальное название показателя из документа
4. value — только числовое значение (например "5.4")
5. unit — единица измерения (например "г/л", "ммоль/л", "%" и т.п.), если не указана — пустая строка
6. Если в документе нет медицинских показателей — верни пустой массив: []
7. НЕ интерпретируй, НЕ добавляй статус, НЕ пиши ничего кроме JSON.
"""
    return gemini_generate(
        models=GEMINI_MODELS_EXTRACT,
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=get_mime_type(filename)),
            prompt,
        ],
    )


# ========================
# Gemini: анализ показателей → JSON
# ========================
def analyze_verified_indicators(indicators_json: str, age: str, gender: str) -> str:
    prompt = f"""
Ты — медицинский ассистент, анализирующий лабораторные показатели.

ВХОДНЫЕ ДАННЫЕ:
{indicators_json}

Возраст: {age}
Пол: {gender}

ПРАВИЛА:
1. Определи тип анализа (например: "Общий анализ крови", "Биохимический анализ крови", "Гормоны щитовидной железы", "Общий анализ мочи" и т.п.)
2. Определи группу анализа — ОДНО значение из списка: blood | hormones | infections | biomaterials | genetics | microbiome | oncology | functional
   - blood: общий анализ крови, биохимия, коагулограмма, глюкоза, холестерин, СРБ, СОЭ
   - hormones: щитовидная железа, половые гормоны, кортизол, инсулин, пролактин
   - infections: ВИЧ, гепатит, ПЦР, TORCH, антитела к инфекциям, хеликобактер
   - biomaterials: анализ мочи, кал, копрограмма, ПАП-тест
   - genetics: кариотип, BRCA, тромбофилия, ДНК-тесты
   - microbiome: посев, дисбактериоз, антибиограмма, грибки
   - oncology: онкомаркеры (ПСА, CA-125, РЭА, АФП и т.п.)
   - functional: ГТТ, HbA1c, спирометрия, холтер, стресс-тест
3. Для каждого показателя определи нормализованное название на русском и статус: "normal", "above" или "below".
4. Дай краткое общее состояние (1-2 предложения).
5. Дай конкретные рекомендации только при наличии отклонений, без повторов. Если отклонений нет — одна строка: "Все показатели в норме. Продолжайте вести здоровый образ жизни."

ФОРМАТ ОТВЕТА — ТОЛЬКО валидный JSON, без markdown-блоков, без пояснений:

{{
  "analysis_type": "<название типа анализа>",
  "group_key": "<одно из: blood|hormones|infections|biomaterials|genetics|microbiome|oncology|functional>",
  "summary": "<1-2 предложения с общей оценкой>",
  "recommendations": ["<рекомендация 1>", "<рекомендация 2>"],
  "indicators": [
    {{
      "original_name": "<оригинальное название из документа>",
      "name": "<нормализованное название на русском>",
      "value": "<значение с единицей измерения>",
      "status": "<normal|above|below>"
    }}
  ]
}}
"""
    return gemini_generate(
        models=GEMINI_MODELS_ANALYZE,
        contents=[prompt],
    )


# ========================
# Парсинг JSON-ответа анализатора
# ========================
def parse_analysis_result(raw: str) -> dict:
    """
    Парсит JSON-ответ от analyze_verified_indicators.
    Возвращает dict с ключами: analysis_type, group_key, summary, recommendations, indicators.
    При ошибке парсинга возвращает безопасный дефолт.
    """
    data = parse_gemini_json(raw, expect_type=dict)
    if not data:
        log.error("parse_analysis_result: не удалось распарсить JSON: %s", raw[:200])
        return {
            "analysis_type":   "",
            "group_key":       "blood",
            "summary":         "",
            "recommendations": [],
            "indicators":      [],
        }

    # Нормализуем group_key
    group_key = str(data.get("group_key", "blood")).strip().lower()
    if group_key not in VALID_GROUP_KEYS:
        group_key = "blood"

    # Нормализуем статусы показателей
    indicators = []
    for ind in (data.get("indicators") or []):
        status = str(ind.get("status", "normal")).strip().lower()
        if status not in ("normal", "above", "below"):
            status = "normal"
        indicators.append({
            "original_name": str(ind.get("original_name", "") or ind.get("name", "")),
            "name":          str(ind.get("name", "")),
            "value":         str(ind.get("value", "")),
            "status":        status,
        })

    return {
        "analysis_type":   str(data.get("analysis_type", "")),
        "group_key":       group_key,
        "summary":         str(data.get("summary", "")),
        "recommendations": [str(r) for r in (data.get("recommendations") or []) if r],
        "indicators":      indicators,
    }


# ========================
# Supabase: доп. запросы
# ========================
def db_select_filter(table: str, select: str, col: str, val: str) -> list:
    """Выборка с фильтром по одной колонке."""
    resp = httpx.get(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=supa_headers(),
        params={"select": select, col: f"eq.{val}"},
        timeout=10,
    )
    if resp.status_code != 200:
        raise Exception(f"DB select error {resp.status_code}: {resp.text}")
    return resp.json()


def db_upsert(table: str, data: dict, on_conflict: str):
    """INSERT ... ON CONFLICT (on_conflict) DO NOTHING, возвращает строку или None."""
    headers = supa_headers()
    headers["Prefer"] = "resolution=ignore-duplicates,return=representation"
    resp = httpx.post(
        f"{SUPABASE_URL}/rest/v1/{table}?on_conflict={on_conflict}",
        headers=headers,
        json=data,
        timeout=10,
    )
    if resp.status_code not in (200, 201):
        raise Exception(f"DB upsert error {resp.status_code}: {resp.text}")
    rows = resp.json()
    return rows[0] if rows else None


# ========================
# Пакетный resolve показателей
# ========================
def resolve_indicators_batch(indicators: list) -> dict:
    """
    indicators: [{"original_name": ..., "name": ..., "group_key": ...}, ...]
    Возвращает dict: canonical_lower -> indicator_id
    """
    all_indicators = db_select("indicators", "id,name,group_key", {})
    known_by_name: dict = {r["name"].lower(): r for r in all_indicators}

    all_names_rows = db_select("indicator_names", "name,indicator_id", {})
    known_names: dict = {r["name"].lower(): r["indicator_id"] for r in all_names_rows}

    result_map: dict = {}
    unknown: list = []

    for ind in indicators:
        canonical = clean_name(ind["name"])
        original  = ind.get("original_name", canonical)
        group_key = ind.get("group_key", "blood")
        c_lower   = canonical.lower()
        o_lower   = clean_name(original).lower()

        ind_id = known_names.get(c_lower) or known_names.get(o_lower)
        if not ind_id:
            match = known_by_name.get(c_lower)
            if match:
                ind_id = match["id"]

        if ind_id:
            result_map[c_lower] = ind_id
            if o_lower not in known_names and o_lower != c_lower:
                try:
                    db_upsert("indicator_names", {"indicator_id": ind_id, "name": original}, "name")
                    known_names[o_lower] = ind_id
                except Exception as e:
                    log.warning("Синоним не добавлен %s: %s", original, e)
        else:
            unknown.append({"canonical": canonical, "original": original,
                            "group_key": group_key, "c_lower": c_lower})

    if not unknown:
        return result_map

    known_names_list = [r["name"] for r in all_indicators]
    unknown_list_str = json.dumps(
        [{"id": i, "name": u["canonical"], "original": u["original"]}
         for i, u in enumerate(unknown)],
        ensure_ascii=False,
    )

    prompt = f"""Ты — классификатор медицинских показателей.

Список уже известных показателей в системе:
{json.dumps(known_names_list, ensure_ascii=False)}

Новые показатели из анализа — каждый имеет id, название из документа (original) и название которое дал ИИ-анализатор (name):
{unknown_list_str}

Для КАЖДОГО реши одно из двух:
- Это тот же показатель что уже есть в системе (синоним/другое написание) → action="alias", match=точное название из списка известных
- Это новый показатель которого нет в системе → action="new"

Верни ТОЛЬКО JSON-массив без пояснений и markdown:
[{{"id": 0, "action": "alias", "match": "Гемоглобин"}}, {{"id": 1, "action": "new"}}]
"""

    try:
        raw = gemini_generate(GEMINI_MODELS_EXTRACT, [prompt])
        decisions = parse_gemini_json(raw, expect_type=list)
        if not decisions:
            raise ValueError("пустой ответ")
    except Exception as e:
        log.error("resolve_indicators_batch Gemini error: %s", e)
        decisions = [{"id": i, "action": "new"} for i in range(len(unknown))]

    for decision in decisions:
        idx = decision.get("id")
        if idx is None or idx >= len(unknown):
            continue
        u = unknown[idx]
        try:
            if decision.get("action") == "alias":
                match_name = decision.get("match", "")
                matched = next(
                    (r for r in all_indicators if r["name"].lower() == match_name.lower()), None
                )
                if matched:
                    ind_id = matched["id"]
                    for alias in set([u["canonical"], u["original"]]):
                        if alias.lower() not in known_names:
                            try:
                                db_upsert("indicator_names", {"indicator_id": ind_id, "name": alias}, "name")
                                known_names[alias.lower()] = ind_id
                            except Exception:
                                pass
                    result_map[u["c_lower"]] = ind_id
                    log.info("resolve: '%s' → alias '%s'", u["canonical"], match_name)
                    continue

            # new (или alias без match)
            new_ind = db_upsert("indicators", {"name": u["canonical"], "group_key": u["group_key"]}, "name")
            if new_ind:
                ind_id = new_ind["id"]
            else:
                rows = db_select_filter("indicators", "id,name,group_key", "name", u["canonical"])
                if not rows:
                    continue
                ind_id = rows[0]["id"]

            for alias in set([u["canonical"], u["original"]]):
                if alias.lower() not in known_names:
                    try:
                        db_upsert("indicator_names", {"indicator_id": ind_id, "name": alias}, "name")
                        known_names[alias.lower()] = ind_id
                    except Exception:
                        pass

            result_map[u["c_lower"]] = ind_id
            known_by_name[u["c_lower"]] = {"id": ind_id, "name": u["canonical"], "group_key": u["group_key"]}
            log.info("resolve: '%s' → new indicator", u["canonical"])

        except Exception as e:
            log.error("resolve error '%s': %s", u["canonical"], e)

    return result_map


def save_user_indicators(
    user_id: str, analysis_id: str, parsed_indicators: list, group_key: str, measured_at: str | None
):
    """Сохраняет значения показателей пользователя после resolve."""
    if not parsed_indicators:
        return
    to_resolve = [
        {"original_name": ind.get("original_name", ind["name"]),
         "name": ind["name"], "group_key": group_key}
        for ind in parsed_indicators
    ]
    try:
        id_map = resolve_indicators_batch(to_resolve)
    except Exception as e:
        log.error("save_user_indicators resolve failed: %s", e)
        return

    for ind in parsed_indicators:
        canonical = clean_name(ind["name"])
        ind_id = id_map.get(canonical.lower())
        if not ind_id:
            log.warning("нет indicator_id для '%s'", canonical)
            continue
        try:
            row = {
                "user_id": user_id, "indicator_id": ind_id,
                "analysis_id": analysis_id, "value": ind["value"], "status": ind["status"],
            }
            if measured_at:
                row["measured_at"] = measured_at
            db_insert("user_indicators", row)
        except Exception as e:
            log.error("user_indicators insert error '%s': %s", canonical, e)


def save_user_indicators_async(
    user_id: str, analysis_id: str, parsed_indicators: list, group_key: str, measured_at: str | None
):
    """Запускает save_user_indicators в фоновом потоке, не блокируя ответ."""
    t = threading.Thread(
        target=save_user_indicators,
        args=(user_id, analysis_id, parsed_indicators, group_key, measured_at),
        daemon=True,
    )
    t.start()


def check_duplicate_hash(user_id: str, file_hash: str) -> dict | None:
    """Возвращает данные дубликата или None."""
    existing = db_select(
        "analyses",
        select="id,analysis_name,analysis_date",
        filters={"user_id": user_id, "file_hash": file_hash},
    )
    if not existing:
        return None
    dup = existing[0]
    name = dup.get("analysis_name") or "—"
    date = dup.get("analysis_date") or ""
    msg  = f"Этот файл уже загружен как «{name}»"
    if date:
        msg += f" (дата анализа: {date})"
    return {"message": msg}


def read_file_from_request() -> tuple[bytes, str, str]:
    """Читает файл из request.files['file']. Возвращает (bytes, filename, mime_type)."""
    if "file" not in request.files:
        raise ValueError("Файл не найден")
    file = request.files["file"]
    mime_type = get_mime_type(file.filename)
    if mime_type is None:
        raise ValueError("Недопустимый тип файла. Разрешены: PDF, PNG, JPG")
    file.seek(0)
    file_bytes = file.read()
    if not file_bytes:
        raise ValueError("Файл пустой")
    return file_bytes, file.filename, mime_type


# ========================
# API Routes
# ========================

@app.route("/check-duplicate", methods=["POST"])
def check_duplicate():
    try:
        user = get_current_user(request.headers.get("Authorization"))
        file_bytes, filename, _ = read_file_from_request()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        dup = check_duplicate_hash(user["id"], file_hash)
        if dup:
            return jsonify({"duplicate": True, "message": dup["message"], "file_hash": file_hash})
        return jsonify({"duplicate": False, "file_hash": file_hash})
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/extract", methods=["POST"])
def extract():
    """Шаг 1: извлечь показатели из файла. Поддерживает авторизованный и публичный режим."""
    try:
        try_get_current_user(request.headers.get("Authorization"))

        file_bytes, filename, _ = read_file_from_request()
        raw        = extract_indicators_from_file(file_bytes, filename)
        indicators = parse_gemini_json(raw, expect_type=list)
        return jsonify({"indicators": indicators, "raw": raw})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-indicators", methods=["POST"])
def analyze_indicators():
    """Шаг 2: анализ проверенных показателей. Поддерживает авторизованный и публичный режим."""
    try:
        try_get_current_user(request.headers.get("Authorization"))

        indicators_json = request.form.get("indicators", "[]").strip()
        age             = request.form.get("age", "").strip()
        gender          = request.form.get("gender", "").strip()

        if not age or not gender:
            return jsonify({"error": "Возраст или пол не указаны"}), 400
        validate_age(age)

        raw      = analyze_verified_indicators(indicators_json, age, gender)
        analysis = parse_analysis_result(raw)
        return jsonify({"analysis": analysis})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/save-analysis", methods=["POST"])
def save_analysis():
    """Шаг 3: сохранить файл и результат анализа."""
    try:
        user = get_current_user(request.headers.get("Authorization"))

        file_bytes, filename, mime_type = read_file_from_request()
        # analysis — теперь JSON-строка (сериализованный dict от parse_analysis_result)
        analysis_raw  = request.form.get("analysis", "").strip()
        analysis_name = request.form.get("analysis_name", filename).strip()
        analysis_date = request.form.get("analysis_date", "").strip()
        age           = request.form.get("age", "").strip()
        gender        = request.form.get("gender", "").strip()

        if not analysis_raw:
            return jsonify({"error": "Текст анализа отсутствует"}), 400

        # Парсим структурированный результат
        analysis = parse_analysis_result(analysis_raw)

        file_hash = hashlib.sha256(file_bytes).hexdigest()
        dup = check_duplicate_hash(user["id"], file_hash)
        if dup:
            return jsonify({"error": dup["message"]}), 409

        file_url = upload_file_to_storage(user["id"], filename, file_bytes, mime_type)

        row = {
            "user_id":         user["id"],
            "filename":        filename,
            "analysis_name":   analysis_name,
            "age":             age,
            "gender":          gender,
            "result":          analysis_raw,
            "file_url":        file_url,
            "file_hash":       file_hash,
            "summary":         analysis["summary"],
            "recommendations": json.dumps(analysis["recommendations"], ensure_ascii=False),
            "group_key":       analysis["group_key"],
        }
        if analysis_date:
            row["analysis_date"] = analysis_date

        try:
            inserted = db_insert("analyses", row)
        except Exception as db_err:
            log.error("DB insert failed, cleaning storage: %s", db_err)
            delete_file_from_storage(file_url)
            raise

        # Сохраняем показатели в user_indicators асинхронно
        analysis_id = inserted[0]["id"] if inserted else None
        if analysis_id and analysis["indicators"]:
            save_user_indicators_async(
                user_id=user["id"],
                analysis_id=analysis_id,
                parsed_indicators=analysis["indicators"],
                group_key=analysis["group_key"],
                measured_at=analysis_date or None,
            )

        return jsonify({"ok": True})
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def history():
    try:
        user = get_current_user(request.headers.get("Authorization"))
        rows = db_select(
            "analyses",
            # result намеренно исключён — тяжёлое поле, не нужно в списке истории
            select="id,filename,analysis_name,age,gender,file_url,analysis_date,created_at,summary,group_key",
            filters={"user_id": user["id"]},
        )
        return jsonify({"history": rows})
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/analysis/<analysis_id>", methods=["GET"])
def get_analysis(analysis_id):
    try:
        user = get_current_user(request.headers.get("Authorization"))
        rows = db_select(
            "analyses",
            select="id,filename,analysis_name,age,gender,result,file_url,analysis_date,created_at",
            filters={"id": analysis_id, "user_id": user["id"]},
        )
        if not rows:
            return jsonify({"error": "Анализ не найден"}), 404
        return jsonify({"analysis": rows[0]})
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/analysis/<analysis_id>", methods=["DELETE"])
def delete_analysis(analysis_id):
    try:
        user = get_current_user(request.headers.get("Authorization"))
        rows = db_select(
            "analyses",
            select="id,file_url",
            filters={"id": analysis_id, "user_id": user["id"]},
        )
        if not rows:
            return jsonify({"error": "Анализ не найден"}), 404

        file_url = rows[0].get("file_url")
        db_delete("analyses", {"id": analysis_id, "user_id": user["id"]})
        delete_file_from_storage(file_url)
        return jsonify({"ok": True})
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Один запрос: history + indicators + recommendations для личного кабинета."""
    try:
        user = get_current_user(request.headers.get("Authorization"))
        uid  = user["id"]

        # 1. История анализов (без тяжёлого result)
        history = db_select(
            "analyses",
            select="id,filename,analysis_name,age,gender,file_url,analysis_date,created_at,summary,group_key",
            filters={"user_id": uid},
        )

        # 2. Последние показатели из user_indicators
        ind_resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/user_indicators",
            headers=supa_headers(),
            params={
                "select":  "value,status,measured_at,indicator_id,indicators(name,group_key)",
                "user_id": f"eq.{uid}",
                "order":   "measured_at.desc",
            },
            timeout=10,
        )
        ind_rows = ind_resp.json() if ind_resp.status_code == 200 else []
        seen_ind = set()
        indicators = []
        for row in ind_rows:
            ind_id = row.get("indicator_id")
            if ind_id in seen_ind:
                continue
            seen_ind.add(ind_id)
            ind = row.get("indicators") or {}
            indicators.append({
                "name":      ind.get("name", ""),
                "group_key": ind.get("group_key", "blood"),
                "value":     row.get("value", ""),
                "status":    row.get("status", "normal"),
                "date":      row.get("measured_at", ""),
            })
        indicators.sort(key=lambda x: x["name"])

        # 3. Рекомендации из колонки analyses.recommendations
        rec_rows = db_select(
            "analyses",
            select="recommendations,analysis_name,analysis_date",
            filters={"user_id": uid},
        )
        seen_rec = set()
        recommendations = []
        for row in rec_rows:
            raw = row.get("recommendations")
            if not raw:
                continue
            try:
                items = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                continue
            source = row.get("analysis_name", "")
            for text in (items or []):
                if "все показатели в норме" in text.lower():
                    continue
                key = text[:60].lower()
                if key not in seen_rec:
                    seen_rec.add(key)
                    recommendations.append({"text": text, "source": source})

        return jsonify({
            "history":         history,
            "indicators":      indicators,
            "recommendations": recommendations[:20],
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/indicators", methods=["GET"])
def indicators():
    try:
        user = get_current_user(request.headers.get("Authorization"))

        resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/user_indicators",
            headers=supa_headers(),
            params={
                "select":  "value,status,measured_at,indicator_id,analysis_id,indicators(name,group_key)",
                "user_id": f"eq.{user['id']}",
                "order":   "measured_at.desc",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            raise Exception(f"DB error {resp.status_code}: {resp.text}")

        rows = resp.json()

        # Дедуплицируем — берём первую (самую свежую) запись по каждому indicator_id
        seen = set()
        result = []
        for row in rows:
            ind_id = row.get("indicator_id")
            if ind_id in seen:
                continue
            seen.add(ind_id)
            ind = row.get("indicators") or {}
            result.append({
                "name":       ind.get("name", ""),
                "group_key":  ind.get("group_key", "blood"),
                "value":      row.get("value", ""),
                "status":     row.get("status", "normal"),
                "date":       row.get("measured_at", ""),
            })

        result.sort(key=lambda x: x["name"])
        return jsonify({"indicators": result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/indicator-history", methods=["GET"])
def indicator_history():
    """История значений одного показателя по всем анализам (один RPC-запрос)."""
    try:
        user = get_current_user(request.headers.get("Authorization"))
        name = request.args.get("name", "").strip()
        if not name:
            return jsonify({"error": "Параметр name обязателен"}), 400

        resp = httpx.post(
            f"{SUPABASE_URL}/rest/v1/rpc/get_indicator_history",
            headers=supa_headers(),
            json={"p_user_id": user["id"], "p_name": name},
            timeout=10,
        )
        if resp.status_code != 200:
            raise Exception(f"DB error {resp.status_code}: {resp.text}")

        rows = resp.json()
        history = [
            {
                "value":  r.get("value", ""),
                "status": r.get("status", "normal"),
                "date":   r.get("measured_at", ""),
                "source": r.get("analysis_name", ""),
            }
            for r in rows
        ]
        return jsonify({"name": name, "history": history})
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/recommendations", methods=["GET"])
def recommendations():
    try:
        user = get_current_user(request.headers.get("Authorization"))
        rows = db_select(
            "analyses",
            select="recommendations,analysis_name,analysis_date",
            filters={"user_id": user["id"]},
        )

        seen = set()
        recs = []
        for row in rows:
            raw = row.get("recommendations")
            if not raw:
                continue
            try:
                items = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                continue
            source = row.get("analysis_name", "")
            for text in items:
                if "все показатели в норме" in text.lower():
                    continue
                key = text[:60].lower()
                if key not in seen:
                    seen.add(key)
                    recs.append({"text": text, "source": source})

        return jsonify({"recommendations": recs[:20]})
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ========================
# Запуск
# ========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
