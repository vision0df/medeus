"""
Medeus Backend — OpenRouter edition.

Изменения vs Gemini-версии:
  - google-genai заменён на openai SDK (OpenRouter совместим с OpenAI API)
  - _gemini_call → _ai_call (текстовые запросы через chat/completions)
  - extract_indicators_from_file → передаёт файл как base64 image/document
  - PDF передаётся как изображение через url (data URI base64)
  - Переменная окружения: OPENROUTER_API_KEY вместо GEMINI_API_KEY
  - Всё остальное (Supabase, маршруты, логика) — без изменений
"""

import os
import re
import json
import uuid
import base64
import hashlib
import logging
import threading
import traceback

import httpx
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Логирование
# ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Flask
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
SUPABASE_URL       = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY       = os.environ.get("SUPABASE_SERVICE_KEY", "")
STORAGE_BUCKET     = "analyses-files"

log.info("OPENROUTER KEY: %s", "OK" if OPENROUTER_API_KEY else "MISSING")
log.info("SUPABASE URL:   %s", "OK" if SUPABASE_URL       else "MISSING")
log.info("SUPABASE KEY:   %s", "OK" if SUPABASE_KEY       else "MISSING")

# OpenRouter — OpenAI-совместимый клиент
ai_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# Модели OpenRouter (бесплатные, с vision)
# Основная — Qwen 2.5 VL 72B (сильная vision-модель, читает PDF и изображения)
# Резервная — Llama 4 Maverick (Meta, поддерживает vision для изображений)
MODEL_VISION   = "qwen/qwen2.5-vl-72b-instruct:free"  # для извлечения из файлов
MODEL_TEXT     = "qwen/qwen2.5-vl-72b-instruct:free"  # для анализа текста
MODEL_FALLBACK = "meta-llama/llama-4-maverick:free"    # резерв при лимите

VALID_GROUP_KEYS = {
    "blood", "hormones", "infections", "biomaterials",
    "genetics", "microbiome", "oncology", "functional",
}

# White-list расширений → MIME-тип
ALLOWED_EXTENSIONS: dict[str, str] = {
    ".pdf":  "application/pdf",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
}


# ──────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────
def clean_name(raw: str) -> str:
    return (raw or "").strip()


def get_mime_type(filename: str) -> str | None:
    ext = os.path.splitext(filename)[1].lower()
    return ALLOWED_EXTENSIONS.get(ext)


def parse_gemini_json(raw: str, expect_type: type = list):
    """
    Парсит JSON из ответа модели. Снимает markdown-обёртки ```json ... ```.
    Возвращает объект нужного типа или пустой list/dict.
    """
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, expect_type):
            return result
        if expect_type is list and isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list):
                    return v
        return expect_type()
    except Exception:
        return expect_type()


def validate_age(age: str) -> int:
    try:
        v = int(age)
        if not (0 <= v <= 120):
            raise ValueError()
        return v
    except (ValueError, TypeError):
        raise ValueError("Возраст должен быть числом от 0 до 120")


# ──────────────────────────────────────────────
# Supabase helpers
# ──────────────────────────────────────────────
def _supa_headers(content_type: str = "application/json") -> dict:
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  content_type,
        "Prefer":        "return=representation",
    }


def _get(path: str, params: dict | None = None, timeout: int = 10) -> list | dict:
    resp = httpx.get(
        f"{SUPABASE_URL}{path}",
        headers=_supa_headers(),
        params=params or {},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Supabase GET {path} → {resp.status_code}: {resp.text}")
    return resp.json()


def _post(path: str, data: dict | list, headers_extra: dict | None = None, timeout: int = 10):
    h = _supa_headers()
    if headers_extra:
        h.update(headers_extra)
    resp = httpx.post(
        f"{SUPABASE_URL}{path}",
        headers=h,
        json=data,
        timeout=timeout,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Supabase POST {path} → {resp.status_code}: {resp.text}")
    return resp.json()


def _delete(path: str, params: dict | None = None, timeout: int = 10):
    resp = httpx.delete(
        f"{SUPABASE_URL}{path}",
        headers=_supa_headers(),
        params=params or {},
        timeout=timeout,
    )
    if resp.status_code not in (200, 204):
        raise RuntimeError(f"Supabase DELETE {path} → {resp.status_code}: {resp.text}")


def db_insert(table: str, row: dict) -> list:
    return _post(f"/rest/v1/{table}", row)


def db_select(table: str, select: str, filters: dict, order: str = "created_at.desc") -> list:
    params: dict = {"select": select, "order": order}
    for k, v in filters.items():
        params[k] = f"eq.{v}"
    return _get(f"/rest/v1/{table}", params)


def db_delete(table: str, filters: dict):
    params = {k: f"eq.{v}" for k, v in filters.items()}
    _delete(f"/rest/v1/{table}", params)


def db_upsert(table: str, row: dict, on_conflict: str) -> dict | None:
    h = {"Prefer": "resolution=ignore-duplicates,return=representation"}
    result = _post(f"/rest/v1/{table}?on_conflict={on_conflict}", row, headers_extra=h)
    if isinstance(result, list):
        return result[0] if result else None
    return result or None


def upload_to_storage(user_id: str, filename: str, data: bytes, mime: str) -> str:
    ext  = os.path.splitext(filename)[1].lower()
    path = f"{user_id}/{uuid.uuid4()}{ext}"
    resp = httpx.post(
        f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{path}",
        headers={
            "apikey":        SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type":  mime,
            "x-upsert":      "false",
        },
        content=data,
        timeout=30,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Storage upload → {resp.status_code}: {resp.text}")
    return f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{path}"


def delete_from_storage(file_url: str):
    if not file_url:
        return
    marker = f"/object/public/{STORAGE_BUCKET}/"
    if marker not in file_url:
        return
    path = file_url.split(marker, 1)[-1]
    try:
        httpx.delete(
            f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{path}",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
            timeout=10,
        )
    except Exception as e:
        log.warning("storage delete failed: %s", e)


# ──────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────
def get_user(auth_header: str | None) -> dict:
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


def try_get_user(auth_header: str | None) -> dict | None:
    if not auth_header:
        return None
    try:
        return get_user(auth_header)
    except ValueError:
        return None


# ──────────────────────────────────────────────
# AI helpers (OpenRouter)
# ──────────────────────────────────────────────
def _ai_call(messages: list[dict], model: str = MODEL_TEXT) -> str:
    """
    Текстовый запрос к OpenRouter. При лимите (429) пробует резервную модель.
    """
    models_to_try = [model]
    if model != MODEL_FALLBACK:
        models_to_try.append(MODEL_FALLBACK)

    last_err: Exception | None = None
    for m in models_to_try:
        try:
            resp = ai_client.chat.completions.create(
                model=m,
                messages=messages,
                max_tokens=4096,
                timeout=60,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            if any(s in msg for s in ("429", "503", "rate_limit", "overloaded", "quota")):
                last_err = e
                log.warning("AI rate limit on %s, trying fallback: %s", m, msg[:100])
                continue
            raise
    raise RuntimeError(
        "AI сервис временно недоступен (лимит запросов). "
        f"Попробуйте через несколько минут. Последняя ошибка: {last_err}"
    )


def _ai_call_vision(file_bytes: bytes, filename: str, prompt: str) -> str:
    """
    Vision-запрос: передаёт файл (изображение или PDF) как base64 data URI.
    При лимите пробует резервную модель.
    """
    mime = get_mime_type(filename) or "image/jpeg"
    b64  = base64.b64encode(file_bytes).decode("utf-8")

    # Все форматы (PDF, PNG, JPG) передаём как image_url с data URI base64.
    # Gemini через OpenRouter принимает PDF через data URI так же как изображения.
    content_part = {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }

    messages = [
        {
            "role": "user",
            "content": [
                content_part,
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Для PDF резервная модель (Llama 4) не используется — она не понимает PDF.
    # Для изображений — пробуем обе модели.
    if mime == "application/pdf":
        models_to_try = [MODEL_VISION]
    else:
        models_to_try = [MODEL_VISION]
        if MODEL_VISION != MODEL_FALLBACK:
            models_to_try.append(MODEL_FALLBACK)

    last_err: Exception | None = None
    for m in models_to_try:
        try:
            resp = ai_client.chat.completions.create(
                model=m,
                messages=messages,
                max_tokens=4096,
                timeout=90,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            if any(s in msg for s in ("429", "503", "rate_limit", "overloaded", "quota")):
                last_err = e
                log.warning("Vision rate limit on %s, trying fallback: %s", m, msg[:100])
                continue
            raise
    raise RuntimeError(
        "AI сервис временно недоступен (лимит запросов). "
        f"Попробуйте через несколько минут. Последняя ошибка: {last_err}"
    )


def extract_indicators_from_file(file_bytes: bytes, filename: str) -> str:
    prompt = (
        "Ты — парсер медицинских документов. Извлеки название анализа и ВСЕ показатели из документа.\n\n"
        "Верни JSON-объект строго в этом формате (без markdown и пояснений):\n"
        "{\n"
        "  \"analysis_name\": \"краткое название анализа на русском (например: Общий анализ крови, Биохимия, Гормоны щитовидной железы)\",\n"
        "  \"indicators\": [\n"
        "    {\"name\": \"название показателя\", \"value\": \"значение\", \"unit\": \"единица измерения или пустая строка\"}\n"
        "  ]\n"
        "}\n\n"
        "ПРАВИЛА:\n"
        "1. analysis_name — короткое понятное название анализа из документа.\n"
        "2. value — числовое или текстовое значение. Если пусто или прочерк — ставь \"-\".\n"
        "3. unit — единица измерения или пустая строка.\n"
        "4. Если показателей нет — indicators: [].\n"
        "5. НЕ интерпретируй, НЕ добавляй статус, только JSON."
    )
    return _ai_call_vision(file_bytes, filename, prompt)


def analyze_indicators(indicators_json: str, age: str, gender: str) -> str:
    groups_desc = (
        "blood=анализы крови, hormones=гормоны, infections=инфекции и иммунитет, "
        "biomaterials=биоматериалы (моча/кал/слюна), genetics=генетика, "
        "microbiome=микрофлора, oncology=онкомаркеры, functional=функциональные тесты"
    )
    prompt = (
        f"Ты — медицинский ассистент. Анализируй лабораторные показатели.\n\n"
        f"Возраст: {age}, Пол: {gender}\n\n"
        f"Показатели:\n{indicators_json}\n\n"
        "Верни JSON строго в этом формате:\n"
        "{\n"
        "  \"analysis_type\": \"...\" ,\n"
        "  \"group_key\": \"одно значение из: blood|hormones|infections|biomaterials|genetics|microbiome|oncology|functional\",\n"
        "  \"summary\": \"1-2 предложения с общей оценкой\",\n"
        "  \"recommendations\": [\"...\"],\n"
        "  \"indicators\": [\n"
        "    {\n"
        "      \"original_name\": \"оригинальное название\",\n"
        "      \"name\": \"нормализованное название на русском\",\n"
        "      \"value\": \"значение с единицей измерения\",\n"
        "      \"status\": \"норма|выше нормы|ниже нормы|отклонение\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Группы: {groups_desc}\n\n"
        "ТОЛЬКО JSON, без текста до и после."
    )
    return _ai_call([{"role": "user", "content": prompt}], model=MODEL_TEXT)


def _normalize_status(raw: str) -> str:
    s = str(raw or "").strip().lower()
    exact = {"норма": "normal", "выше нормы": "above", "ниже нормы": "below", "отклонение": "deviation"}
    if s in exact:
        return exact[s]
    if s in ("normal", "above", "below", "deviation"):
        return s
    if any(w in s for w in ("выше", "high", "повышен")):
        return "above"
    if any(w in s for w in ("ниже", "low", "понижен")):
        return "below"
    if any(w in s for w in ("откл", "abnormal", "патол")):
        return "deviation"
    return "normal"


def parse_analysis_result(raw: str) -> dict:
    data = parse_gemini_json(raw, expect_type=dict)
    if not data:
        log.error("parse_analysis_result: не удалось распарсить: %s", raw[:300])
        return {
            "analysis_type": "", "group_key": "blood",
            "summary": "", "recommendations": [], "indicators": [],
        }

    group_key = str(data.get("group_key", "blood")).strip().lower()
    if group_key not in VALID_GROUP_KEYS:
        group_key = "blood"

    indicators = []
    for ind in (data.get("indicators") or []):
        indicators.append({
            "original_name": str(ind.get("original_name") or ind.get("name", "")),
            "name":          str(ind.get("name", "")),
            "value":         str(ind.get("value", "")),
            "status":        _normalize_status(ind.get("status", "норма")),
        })

    return {
        "analysis_type":   str(data.get("analysis_type", "")),
        "group_key":       group_key,
        "summary":         str(data.get("summary", "")),
        "recommendations": [str(r) for r in (data.get("recommendations") or []) if r],
        "indicators":      indicators,
    }


# ──────────────────────────────────────────────
# Resolve indicators (batch)
# ──────────────────────────────────────────────
def _resolve_batch(indicators: list[dict]) -> dict[str, str]:
    def _rk(name_lo: str, gk: str) -> str:
        return f"{name_lo}||{gk}"

    all_inds  = db_select("indicators",      "id,name,group_key", {}, order="name.asc")
    all_names = db_select("indicator_names", "name,indicator_id,group_key", {}, order="name.asc")

    by_name:  dict[str, str] = {_rk(r["name"].lower(), r["group_key"]): r["id"] for r in all_inds}
    by_alias: dict[str, str] = {
        _rk(r["name"].lower(), r["group_key"]): r["indicator_id"]
        for r in all_names
        if r.get("group_key")
    }
    inds_by_id: dict[str, dict] = {r["id"]: r for r in all_inds}
    known: dict[str, str] = {**by_name, **by_alias}

    result_map: dict[str, str] = {}
    unknown: list[dict] = []

    for ind in indicators:
        canonical = clean_name(ind["name"])
        original  = clean_name(ind.get("original_name", canonical))
        gk        = ind.get("group_key", "blood")
        c_lo      = canonical.lower()
        o_lo      = original.lower()

        ind_id = known.get(_rk(c_lo, gk)) or known.get(_rk(o_lo, gk))
        if ind_id:
            result_map[_rk(c_lo, gk)] = ind_id
            if o_lo != c_lo and _rk(o_lo, gk) not in known:
                try:
                    db_upsert(
                        "indicator_names",
                        {"indicator_id": ind_id, "name": original, "group_key": gk},
                        "name,group_key",
                    )
                    known[_rk(o_lo, gk)] = ind_id
                except Exception as e:
                    log.debug("alias insert skip (%s): %s", original, e)
        else:
            unknown.append({
                "canonical": canonical,
                "original":  original,
                "group_key": gk,
                "c_lo":      c_lo,
                "o_lo":      o_lo,
            })

    if not unknown:
        return result_map

    used_groups  = {u["group_key"] for u in unknown}
    known_subset = [
        {"name": r["name"], "group_key": r["group_key"]}
        for r in all_inds
        if r["group_key"] in used_groups
    ]

    unknown_payload = json.dumps(
        [
            {"id": i, "name": u["canonical"], "original": u["original"], "group_key": u["group_key"]}
            for i, u in enumerate(unknown)
        ],
        ensure_ascii=False,
    )
    prompt = (
        f"Ты — классификатор медицинских показателей.\n\n"
        f"Уже существующие показатели (name + group_key):\n"
        f"{json.dumps(known_subset, ensure_ascii=False)}\n\n"
        f"Новые показатели:\n{unknown_payload}\n\n"
        f"Для каждого реши (сравнивай ТОЛЬКО внутри той же group_key):\n"
        f"- Синоним существующего → action=\"alias\", match=\"<точное name из списка>\"\n"
        f"- Новый → action=\"new\"\n\n"
        f"Верни ровно {len(unknown)} элементов:\n"
        f"[{{\"id\": 0, \"action\": \"alias\", \"match\": \"Лейкоциты\"}}, {{\"id\": 1, \"action\": \"new\"}}]\n\n"
        f"ТОЛЬКО JSON."
    )

    try:
        raw_decisions = _ai_call([{"role": "user", "content": prompt}])
        decisions: list = parse_gemini_json(raw_decisions, expect_type=list)
        if not decisions:
            raise ValueError("пустой ответ")
    except Exception as e:
        log.error("_resolve_batch AI error: %s", e)
        decisions = [{"id": i, "action": "new"} for i in range(len(unknown))]

    inds_by_name_gk: dict[str, dict] = {
        _rk(r["name"].lower(), r["group_key"]): r for r in all_inds
    }

    for dec in decisions:
        idx = dec.get("id")
        if idx is None or idx >= len(unknown):
            continue
        u = unknown[idx]
        gk = u["group_key"]
        try:
            if dec.get("action") == "alias":
                match_lo = dec.get("match", "").lower()
                matched  = inds_by_name_gk.get(_rk(match_lo, gk))
                if matched:
                    ind_id = matched["id"]
                    for alias in {u["canonical"], u["original"]}:
                        rk = _rk(alias.lower(), gk)
                        if rk not in known:
                            try:
                                db_upsert(
                                    "indicator_names",
                                    {"indicator_id": ind_id, "name": alias, "group_key": gk},
                                    "name,group_key",
                                )
                                known[rk] = ind_id
                            except Exception:
                                pass
                    result_map[_rk(u["c_lo"], gk)] = ind_id
                    log.info("resolve alias: '%s' [%s] → '%s'", u["canonical"], gk, matched["name"])
                    continue

            new_row = db_upsert(
                "indicators",
                {"name": u["canonical"], "group_key": gk},
                "name,group_key",
            )
            if new_row:
                ind_id = new_row["id"]
            else:
                rows = db_select("indicators", "id,group_key", {"name": u["canonical"], "group_key": gk})
                if not rows:
                    continue
                ind_id = rows[0]["id"]

            for alias in {u["canonical"], u["original"]}:
                rk = _rk(alias.lower(), gk)
                if rk not in known:
                    try:
                        db_upsert(
                            "indicator_names",
                            {"indicator_id": ind_id, "name": alias, "group_key": gk},
                            "name,group_key",
                        )
                        known[rk] = ind_id
                    except Exception:
                        pass

            result_map[_rk(u["c_lo"], gk)] = ind_id
            new_entry = {"id": ind_id, "name": u["canonical"], "group_key": gk}
            inds_by_name_gk[_rk(u["c_lo"], gk)] = new_entry
            inds_by_id[ind_id]                   = new_entry
            log.info("resolve new: '%s' [%s]", u["canonical"], gk)

        except Exception as e:
            log.error("resolve error '%s': %s", u["canonical"], e)

    return result_map


def _save_user_indicators(
    user_id: str,
    analysis_id: str,
    indicators: list[dict],
    group_key: str,
    measured_at: str | None,
):
    if not indicators:
        return

    to_resolve = [
        {
            "original_name": ind.get("original_name", ind["name"]),
            "name":          ind["name"],
            "group_key":     group_key,
        }
        for ind in indicators
    ]
    try:
        id_map = _resolve_batch(to_resolve)
    except Exception as e:
        log.error("_save_user_indicators resolve failed: %s", e)
        return

    for ind in indicators:
        canonical = clean_name(ind["name"])
        ind_id    = id_map.get(f"{canonical.lower()}||{group_key}")
        if not ind_id:
            log.warning("нет indicator_id для '%s'", canonical)
            continue
        try:
            row: dict = {
                "user_id":      user_id,
                "indicator_id": ind_id,
                "analysis_id":  analysis_id,
                "value":        ind["value"],
                "status":       ind["status"],
                "group_key":    group_key,
            }
            if measured_at:
                row["measured_at"] = measured_at
            db_insert("user_indicators", row)
        except Exception as e:
            log.error("user_indicators insert '%s': %s", canonical, e)


def _save_user_indicators_async(
    user_id: str,
    analysis_id: str,
    indicators: list[dict],
    group_key: str,
    measured_at: str | None,
):
    t = threading.Thread(
        target=_save_user_indicators,
        args=(user_id, analysis_id, indicators, group_key, measured_at),
        daemon=True,
    )
    t.start()


# ──────────────────────────────────────────────
# File helpers
# ──────────────────────────────────────────────
def read_uploaded_file() -> tuple[bytes, str, str]:
    if "file" not in request.files:
        raise ValueError("Файл не найден в запросе")
    f = request.files["file"]
    if not f or not f.filename:
        raise ValueError("Файл не выбран")
    mime = get_mime_type(f.filename)
    if mime is None:
        raise ValueError("Недопустимый тип файла. Разрешены: PDF, PNG, JPG/JPEG")
    f.seek(0)
    data = f.read()
    if not data:
        raise ValueError("Файл пустой")
    return data, f.filename, mime


def check_duplicate(user_id: str, file_hash: str) -> str | None:
    rows = db_select("analyses", "id,analysis_name,analysis_date", {"user_id": user_id, "file_hash": file_hash})
    if not rows:
        return None
    dup  = rows[0]
    name = dup.get("analysis_name") or "—"
    date = dup.get("analysis_date") or ""
    msg  = f"Этот файл уже загружен как «{name}»"
    if date:
        msg += f" (дата анализа: {date})"
    return msg


def _parse_recommendations(raw) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(r) for r in raw if r]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(r) for r in parsed if r]
    except Exception:
        pass
    return []


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/check-duplicate", methods=["POST"])
def route_check_duplicate():
    try:
        user      = get_user(request.headers.get("Authorization"))
        data, _, _= read_uploaded_file()
        h         = hashlib.sha256(data).hexdigest()
        msg       = check_duplicate(user["id"], h)
        if msg:
            return jsonify({"duplicate": True, "message": msg, "file_hash": h})
        return jsonify({"duplicate": False, "file_hash": h})
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/extract", methods=["POST"])
def route_extract():
    try:
        try_get_user(request.headers.get("Authorization"))
        data, filename, _ = read_uploaded_file()
        raw        = extract_indicators_from_file(data, filename)
        parsed     = parse_gemini_json(raw, expect_type=dict)
        if isinstance(parsed, dict):
            indicators    = parsed.get("indicators", [])
            analysis_name = parsed.get("analysis_name", "")
        else:
            indicators    = parse_gemini_json(raw, expect_type=list)
            analysis_name = ""
        return jsonify({"indicators": indicators, "analysis_name": analysis_name})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-indicators", methods=["POST"])
def route_analyze_indicators():
    try:
        try_get_user(request.headers.get("Authorization"))

        indicators_json = request.form.get("indicators", "[]").strip()
        age             = request.form.get("age", "").strip()
        gender          = request.form.get("gender", "").strip()

        if not age or not gender:
            return jsonify({"error": "Не указаны возраст или пол"}), 400
        validate_age(age)

        raw      = analyze_indicators(indicators_json, age, gender)
        analysis = parse_analysis_result(raw)
        return jsonify({"analysis": analysis})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/save-analysis", methods=["POST"])
def route_save_analysis():
    try:
        user = get_user(request.headers.get("Authorization"))

        analysis_raw  = request.form.get("analysis", "").strip()
        analysis_name = request.form.get("analysis_name", "").strip()
        analysis_date = request.form.get("analysis_date", "").strip()
        age           = request.form.get("age", "").strip()
        gender        = request.form.get("gender", "").strip()

        if not analysis_raw:
            return jsonify({"error": "Отсутствует результат анализа"}), 400

        analysis = parse_analysis_result(analysis_raw)

        file_url  = None
        filename  = None
        file_hash = None

        if "file" in request.files and request.files["file"].filename:
            try:
                file_bytes, filename, mime = read_uploaded_file()
                file_hash = hashlib.sha256(file_bytes).hexdigest()
                dup_msg   = check_duplicate(user["id"], file_hash)
                if dup_msg:
                    return jsonify({"error": dup_msg}), 409
                file_url = upload_to_storage(user["id"], filename, file_bytes, mime)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400

        if not analysis_name:
            analysis_name = filename or "Анализ"

        row: dict = {
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
            log.error("DB insert failed, rolling back storage: %s", db_err)
            delete_from_storage(file_url)
            raise

        analysis_id = (inserted[0]["id"] if isinstance(inserted, list) and inserted
                       else inserted.get("id") if isinstance(inserted, dict) else None)

        if analysis_id and analysis["indicators"]:
            _save_user_indicators_async(
                user_id     = user["id"],
                analysis_id = analysis_id,
                indicators  = analysis["indicators"],
                group_key   = analysis["group_key"],
                measured_at = analysis_date or None,
            )

        return jsonify({"ok": True})
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except RuntimeError as e:
        msg = str(e)
        if "409" in msg or "duplicate" in msg.lower():
            return jsonify({"error": msg}), 409
        log.error(traceback.format_exc())
        return jsonify({"error": msg}), 500
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def route_history():
    try:
        user = get_user(request.headers.get("Authorization"))
        rows = db_select(
            "analyses",
            "id,filename,analysis_name,age,gender,file_url,analysis_date,created_at,summary,group_key",
            {"user_id": user["id"]},
        )
        return jsonify({"history": rows})
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/analysis/<analysis_id>", methods=["GET"])
def route_get_analysis(analysis_id: str):
    try:
        user = get_user(request.headers.get("Authorization"))
        rows = db_select(
            "analyses",
            "id,filename,analysis_name,age,gender,result,file_url,analysis_date,created_at",
            {"id": analysis_id, "user_id": user["id"]},
        )
        if not rows:
            return jsonify({"error": "Анализ не найден"}), 404
        return jsonify({"analysis": rows[0]})
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/analysis/<analysis_id>", methods=["DELETE"])
def route_delete_analysis(analysis_id: str):
    try:
        user = get_user(request.headers.get("Authorization"))
        rows = db_select("analyses", "id,file_url", {"id": analysis_id, "user_id": user["id"]})
        if not rows:
            return jsonify({"error": "Анализ не найден"}), 404

        file_url = rows[0].get("file_url")
        db_delete("analyses", {"id": analysis_id, "user_id": user["id"]})
        delete_from_storage(file_url)
        return jsonify({"ok": True})
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard", methods=["GET"])
def route_dashboard():
    try:
        user = get_user(request.headers.get("Authorization"))
        uid  = user["id"]

        history = db_select(
            "analyses",
            "id,filename,analysis_name,age,gender,file_url,analysis_date,created_at,summary,group_key",
            {"user_id": uid},
        )

        ind_rows = _get(
            "/rest/v1/user_indicators",
            params={
                "select":  "value,status,measured_at,group_key,indicator_id,indicators(name)",
                "user_id": f"eq.{uid}",
                "order":   "measured_at.desc",
            },
        )
        if not isinstance(ind_rows, list):
            ind_rows = []

        seen_ind:    set[str] = set()
        indicators: list[dict] = []
        for row in ind_rows:
            ind_id = row.get("indicator_id")
            if ind_id in seen_ind:
                continue
            seen_ind.add(ind_id)
            ind = row.get("indicators") or {}
            indicators.append({
                "name":      ind.get("name", ""),
                "group_key": row.get("group_key", "blood"),
                "value":     row.get("value", ""),
                "status":    row.get("status", "normal"),
                "date":      row.get("measured_at", ""),
            })
        indicators.sort(key=lambda x: x["name"])

        rec_rows = db_select(
            "analyses",
            "recommendations,analysis_name,analysis_date",
            {"user_id": uid},
        )
        seen_rec: set[str] = set()
        recommendations: list[dict] = []
        for row in rec_rows:
            items  = _parse_recommendations(row.get("recommendations"))
            source = row.get("analysis_name", "")
            for text in items:
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
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/indicators", methods=["GET"])
def route_indicators():
    try:
        user = get_user(request.headers.get("Authorization"))

        rows = _get(
            "/rest/v1/user_indicators",
            params={
                "select":  "value,status,measured_at,group_key,indicator_id,analysis_id,indicators(name),analyses(analysis_name)",
                "user_id": f"eq.{user['id']}",
                "order":   "measured_at.desc",
            },
        )
        if not isinstance(rows, list):
            rows = []

        seen: set[str] = set()
        result: list[dict] = []
        for row in rows:
            ind_id = row.get("indicator_id")
            if ind_id in seen:
                continue
            seen.add(ind_id)
            ind      = row.get("indicators") or {}
            analysis = row.get("analyses")   or {}
            result.append({
                "name":      ind.get("name", ""),
                "group_key": row.get("group_key", "blood"),
                "value":     row.get("value", ""),
                "status":    row.get("status", "normal"),
                "date":      row.get("measured_at", ""),
                "source":    analysis.get("analysis_name", ""),
            })
        result.sort(key=lambda x: x["name"])
        return jsonify({"indicators": result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/indicator-history", methods=["GET"])
def route_indicator_history():
    try:
        user = get_user(request.headers.get("Authorization"))
        name = request.args.get("name", "").strip()
        if not name:
            return jsonify({"error": "Параметр name обязателен"}), 400

        rows = _post(
            "/rest/v1/rpc/get_indicator_history",
            {"p_user_id": user["id"], "p_name": name},
        )
        if not isinstance(rows, list):
            rows = []

        history = [
            {
                "value":       r.get("value", ""),
                "status":      r.get("status", "normal"),
                "date":        r.get("measured_at", ""),
                "source":      r.get("analysis_name", ""),
                "analysis_id": r.get("analysis_id", ""),
            }
            for r in rows
        ]

        description = ""
        ind_rows = db_select("indicators", "description", {"name": name})
        if ind_rows and ind_rows[0].get("description"):
            description = ind_rows[0]["description"]

        return jsonify({"name": name, "history": history, "description": description})
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/recommendations", methods=["GET"])
def route_recommendations():
    try:
        user = get_user(request.headers.get("Authorization"))
        rows = db_select(
            "analyses",
            "recommendations,analysis_name,analysis_date",
            {"user_id": user["id"]},
        )

        seen: set[str] = set()
        recs: list[dict] = []
        for row in rows:
            items  = _parse_recommendations(row.get("recommendations"))
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
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# Background: auto-fill descriptions for indicators
# ──────────────────────────────────────────────

def _fetch_indicator_no_description() -> dict | None:
    h = _supa_headers()
    for flt_val in ("is.null", "eq."):
        r = httpx.get(
            f"{SUPABASE_URL}/rest/v1/indicators",
            headers=h,
            params={"select": "id,name,group_key", "description": flt_val,
                    "limit": "1", "order": "created_at.asc"},
            timeout=10,
        )
        if r.status_code == 200:
            rows = r.json()
            if rows:
                return rows[0]
    return None


def _fetch_analysis_names_for_indicator(indicator_id: str) -> list[str]:
    try:
        rows = _get(
            "/rest/v1/user_indicators",
            params={
                "select":       "analyses(analysis_name)",
                "indicator_id": f"eq.{indicator_id}",
                "order":        "created_at.desc",
                "limit":        "20",
            },
        )
    except Exception:
        return []

    seen: list[str] = []
    for row in (rows if isinstance(rows, list) else []):
        analysis = row.get("analyses") or {}
        name = analysis.get("analysis_name", "").strip()
        if name and name not in seen:
            seen.append(name)
        if len(seen) >= 3:
            break
    return seen


def _build_description_prompt(indicator_name: str, analysis_names: list[str]) -> str:
    analyses_str = ", ".join(analysis_names) if analysis_names else "неизвестно"
    return (
        f"Показатель: {indicator_name} из анализов: {analyses_str}\n\n"
        "Ответь строго в формате JSON (без markdown, только объект):\n"
        "{\n"
        '  "about": "1-3 предложения — что это за показатель",\n'
        '  "norms": "норма для разных групп: мужчины, женщины, дети",\n'
        '  "deviations": "с чем могут быть связаны отклонения",\n'
        '  "improvement": "как можно улучшить состояние"\n'
        "}\n\n"
        "Язык: русский. Каждое поле — одно-два предложения, чистый текст без списков."
    )


def _parse_description_response(raw: str) -> str | None:
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1]
            clean = clean.rsplit("```", 1)[0]
        data = json.loads(clean)
    except Exception:
        log.warning("[desc-worker] не удалось распарсить JSON: %s", raw[:200])
        return None

    required = ("about", "norms", "deviations", "improvement")
    if not all(k in data and isinstance(data[k], str) and data[k].strip() for k in required):
        log.warning("[desc-worker] JSON неполный: %s", list(data.keys()))
        return None

    parts = [
        data["about"].strip(),
        "Нормы: " + data["norms"].strip(),
        "Отклонения: " + data["deviations"].strip(),
        "Улучшение: " + data["improvement"].strip(),
    ]
    return "\n\n".join(parts)


def _patch_indicator_description(indicator_id: str, description: str | None) -> None:
    h = _supa_headers()
    h["Prefer"] = "return=minimal"
    resp = httpx.patch(
        f"{SUPABASE_URL}/rest/v1/indicators",
        headers=h,
        params={"id": f"eq.{indicator_id}"},
        json={"description": description},
        timeout=15,
    )
    if resp.status_code not in (200, 204):
        raise RuntimeError(
            f"PATCH indicators {indicator_id} → {resp.status_code}: {resp.text}"
        )


def _description_worker() -> None:
    import time

    log.info("[desc-worker] запущен, старт через 40 сек")
    time.sleep(40)

    while True:
        wait_next = 60

        try:
            ind = _fetch_indicator_no_description()

            if ind is None:
                log.info("[desc-worker] все показатели заполнены, жду 10 мин")
                time.sleep(600)
                continue

            ind_id   = ind["id"]
            ind_name = ind["name"]

            log.info("[desc-worker] обрабатываю: '%s'", ind_name)

            try:
                analysis_names = _fetch_analysis_names_for_indicator(ind_id)
                prompt = _build_description_prompt(ind_name, analysis_names)
                raw = _ai_call([{"role": "user", "content": prompt}])
                description = _parse_description_response(raw)

                if description is None:
                    log.warning("[desc-worker] невалидный ответ для '%s', жду 10 мин", ind_name)
                    wait_next = 600
                else:
                    _patch_indicator_description(ind_id, description[:1500])
                    log.info("[desc-worker] сохранено: '%s'", ind_name)

            except Exception as e:
                log.error("[desc-worker] ошибка для '%s': %s", ind_name, e)
                wait_next = 600

        except Exception as e:
            log.error("[desc-worker] критическая ошибка: %s", e)
            wait_next = 600

        time.sleep(wait_next)


_desc_thread = threading.Thread(target=_description_worker, daemon=True, name="desc-worker")
_desc_thread.start()


# ──────────────────────────────────────────────
# Entry point (dev only)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
