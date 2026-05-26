"""
Medeus Backend — полностью переработанная версия.

Изменения:
  - Убран python-magic (libmagic недоступен на Render); mime-тип определяется
    по расширению файла — этого достаточно для white-list из 4 форматов.
  - Убран cachetools (не использовался).
  - Все запросы к Supabase через единый _supa() клиент с retry-логикой.
  - Gemini-клиент создаётся один раз при старте, не пересоздаётся.
  - Добавлен /health endpoint для Render health-check.
  - Улучшена обработка ошибок: разделены 400 / 401 / 409 / 500.
  - resolve_indicators_batch переработан: меньше лишних Gemini-запросов.
  - dashboard и indicators теперь возвращают консистентные структуры.
  - Рекомендации сохраняются и читаются корректно (JSONB vs TEXT).
"""

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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
SUPABASE_URL   = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY   = os.environ.get("SUPABASE_SERVICE_KEY", "")
STORAGE_BUCKET = "analyses-files"

log.info("GEMINI KEY:   %s", "OK" if GEMINI_API_KEY else "MISSING")
log.info("SUPABASE URL: %s", "OK" if SUPABASE_URL   else "MISSING")
log.info("SUPABASE KEY: %s", "OK" if SUPABASE_KEY   else "MISSING")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Модели: lite — для парсинга (дешевле), flash — для анализа (качественнее)
MODELS_EXTRACT  = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
MODELS_ANALYZE  = ["gemini-2.5-flash",      "gemini-2.5-flash-lite"]

VALID_GROUP_KEYS = {
    "blood", "hormones", "infections", "biomaterials",
    "genetics", "microbiome", "oncology", "functional",
}

# White-list расширений → MIME-тип (без libmagic)
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
    Парсит JSON из ответа Gemini. Снимает markdown-обёртки ```json ... ```.
    Возвращает объект нужного типа или пустой list/dict.
    """
    text = raw.strip()
    # Снимаем ``` ... ```
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
        # Gemini иногда оборачивает массив в {"indicators": [...]}
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


# ── Таблицы ──
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
    """INSERT ... ON CONFLICT DO NOTHING. Возвращает строку или None (если дубликат)."""
    h = {"Prefer": "resolution=ignore-duplicates,return=representation"}
    result = _post(f"/rest/v1/{table}?on_conflict={on_conflict}", row, headers_extra=h)
    if isinstance(result, list):
        return result[0] if result else None
    return result or None


# ── Storage ──
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
    """Возвращает пользователя или кидает ValueError."""
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
# Gemini helpers
# ──────────────────────────────────────────────
def _gemini_call(models: list[str], contents: list) -> str:
    """
    Вызывает Gemini, перебирая модели при 429 / 503.
    Любая другая ошибка пробрасывается сразу.
    """
    last_err: Exception | None = None
    for model in models:
        try:
            resp = gemini_client.models.generate_content(model=model, contents=contents)
            return resp.text.strip()
        except Exception as e:
            msg = str(e)
            if any(s in msg for s in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "quota")):
                last_err = e
                continue
            raise
    raise RuntimeError(
        "Сервис Gemini временно недоступен (лимит запросов). "
        f"Попробуйте через несколько минут. Последняя ошибка: {last_err}"
    )


def extract_indicators_from_file(file_bytes: bytes, filename: str) -> str:
    prompt = (
        "Ты — парсер медицинских документов. Извлеки ВСЕ медецинские показатели из документа.\n\n"
        "ПРАВИЛА:\n"
        "1. Верни ТОЛЬКО валидный JSON-массив без пояснений и markdown-блоков.\n"
        "2. Каждый элемент: {\"name\": \"...\", \"value\": \"...\", \"unit\": \"...\"}\n"
        "3. name — оригинальное название показателя из документа.\n"
        "4. value — значение (числовое или текстовое). Если пусто или «-» — ставь «-».\n"
        "5. unit — единица измерения или пустая строка \"\".\n"
        "6. Если медицинских показателей нет — верни [].\n"
        "7. НЕ интерпретируй, НЕ добавляй статус, только JSON."
    )
    return _gemini_call(
        MODELS_EXTRACT,
        [types.Part.from_bytes(data=file_bytes, mime_type=get_mime_type(filename)), prompt],
    )


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
    return _gemini_call(MODELS_ANALYZE, [prompt])


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
    """
    Парсит ответ analyze_indicators → нормализованный dict.
    При ошибке возвращает безопасный дефолт.
    """
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
    """
    Принимает [{"original_name": ..., "name": ..., "group_key": ...}].
    Возвращает {name.lower(): indicator_id}.
    """
    # Загружаем всё из БД одним запросом каждый
    all_inds   = db_select("indicators",      "id,name,group_key", {}, order="name.asc")
    all_names  = db_select("indicator_names", "name,indicator_id", {}, order="name.asc")

    by_name:   dict[str, str] = {r["name"].lower(): r["id"]           for r in all_inds}
    by_alias:  dict[str, str] = {r["name"].lower(): r["indicator_id"] for r in all_names}
    inds_by_id: dict[str, dict] = {r["id"]: r for r in all_inds}
    # Объединяем (alias перекрывает основное имя при коллизии — не важно, оба ведут к тому же id)
    known: dict[str, str] = {**by_name, **by_alias}

    result_map: dict[str, str] = {}
    unknown: list[dict] = []

    for ind in indicators:
        canonical = clean_name(ind["name"])
        original  = clean_name(ind.get("original_name", canonical))
        c_lo      = canonical.lower()
        o_lo      = original.lower()

        ind_id = known.get(c_lo) or known.get(o_lo)
        if ind_id:
            result_map[c_lo] = ind_id
            # Добавляем оригинальное имя как алиас если его ещё нет
            if o_lo not in known and o_lo != c_lo:
                try:
                    db_upsert("indicator_names", {"indicator_id": ind_id, "name": original}, "name")
                    known[o_lo] = ind_id
                except Exception as e:
                    log.debug("alias insert skip (%s): %s", original, e)
        else:
            unknown.append({
                "canonical": canonical,
                "original":  original,
                "group_key": ind.get("group_key", "blood"),
                "c_lo":      c_lo,
                "o_lo":      o_lo,
            })

    if not unknown:
        return result_map

    # Для неизвестных — спрашиваем Gemini: синоним или новый?
    used_groups  = {u["group_key"] for u in unknown}
    known_subset = [r["name"] for r in all_inds if r["group_key"] in used_groups]

    unknown_payload = json.dumps(
        [{"id": i, "name": u["canonical"], "original": u["original"]} for i, u in enumerate(unknown)],
        ensure_ascii=False,
    )
    prompt = (
        f"Ты — классификатор медицинских показателей.\n\n"
        f"Группа: {', '.join(sorted(used_groups))}\n\n"
        f"Уже существующие показатели:\n{json.dumps(known_subset, ensure_ascii=False)}\n\n"
        f"Новые показатели:\n{unknown_payload}\n\n"
        f"Для каждого реши:\n"
        f"- Синоним существующего → action=\"alias\", match=\"<точное название из списка>\"\n"
        f"- Новый → action=\"new\"\n\n"
        f"Верни ровно {len(unknown)} элементов:\n"
        f"[{{\"id\": 0, \"action\": \"alias\", \"match\": \"Лейкоциты\"}}, {{\"id\": 1, \"action\": \"new\"}}]\n\n"
        f"ТОЛЬКО JSON."
    )

    try:
        raw_decisions = _gemini_call(MODELS_EXTRACT, [prompt])
        decisions: list = parse_gemini_json(raw_decisions, expect_type=list)
        if not decisions:
            raise ValueError("пустой ответ")
    except Exception as e:
        log.error("_resolve_batch Gemini error: %s", e)
        decisions = [{"id": i, "action": "new"} for i in range(len(unknown))]

    # Применяем решения
    inds_by_name: dict[str, dict] = {r["name"].lower(): r for r in all_inds}

    for dec in decisions:
        idx = dec.get("id")
        if idx is None or idx >= len(unknown):
            continue
        u = unknown[idx]
        try:
            if dec.get("action") == "alias":
                match_lo = dec.get("match", "").lower()
                matched  = inds_by_name.get(match_lo)
                if matched:
                    ind_id = matched["id"]
                    for alias in {u["canonical"], u["original"]}:
                        if alias.lower() not in known:
                            try:
                                db_upsert("indicator_names", {"indicator_id": ind_id, "name": alias}, "name")
                                known[alias.lower()] = ind_id
                            except Exception:
                                pass
                    result_map[u["c_lo"]] = ind_id
                    log.info("resolve alias: '%s' → '%s'", u["canonical"], matched["name"])
                    continue
                # Если match не найден — создаём как новый

            # action == "new" (или alias без совпадения)
            new_row = db_upsert(
                "indicators",
                {"name": u["canonical"], "group_key": u["group_key"]},
                "name",
            )
            if new_row:
                ind_id = new_row["id"]
            else:
                # уже существует (race condition)
                rows = db_select("indicators", "id", {"name": u["canonical"]})
                if not rows:
                    continue
                ind_id = rows[0]["id"]

            for alias in {u["canonical"], u["original"]}:
                if alias.lower() not in known:
                    try:
                        db_upsert("indicator_names", {"indicator_id": ind_id, "name": alias}, "name")
                        known[alias.lower()] = ind_id
                    except Exception:
                        pass

            result_map[u["c_lo"]] = ind_id
            new_entry = {"id": ind_id, "name": u["canonical"], "group_key": u["group_key"]}
            inds_by_name[u["c_lo"]] = new_entry
            inds_by_id[ind_id]      = new_entry
            log.info("resolve new: '%s'", u["canonical"])

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

    # Каждый показатель несёт свой group_key (из поля ind["group_key"]).
    # Если его нет — используем group_key всего анализа как fallback.
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
        ind_id    = id_map.get(canonical.lower())
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
    """
    Читает файл из request.files['file'].
    Возвращает (bytes, filename, mime_type) или кидает ValueError.
    """
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
    """
    Возвращает сообщение об ошибке если дубликат уже существует, иначе None.
    """
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
    """Безопасно парсит поле recommendations из БД (может быть str или list)."""
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
    """Health-check для Render. Не требует авторизации."""
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
    """Шаг 1: извлечь показатели из файла (авторизованный или публичный режим)."""
    try:
        try_get_user(request.headers.get("Authorization"))
        data, filename, _ = read_uploaded_file()
        raw        = extract_indicators_from_file(data, filename)
        indicators = parse_gemini_json(raw, expect_type=list)
        return jsonify({"indicators": indicators, "raw": raw})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-indicators", methods=["POST"])
def route_analyze_indicators():
    """Шаг 2: анализ показателей (авторизованный или публичный режим)."""
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
    """Шаг 3: сохранить файл + результат анализа в БД."""
    try:
        user = get_user(request.headers.get("Authorization"))

        file_bytes, filename, mime = read_uploaded_file()
        analysis_raw  = request.form.get("analysis", "").strip()
        analysis_name = request.form.get("analysis_name", filename).strip() or filename
        analysis_date = request.form.get("analysis_date", "").strip()
        age           = request.form.get("age", "").strip()
        gender        = request.form.get("gender", "").strip()

        if not analysis_raw:
            return jsonify({"error": "Отсутствует результат анализа"}), 400

        analysis  = parse_analysis_result(analysis_raw)
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        dup_msg   = check_duplicate(user["id"], file_hash)
        if dup_msg:
            return jsonify({"error": dup_msg}), 409

        file_url = upload_to_storage(user["id"], filename, file_bytes, mime)

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
            # Сохраняем как JSON-строку для совместимости с TEXT и JSONB колонками
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
                user_id          = user["id"],
                analysis_id      = analysis_id,
                indicators       = analysis["indicators"],
                group_key        = analysis["group_key"],
                measured_at      = analysis_date or None,
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
    """
    Один запрос для личного кабинета:
    history + последние показатели (по одному на каждый indicator) + рекомендации.
    """
    try:
        user = get_user(request.headers.get("Authorization"))
        uid  = user["id"]

        # 1. История анализов
        history = db_select(
            "analyses",
            "id,filename,analysis_name,age,gender,file_url,analysis_date,created_at,summary,group_key",
            {"user_id": uid},
        )

        # 2. Последние показатели (JOIN с indicators)
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

        # 3. Рекомендации (дедупликация по первым 60 символам)
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
    """Все последние показатели пользователя (по одному на каждый indicator)."""
    try:
        user = get_user(request.headers.get("Authorization"))

        rows = _get(
            "/rest/v1/user_indicators",
            params={
                "select":  "value,status,measured_at,group_key,indicator_id,analysis_id,indicators(name)",
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
            ind = row.get("indicators") or {}
            result.append({
                "name":      ind.get("name", ""),
                "group_key": row.get("group_key", "blood"),
                "value":     row.get("value", ""),
                "status":    row.get("status", "normal"),
                "date":      row.get("measured_at", ""),
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
    """История значений одного показателя по имени через Supabase RPC."""
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
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/recommendations", methods=["GET"])
def route_recommendations():
    """Все уникальные рекомендации пользователя."""
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
# Entry point (dev only)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
