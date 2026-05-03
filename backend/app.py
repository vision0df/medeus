import os
import hashlib
import traceback
import httpx
import magic
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, origins=["https://medeus.vercel.app"])

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# ── Rate limiter (in-memory, per user IP) ──
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Файл превышает максимальный размер 10 МБ"}), 413

@app.errorhandler(429)
def rate_limit_hit(e):
    return jsonify({"error": "Слишком много запросов. Попробуйте через несколько минут."}), 429

# ── Wake-up / health check ──
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True}), 200

# ========================
# Ключи
# ========================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL   = os.environ.get("SUPABASE_URL")
SUPABASE_KEY   = os.environ.get("SUPABASE_SERVICE_KEY")
STORAGE_BUCKET = "analyses-files"

print("GEMINI KEY:",  "OK" if GEMINI_API_KEY else "MISSING", flush=True)
print("SUPABASE URL:", "OK" if SUPABASE_URL  else "MISSING", flush=True)
print("SUPABASE KEY:", "OK" if SUPABASE_KEY  else "MISSING", flush=True)

gemini = genai.Client(api_key=GEMINI_API_KEY)

SUPA_HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "return=representation",
}

# ========================
# Helpers
# ========================
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
ALLOWED_MIMES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
}
EXT_TO_MIME = {
    ".pdf":  "application/pdf",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
}

def get_mime_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return EXT_TO_MIME.get(ext, "image/jpeg")

def validate_file_type(filename: str, file_bytes: bytes) -> str:
    """Проверяет расширение и реальный тип файла. Возвращает mime или бросает ValueError."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Неподдерживаемый формат файла. Разрешены: PDF, PNG, JPG.")
    # Проверяем реальный content-type через magic bytes
    try:
        real_mime = magic.from_buffer(file_bytes[:2048], mime=True)
        if real_mime not in ALLOWED_MIMES:
            raise ValueError(f"Содержимое файла не соответствует расширению {ext}.")
    except Exception as e:
        if "не соответствует" in str(e):
            raise
        # Если python-magic не установлен — не блокируем, просто логируем
        print(f"⚠️ magic check skipped: {e}", flush=True)
    return EXT_TO_MIME.get(ext, "image/jpeg")


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


def db_insert(table: str, data: dict):
    resp = httpx.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=SUPA_HEADERS,
        json=data,
        timeout=10,
    )
    if resp.status_code not in (200, 201):
        raise Exception(f"DB insert error {resp.status_code}: {resp.text}")
    return resp.json()


def db_select(table: str, select: str, filters: dict) -> list:
    params = {"select": select, **{k: f"eq.{v}" for k, v in filters.items()},
              "order": "created_at.desc"}
    resp = httpx.get(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=SUPA_HEADERS,
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
        headers=SUPA_HEADERS,
        params=params,
        timeout=10,
    )
    if resp.status_code not in (200, 204):
        raise Exception(f"DB delete error {resp.status_code}: {resp.text}")


def upload_file_to_storage(user_id: str, filename: str, file_bytes: bytes, mime_type: str) -> str:
    """Загружает файл в Supabase Storage и возвращает публичный URL."""
    import uuid
    ext = os.path.splitext(filename)[1].lower()
    storage_path = f"{user_id}/{uuid.uuid4()}{ext}"

    resp = httpx.post(
        f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{storage_path}",
        headers={
            "apikey":         SUPABASE_KEY,
            "Authorization":  f"Bearer {SUPABASE_KEY}",
            "Content-Type":   mime_type,
            "x-upsert":       "false",
        },
        content=file_bytes,
        timeout=30,
    )
    if resp.status_code not in (200, 201):
        raise Exception(f"Storage upload error {resp.status_code}: {resp.text}")

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{storage_path}"
    return public_url


def delete_file_from_storage(file_url: str):
    """Удаляет файл из Supabase Storage по его публичному URL."""
    if not file_url:
        return
    marker = f"/object/public/{STORAGE_BUCKET}/"
    if marker not in file_url:
        return
    storage_path = file_url.split(marker)[-1]
    resp = httpx.delete(
        f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{storage_path}",
        headers={
            "apikey":        SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        },
        timeout=10,
    )
    if resp.status_code not in (200, 204):
        print(f"⚠️ Storage delete warning {resp.status_code}: {resp.text}", flush=True)


# ========================
# Gemini анализ
# ========================
def analyze_with_gemini(file_bytes: bytes, filename: str, age: str, gender: str) -> str:
    print("🧠 Gemini START", flush=True)

    prompt = f"""Ты — медицинский ассистент. Проанализируй документ с результатами анализов.

Возраст пациента: {age}
Пол пациента: {gender}

Верни ТОЛЬКО валидный JSON следующей структуры (без markdown, без пояснений, только JSON):
{{
  "indicators": [
    {{"name": "Гемоглобин", "value": "140 г/л", "status": "норма"}},
    {{"name": "Лейкоциты", "value": "12.5 10^9/л", "status": "выше нормы"}}
  ],
  "summary": "Общее состояние пациента в 2-3 предложениях.",
  "attention": "На что стоит обратить внимание — 1-3 пункта.",
  "recommendations": [
    "Первая рекомендация",
    "Вторая рекомендация"
  ]
}}

Поле status может быть только одним из: "норма", "выше нормы", "ниже нормы".
Если документ не является медицинским анализом — верни {{"error": "Документ не содержит медицинских показателей"}}.
"""

    mime_type = get_mime_type(filename)

    response = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            prompt,
        ],
    )

    print("✅ Gemini done", flush=True)

    # Парсим JSON-ответ и конвертируем в текст для обратной совместимости
    raw = response.text.strip()
    # Убираем markdown-обёртку если модель всё же добавила
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    import json
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: вернуть как есть если JSON не распарсился
        print("⚠️ Gemini returned non-JSON, using raw text", flush=True)
        return raw

    if "error" in parsed:
        raise ValueError(parsed["error"])

    # Сериализуем обратно в структурированный текст (для совместимости с парсером dashboard)
    lines = []
    for ind in parsed.get("indicators", []):
        lines.append(f"{ind['name']} - {ind['value']} - {ind['status']}")

    if parsed.get("summary"):
        lines.append(f"\n— Общее состояние\n{parsed['summary']}")
    if parsed.get("attention"):
        lines.append(f"\n— На что стоит обратить внимание\n{parsed['attention']}")
    if parsed.get("recommendations"):
        lines.append("\n— Рекомендации")
        for rec in parsed["recommendations"]:
            lines.append(f"• {rec}")

    return "\n".join(lines)


# ========================
# API: /analyze
# ========================
@app.route("/analyze", methods=["POST"])
@limiter.limit("10 per hour")
def analyze():
    try:
        print("🔥 /analyze HIT", flush=True)

        user = get_current_user(request.headers.get("Authorization"))
        print(f"👤 user: {user['id']}", flush=True)

        if "file" not in request.files:
            return jsonify({"error": "Файл не найден"}), 400

        file          = request.files["file"]
        age           = request.form.get("age", "").strip()
        gender        = request.form.get("gender", "").strip()
        analysis_name = request.form.get("analysis_name", file.filename).strip()
        analysis_date = request.form.get("analysis_date", "").strip()

        if not age or not gender:
            return jsonify({"error": "Возраст или пол не указаны"}), 400

        print(f"📥 file={file.filename}  age={age}  gender={gender}  name={analysis_name}  date={analysis_date}", flush=True)

        file.seek(0)
        file_bytes = file.read()

        # ── Валидация типа файла (расширение + magic bytes) ──
        mime_type = validate_file_type(file.filename, file_bytes)

        # ── Проверяем дубликат по SHA-256 ──
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        existing = db_select(
            "analyses",
            select="id,analysis_name,analysis_date",
            filters={"user_id": user["id"], "file_hash": file_hash},
        )
        if existing:
            dup = existing[0]
            dup_name = dup.get("analysis_name") or "—"
            dup_date = dup.get("analysis_date") or ""
            msg = f"Этот файл уже загружен как «{dup_name}»"
            if dup_date:
                msg += f" (дата анализа: {dup_date})"
            return jsonify({"error": msg}), 409

        # ── Gemini анализ (до загрузки файла — экономим Storage если Gemini упадёт) ──
        analysis = analyze_with_gemini(file_bytes, file.filename, age, gender)

        # ── Загружаем файл в Storage ──
        file_url = upload_file_to_storage(user["id"], file.filename, file_bytes, mime_type)
        print(f"📦 File uploaded: {file_url}", flush=True)

        # ── Сохраняем в БД (если падает — удаляем файл из Storage) ──
        row = {
            "user_id":       user["id"],
            "filename":      file.filename,
            "analysis_name": analysis_name,
            "age":           age,
            "gender":        gender,
            "result":        analysis,
            "file_url":      file_url,
            "file_hash":     file_hash,
        }
        if analysis_date:
            row["analysis_date"] = analysis_date

        try:
            db_insert("analyses", row)
        except Exception as db_err:
            print(f"💥 DB insert failed, cleaning up Storage: {db_err}", flush=True)
            delete_file_from_storage(file_url)
            raise

        print("💾 Saved to DB", flush=True)

        return jsonify({"analysis": analysis})

    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        print("🔥 ERROR:", flush=True)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ========================
# API: /history
# ========================
@app.route("/history", methods=["GET"])
def history():
    try:
        user = get_current_user(request.headers.get("Authorization"))

        rows = db_select(
            "analyses",
            select="id,filename,analysis_name,age,gender,result,file_url,analysis_date,created_at",
            filters={"user_id": user["id"]},
        )

        return jsonify({"history": rows})

    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ========================
# API: /analysis/<id>  — получить один анализ
# ========================
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


# ========================
# API: /analysis/<id>  — удалить анализ
# ========================
@app.route("/analysis/<analysis_id>", methods=["DELETE"])
def delete_analysis(analysis_id):
    try:
        user = get_current_user(request.headers.get("Authorization"))

        # Проверяем что анализ принадлежит этому пользователю и берём file_url
        rows = db_select(
            "analyses",
            select="id,file_url",
            filters={"id": analysis_id, "user_id": user["id"]},
        )

        if not rows:
            return jsonify({"error": "Анализ не найден"}), 404

        file_url = rows[0].get("file_url")

        # Удаляем из БД
        db_delete("analyses", {"id": analysis_id, "user_id": user["id"]})
        print(f"🗑️ Deleted from DB: {analysis_id}", flush=True)

        # Удаляем файл из Storage
        delete_file_from_storage(file_url)
        print(f"🗑️ Deleted from Storage: {file_url}", flush=True)

        return jsonify({"ok": True})

    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ========================
# API: /dashboard — всё за один запрос
# ========================
@app.route("/dashboard", methods=["GET"])
def dashboard():
    """
    Один запрос к БД — возвращает history + indicators + recommendations.
    Используется личным кабинетом вместо трёх отдельных вызовов.
    """
    try:
        user = get_current_user(request.headers.get("Authorization"))

        rows = db_select(
            "analyses",
            select="id,filename,analysis_name,age,gender,result,file_url,analysis_date,created_at",
            filters={"user_id": user["id"]},
        )

        # ── History ──────────────────────────────────────────────────────────
        history = rows  # уже готово

        # ── Indicators ───────────────────────────────────────────────────────
        merged: dict = {}
        for row in rows:
            result_text = row.get("result", "") or ""
            row_date    = row.get("analysis_date") or ""
            source      = row.get("analysis_name", "")

            for line in result_text.splitlines():
                line = line.strip()
                if not line or line.startswith("—") or line.startswith("-"):
                    continue
                parts = [p.strip() for p in line.split(" - ")]
                if len(parts) < 3:
                    parts = [p.strip() for p in line.split(" — ")]
                if len(parts) < 3:
                    continue

                name, value, status = parts[0], parts[1], parts[2].lower()

                if len(name) < 2 or len(name) > 80:
                    continue
                if not any(c.isdigit() for c in value):
                    continue

                if "выше" in status:
                    norm_status = "above"
                elif "ниже" in status:
                    norm_status = "below"
                else:
                    norm_status = "normal"

                name_key = name.lower().strip()
                existing = merged.get(name_key)
                if existing is None or row_date > existing["date"]:
                    merged[name_key] = {
                        "name":   name,
                        "value":  value,
                        "status": norm_status,
                        "date":   row_date,
                        "source": source,
                    }

        indicators = sorted(merged.values(), key=lambda x: x["name"])

        # ── Recommendations ──────────────────────────────────────────────────
        REC_START_HEADERS = {"рекоменда"}
        REC_STOP_HEADERS  = {
            "обратить внимание", "общее состояние",
            "на что стоит", "вывод", "заключение",
        }
        seen_keys: set = set()
        recs: list     = []

        for row in sorted(rows, key=lambda r: r.get("analysis_date") or "", reverse=True):
            result_text = row.get("result", "") or ""
            source      = row.get("analysis_name", "")
            in_rec      = False

            for line in result_text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                low = stripped.lower()

                if any(h in low for h in REC_START_HEADERS):
                    in_rec = True
                    continue
                if in_rec and any(h in low for h in REC_STOP_HEADERS):
                    in_rec = False
                    continue
                if in_rec and " - " in stripped and any(c.isdigit() for c in stripped):
                    in_rec = False
                if in_rec and len(stripped) > 15:
                    clean = stripped.lstrip("•·–—-→* ").strip()
                    if len(clean) < 15:
                        continue
                    key = clean[:60].lower()
                    if key not in seen_keys:
                        seen_keys.add(key)
                        recs.append({"text": clean, "source": source})

        return jsonify({
            "history":         history,
            "indicators":      indicators,
            "recommendations": recs[:20],
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ========================
# API: /indicators — сводка уникальных показателей
# ========================
@app.route("/indicators", methods=["GET"])
def indicators():
    """
    Парсим все result из БД пользователя.
    Формат строки в result:
      Название - значение - статус
    Берём последнее значение по analysis_date для каждого уникального показателя.
    """
    try:
        user = get_current_user(request.headers.get("Authorization"))

        rows = db_select(
            "analyses",
            select="result,analysis_date,analysis_name",
            filters={"user_id": user["id"]},
        )

        # Словарь: имя_показателя -> {value, unit, status, date, source}
        merged: dict = {}

        for row in rows:
            result_text = row.get("result", "") or ""
            row_date    = row.get("analysis_date") or ""
            source      = row.get("analysis_name", "")

            for line in result_text.splitlines():
                line = line.strip()
                if not line or line.startswith("—") or line.startswith("-"):
                    continue
                # Ищем паттерн: Название - значение - статус
                parts = [p.strip() for p in line.split(" - ")]
                if len(parts) < 3:
                    parts = [p.strip() for p in line.split(" — ")]
                if len(parts) < 3:
                    continue

                name   = parts[0]
                value  = parts[1]
                status = parts[2].lower()

                # Фильтруем явно нечисловые / служебные строки
                if len(name) < 2 or len(name) > 80:
                    continue
                if not any(c.isdigit() for c in value):
                    continue

                # Нормализуем статус
                if "выше" in status:
                    norm_status = "above"
                elif "ниже" in status:
                    norm_status = "below"
                elif "норм" in status:
                    norm_status = "normal"
                else:
                    norm_status = "normal"

                name_key = name.lower().strip()

                # Обновляем если запись новее
                existing = merged.get(name_key)
                if existing is None or row_date > existing["date"]:
                    merged[name_key] = {
                        "name":   name,
                        "value":  value,
                        "status": norm_status,
                        "date":   row_date,
                        "source": source,
                    }

        result_list = sorted(merged.values(), key=lambda x: x["name"])
        return jsonify({"indicators": result_list})

    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ========================
# API: /recommendations — уникальные рекомендации
# ========================
@app.route("/recommendations", methods=["GET"])
def recommendations():
    """
    Извлекаем блок рекомендаций из каждого result.
    Gemini пишет их после таблицы показателей, под заголовками вроде
    'Рекомендации', 'На что обратить внимание', 'Общее состояние'.
    Дедублируем по смыслу (первые 60 символов как ключ).
    """
    try:
        user = get_current_user(request.headers.get("Authorization"))

        rows = db_select(
            "analyses",
            select="result,analysis_date,analysis_name",
            filters={"user_id": user["id"]},
        )

        seen_keys: set = set()
        recs: list     = []

        # Заголовки секций рекомендаций от Gemini
        REC_START_HEADERS = {"рекоменда"}
        REC_STOP_HEADERS  = {
            "обратить внимание", "общее состояние",
            "на что стоит", "вывод", "заключение",
        }

        for row in sorted(rows, key=lambda r: r.get("analysis_date") or "", reverse=True):
            result_text = row.get("result", "") or ""
            source      = row.get("analysis_name", "")
            in_rec      = False

            for line in result_text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue

                low = stripped.lower()

                # Переключаемся в режим рекомендаций
                if any(h in low for h in REC_START_HEADERS):
                    in_rec = True
                    continue

                # Выходим если началась другая секция
                if in_rec and any(h in low for h in REC_STOP_HEADERS):
                    in_rec = False
                    continue

                # Выходим из блока если снова таблица (содержит " - " с цифрами)
                if in_rec and " - " in stripped and any(c.isdigit() for c in stripped):
                    in_rec = False

                if in_rec and len(stripped) > 15:
                    # Убираем маркеры списка
                    clean = stripped.lstrip("•·–—-→* ").strip()
                    if len(clean) < 15:
                        continue
                    key = clean[:60].lower()
                    if key not in seen_keys:
                        seen_keys.add(key)
                        recs.append({
                            "text":   clean,
                            "source": source,
                        })

        return jsonify({"recommendations": recs[:20]})  # не более 20

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
