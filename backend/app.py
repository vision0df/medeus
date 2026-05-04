import os
import uuid       # ИСПРАВЛЕНИЕ #9: импорт на уровне модуля
import hashlib    # ИСПРАВЛЕНИЕ #9: импорт на уровне модуля
import traceback
import httpx
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ИСПРАВЛЕНИЕ #10: разрешаем localhost для локальной разработки
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

# ИСПРАВЛЕНИЕ #2: SUPA_HEADERS через функцию — не строится при старте модуля
# когда SUPABASE_KEY ещё может быть None.
def get_supa_headers() -> dict:
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }

# ========================
# ИСПРАВЛЕНИЕ #3: разрешённые типы файлов — словарь вместо fallback "image/jpeg"
# ========================
ALLOWED_MIME_TYPES = {
    ".pdf":  "application/pdf",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
}

# ========================
# Helpers
# ========================
def get_mime_type(filename: str) -> str | None:
    """Возвращает mime-тип файла или None если расширение не разрешено."""
    ext = os.path.splitext(filename)[1].lower()
    return ALLOWED_MIME_TYPES.get(ext)  # None для неизвестных расширений


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
        headers=get_supa_headers(),  # ИСПРАВЛЕНИЕ #2
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
        headers=get_supa_headers(),  # ИСПРАВЛЕНИЕ #2
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
        headers=get_supa_headers(),  # ИСПРАВЛЕНИЕ #2
        params=params,
        timeout=10,
    )
    if resp.status_code not in (200, 204):
        raise Exception(f"DB delete error {resp.status_code}: {resp.text}")


def upload_file_to_storage(user_id: str, filename: str, file_bytes: bytes, mime_type: str) -> str:
    """Загружает файл в Supabase Storage и возвращает публичный URL."""
    # ИСПРАВЛЕНИЕ #9: uuid импортирован на уровне модуля
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
# ШАГ 1: Извлечение показателей из документа
# ========================
def extract_indicators_from_file(file_bytes: bytes, filename: str) -> str:
    """
    Только извлекает сырые показатели из файла.
    Возвращает JSON-массив объектов {name, value, unit}.
    """
    print("🔍 Gemini EXTRACT START", flush=True)

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

    mime_type = get_mime_type(filename)

    response = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            prompt,
        ],
    )

    print("✅ Gemini EXTRACT done", flush=True)
    return response.text.strip()


# ========================
# ШАГ 2: Анализ проверенных показателей + рекомендации
# ========================
def analyze_verified_indicators(indicators_json: str, age: str, gender: str) -> str:
    """
    Принимает проверенные пользователем показатели (JSON),
    нормализует их и возвращает полный анализ с рекомендациями.
    """
    print("🧠 Gemini ANALYZE START", flush=True)

    prompt = f"""
Ты — медицинский ассистент, анализирующий лабораторные показатели.

ВХОДНЫЕ ДАННЫЕ:
{indicators_json}

Возраст: {age}
Пол: {gender}

ПРАВИЛА:
Для каждого показателя:
   - Нормализуй название (общепринятое медицинское название на русском языке, для использования как ключи в базе данных.).
   - Определи статус:
     ("норма", "выше нормы", "ниже нормы")
   - Оценка должна учитывать возраст и пол.

ФОРМАТ ОТВЕТА (строго соблюдать):

БЛОК 1 — таблица показателей:
Нормализованное название - значение с единицей - статус 
...

БЛОК 2 — краткий анализ:
Общее состояние: <одно краткое предложение>

Рекомендации:
- [рекомендация 1]
- [рекомендация 2]
...

ОГРАНИЧЕНИЯ:
- Без вступлений
- Только факты из анализа
- Рекомендации только по отклонениям (если всё в норме — рекомендации оставь пустыми)
"""

    response = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
    )

    print("✅ Gemini ANALYZE done", flush=True)
    return response.text


# ========================
# API: /check-duplicate — быстрая проверка файла до вызова ИИ
# ========================
@app.route("/check-duplicate", methods=["POST"])
def check_duplicate():
    """
    Принимает файл, считает SHA-256 и проверяет есть ли такой уже в БД.
    Не вызывает Gemini, не пишет в Storage/БД — только хэш + запрос.
    """
    try:
        user = get_current_user(request.headers.get("Authorization"))

        if "file" not in request.files:
            return jsonify({"error": "Файл не найден"}), 400

        file = request.files["file"]

        mime_type = get_mime_type(file.filename)
        if mime_type is None:
            return jsonify({"error": "Недопустимый тип файла. Разрешены: PDF, PNG, JPG"}), 400

        file.seek(0)
        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"error": "Файл пустой"}), 400

        file_hash = hashlib.sha256(file_bytes).hexdigest()
        existing = db_select(
            "analyses",
            select="id,analysis_name,analysis_date",
            filters={"user_id": user["id"], "file_hash": file_hash},
        )

        if existing:
            dup      = existing[0]
            dup_name = dup.get("analysis_name") or "—"
            dup_date = dup.get("analysis_date") or ""
            msg = f"Этот файл уже загружен как «{dup_name}»"
            if dup_date:
                msg += f" (дата анализа: {dup_date})"
            return jsonify({"duplicate": True, "message": msg, "file_hash": file_hash})

        return jsonify({"duplicate": False, "file_hash": file_hash})

    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ========================
# API: /extract — ШАГ 1: извлечь показатели из файла
# ========================
@app.route("/extract", methods=["POST"])
def extract():
    """
    Принимает файл, возвращает JSON-массив сырых показателей.
    Файл НЕ сохраняется в Storage и БД — это только предпросмотр.
    """
    try:
        print("🔍 /extract HIT", flush=True)

        user = get_current_user(request.headers.get("Authorization"))

        if "file" not in request.files:
            return jsonify({"error": "Файл не найден"}), 400

        file = request.files["file"]

        mime_type = get_mime_type(file.filename)
        if mime_type is None:
            return jsonify({"error": "Недопустимый тип файла. Разрешены: PDF, PNG, JPG"}), 400

        file.seek(0)
        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"error": "Файл пустой"}), 400

        raw = extract_indicators_from_file(file_bytes, file.filename)

        # Пробуем распарсить JSON от Gemini
        import json as _json
        try:
            # Убираем возможные markdown-блоки если Gemini всё же добавил
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
                clean = clean.strip()
            indicators = _json.loads(clean)
            if not isinstance(indicators, list):
                indicators = []
        except Exception:
            indicators = []

        return jsonify({"indicators": indicators, "raw": raw})

    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        print("🔥 /extract ERROR:", flush=True)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ========================
# API: /analyze-indicators — ШАГ 2: анализ проверенных показателей + сохранение
# ========================
@app.route("/analyze-indicators", methods=["POST"])
def analyze_indicators():
    """
    ШАГ 2: Только анализирует показатели через Gemini.
    НЕ сохраняет в Storage и НЕ пишет в БД.
    Принимает: indicators (JSON), age, gender
    """
    try:
        print("🧠 /analyze-indicators HIT", flush=True)
        user = get_current_user(request.headers.get("Authorization"))

        indicators_json = request.form.get("indicators", "[]").strip()
        age             = request.form.get("age", "").strip()
        gender          = request.form.get("gender", "").strip()

        if not age or not gender:
            return jsonify({"error": "Возраст или пол не указаны"}), 400
        try:
            age_int = int(age)
            if not (0 <= age_int <= 120):
                raise ValueError()
        except ValueError:
            return jsonify({"error": "Возраст должен быть числом от 0 до 120"}), 400

        print(f"📊 indicators: {indicators_json[:200]}", flush=True)
        analysis = analyze_verified_indicators(indicators_json, age, gender)
        return jsonify({"analysis": analysis})

    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        print("🔥 /analyze-indicators ERROR:", flush=True)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ========================
# API: /save-analysis — ШАГ 3: сохранить файл и результат (только по кнопке «Сохранить»)
# ========================
@app.route("/save-analysis", methods=["POST"])
def save_analysis():
    """
    ШАГ 3: Загружает файл в Storage и сохраняет запись в БД.
    Вызывается только когда пользователь нажимает «Сохранить».
    Принимает: file, analysis, analysis_name, analysis_date, age, gender
    """
    try:
        print("💾 /save-analysis HIT", flush=True)
        user = get_current_user(request.headers.get("Authorization"))
        print(f"👤 user: {user['id']}", flush=True)

        if "file" not in request.files:
            return jsonify({"error": "Файл не найден"}), 400

        file          = request.files["file"]
        analysis      = request.form.get("analysis", "").strip()
        analysis_name = request.form.get("analysis_name", file.filename).strip()
        analysis_date = request.form.get("analysis_date", "").strip()
        age           = request.form.get("age", "").strip()
        gender        = request.form.get("gender", "").strip()

        if not analysis:
            return jsonify({"error": "Текст анализа отсутствует"}), 400

        mime_type = get_mime_type(file.filename)
        if mime_type is None:
            return jsonify({"error": "Недопустимый тип файла"}), 400

        file.seek(0)
        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"error": "Файл пустой"}), 400

        # Проверка дубликата — здесь, а не в /extract, чтобы не блокировать просмотр
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        existing = db_select(
            "analyses",
            select="id,analysis_name,analysis_date",
            filters={"user_id": user["id"], "file_hash": file_hash},
        )
        if existing:
            dup      = existing[0]
            dup_name = dup.get("analysis_name") or "—"
            dup_date = dup.get("analysis_date") or ""
            msg = f"Этот файл уже загружен как «{dup_name}»"
            if dup_date:
                msg += f" (дата анализа: {dup_date})"
            return jsonify({"error": msg}), 409

        file_url = upload_file_to_storage(user["id"], file.filename, file_bytes, mime_type)
        print(f"📦 File uploaded: {file_url}", flush=True)

        try:
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
            db_insert("analyses", row)
            print("✅ Saved to DB", flush=True)
        except Exception as db_err:
            print(f"💥 DB insert failed, cleaning up storage: {db_err}", flush=True)
            delete_file_from_storage(file_url)
            raise

        return jsonify({"ok": True})

    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        print("🔥 /save-analysis ERROR:", flush=True)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ========================
# API: /analyze — устаревший эндпоинт, оставлен для обратной совместимости
# ========================
@app.route("/analyze", methods=["POST"])
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

        # ИСПРАВЛЕНИЕ #4: валидация возраста — целое число от 0 до 120
        try:
            age_int = int(age)
            if not (0 <= age_int <= 120):
                raise ValueError()
        except ValueError:
            return jsonify({"error": "Возраст должен быть числом от 0 до 120"}), 400

        # ИСПРАВЛЕНИЕ #3: валидация типа файла на бэкенде
        mime_type = get_mime_type(file.filename)
        if mime_type is None:
            return jsonify({"error": "Недопустимый тип файла. Разрешены: PDF, PNG, JPG"}), 400

        print(f"📥 file={file.filename}  age={age}  gender={gender}  name={analysis_name}  date={analysis_date}", flush=True)

        file.seek(0)
        file_bytes = file.read()

        # ИСПРАВЛЕНИЕ #3: проверяем что файл не пустой
        if not file_bytes:
            return jsonify({"error": "Файл пустой"}), 400

        # Проверяем дубликат файла по SHA-256 хэшу
        # ИСПРАВЛЕНИЕ #9: hashlib импортирован на уровне модуля
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        existing = db_select(
            "analyses",
            select="id,analysis_name,analysis_date",
            filters={"user_id": user["id"], "file_hash": file_hash},
        )
        if existing:
            dup      = existing[0]
            dup_name = dup.get("analysis_name") or "—"
            dup_date = dup.get("analysis_date") or ""
            msg = f"Этот файл уже загружен как «{dup_name}»"
            if dup_date:
                msg += f" (дата анализа: {dup_date})"
            return jsonify({"error": msg}), 409

        # Сначала извлекаем показатели, затем анализируем
        import json as _json
        raw = extract_indicators_from_file(file_bytes, file.filename)
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"): clean = clean[4:]
                clean = clean.strip()
            indicators_list = _json.loads(clean)
            if not isinstance(indicators_list, list): indicators_list = []
        except Exception:
            indicators_list = []
        indicators_json = _json.dumps(indicators_list, ensure_ascii=False)
        analysis = analyze_verified_indicators(indicators_json, age, gender)

        file_url = upload_file_to_storage(user["id"], file.filename, file_bytes, mime_type)
        print(f"📦 File uploaded: {file_url}", flush=True)

        # ИСПРАВЛЕНИЕ #5: если DB упала после загрузки файла — удаляем файл из Storage
        try:
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

            db_insert("analyses", row)
            print("💾 Saved to DB", flush=True)
        except Exception as db_err:
            print(f"💥 DB insert failed, cleaning up storage: {db_err}", flush=True)
            delete_file_from_storage(file_url)
            raise

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
# ИСПРАВЛЕНИЕ #6, #7, #8: общие функции парсинга — единый источник правды
# вместо трёх копий одного кода в /dashboard, /indicators, /recommendations
# ========================
def parse_indicators(rows: list) -> list:
    """Парсит показатели из всех result и возвращает список с последними значениями."""
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

            name   = parts[0]
            value  = parts[1]
            status = parts[2].lower()

            if len(name) < 2 or len(name) > 80:
                continue
            if not any(c.isdigit() for c in value):
                continue

            if "выше" in status:
                norm_status = "above"
            elif "ниже" in status:
                norm_status = "below"
            else:
                norm_status = "normal"  # включает "норм" и всё остальное

            name_key = name.lower().strip()
            existing = merged.get(name_key)

            # ИСПРАВЛЕНИЕ #6: корректное сравнение дат —
            # запись с датой всегда приоритетнее записи без даты;
            # между двумя датами побеждает более поздняя.
            if existing is None:
                should_update = True
            elif row_date and not existing["date"]:
                should_update = True   # новая имеет дату, старая нет
            elif row_date and existing["date"]:
                should_update = row_date > existing["date"]
            else:
                should_update = False  # новая без даты — не обновляем

            if should_update:
                merged[name_key] = {
                    "name":   name,
                    "value":  value,
                    "status": norm_status,
                    "date":   row_date,
                    "source": source,
                }

    return sorted(merged.values(), key=lambda x: x["name"])


def parse_recommendations(rows: list) -> list:
    """Парсит рекомендации из всех result и возвращает дедублированный список."""
    REC_START_HEADERS = {"рекоменда"}
    # ИСПРАВЛЕНИЕ #7: убираем из STOP "обратить внимание" и "общее состояние" —
    # они идут ДО рекомендаций в структуре Gemini и не должны прерывать блок.
    REC_STOP_HEADERS = {"вывод", "заключение"}

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

    return recs[:20]


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

        # ИСПРАВЛЕНИЕ #8: используем общие функции вместо дублированного кода
        return jsonify({
            "history":         rows,
            "indicators":      parse_indicators(rows),
            "recommendations": parse_recommendations(rows),
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

        # ИСПРАВЛЕНИЕ #8: используем общую функцию parse_indicators
        return jsonify({"indicators": parse_indicators(rows)})

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

        # ИСПРАВЛЕНИЕ #8: используем общую функцию parse_recommendations
        return jsonify({"recommendations": parse_recommendations(rows)})

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
