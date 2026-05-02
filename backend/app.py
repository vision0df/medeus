import os
import traceback
import httpx
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, origins=["https://medeus.vercel.app"])

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

SUPA_HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "return=representation",
}

# ========================
# Helpers
# ========================
def get_mime_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return {
        ".pdf":  "application/pdf",
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(ext, "image/jpeg")


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

    prompt = f"""
Ты — медицинский ассистент.
Задача — выдать результат в строго заданной структуре.

ПРАВИЛА:
1. Если анализы найдены — отвечай строго в формате:
   Название - значение - статус ("норма", "выше нормы", "ниже нормы")
2. После таблицы анализов кратко напиши:
   — Общее состояние
   — На что стоит обратить внимание
   — Рекомендации

Возраст: {age}
Пол: {gender}
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
    return response.text


# ========================
# API: /analyze
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

        print(f"📥 file={file.filename}  age={age}  gender={gender}  name={analysis_name}  date={analysis_date}", flush=True)

        file.seek(0)
        file_bytes = file.read()
        mime_type  = get_mime_type(file.filename)

        # Загружаем файл в Storage
        file_url = upload_file_to_storage(user["id"], file.filename, file_bytes, mime_type)
        print(f"📦 File uploaded: {file_url}", flush=True)

        # Gemini анализ
        analysis = analyze_with_gemini(file_bytes, file.filename, age, gender)

        # Сохраняем в БД
        row = {
            "user_id":       user["id"],
            "filename":      file.filename,
            "analysis_name": analysis_name,
            "age":           age,
            "gender":        gender,
            "result":        analysis,
            "file_url":      file_url,
        }
        if analysis_date:
            row["analysis_date"] = analysis_date

        db_insert("analyses", row)
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
# Запуск
# ========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
