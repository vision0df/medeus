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
CORS(app)

# ========================
# Ключи
# ========================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL   = os.environ.get("SUPABASE_URL")        # https://xxxx.supabase.co
SUPABASE_KEY   = os.environ.get("SUPABASE_SERVICE_KEY") # sb_secret_...

print("GEMINI KEY:",  "OK" if GEMINI_API_KEY else "MISSING", flush=True)
print("SUPABASE URL:", "OK" if SUPABASE_URL  else "MISSING", flush=True)
print("SUPABASE KEY:", "OK" if SUPABASE_KEY  else "MISSING", flush=True)

gemini = genai.Client(api_key=GEMINI_API_KEY)

# Общие заголовки для всех запросов к Supabase REST API
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
    """
    Проверяет JWT токен через Supabase Auth REST API.
    Возвращает dict с полями id, email и др.
    """
    if not auth_header or not auth_header.startswith("Bearer "):
        raise ValueError("Требуется авторизация")

    token = auth_header.removeprefix("Bearer ").strip()

    resp = httpx.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={
            "apikey":        SUPABASE_KEY,
            "Authorization": f"Bearer {token}",
        },
        timeout=10,
    )

    if resp.status_code != 200:
        raise ValueError("Недействительный токен")

    return resp.json()   # {"id": "...", "email": "...", ...}


def db_insert(table: str, data: dict):
    """Вставляет строку в таблицу через PostgREST."""
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
    """Выбирает строки из таблицы через PostgREST."""
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


# ========================
# Gemini анализ
# ========================
def analyze_with_gemini(file, age: str, gender: str) -> str:
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

    file.seek(0)
    file_bytes = file.read()
    mime_type  = get_mime_type(file.filename)

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

        # --- Auth ---
        user = get_current_user(request.headers.get("Authorization"))
        print(f"👤 user: {user['id']}", flush=True)

        # --- Валидация входных данных ---
        if "file" not in request.files:
            return jsonify({"error": "Файл не найден"}), 400

        file   = request.files["file"]
        age    = request.form.get("age", "").strip()
        gender = request.form.get("gender", "").strip()

        if not age or not gender:
            return jsonify({"error": "Возраст или пол не указаны"}), 400

        print(f"📥 file={file.filename}  age={age}  gender={gender}", flush=True)

        # --- Gemini анализ ---
        analysis = analyze_with_gemini(file, age, gender)

        # --- Сохранение в Supabase DB ---
        db_insert("analyses", {
            "user_id":  user["id"],
            "filename": file.filename,
            "age":      age,
            "gender":   gender,
            "result":   analysis,
        })

        print("💾 Saved to DB", flush=True)

        return jsonify({"analysis": analysis})

    except ValueError as e:
        # ошибки авторизации и валидации
        return jsonify({"error": str(e)}), 401

    except Exception as e:
        print("🔥 ERROR:", flush=True)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ========================
# API: /history  — история анализов пользователя
# ========================
@app.route("/history", methods=["GET"])
def history():
    try:
        user = get_current_user(request.headers.get("Authorization"))

        rows = db_select(
            "analyses",
            select="id,filename,age,gender,result,created_at",
            filters={"user_id": user["id"]},
        )

        return jsonify({"history": rows})

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
