import os
import requests
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai

app = Flask(__name__)
CORS(app)

# ========================
# Ключи из Render
# ========================
OCR_API_KEY = os.environ.get("OCR_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

print("OCR KEY:", "OK" if OCR_API_KEY else "MISSING", flush=True)
print("GEMINI KEY:", "OK" if GEMINI_API_KEY else "MISSING", flush=True)

# ========================
# Новый клиент Gemini
# ========================
client = genai.Client(api_key=GEMINI_API_KEY)

# ========================
# OCR функция
# ========================
def ocr_space(file, retries=3, timeout=120):
    url = "https://api.ocr.space/parse/image"
    payload = {"apikey": OCR_API_KEY, "language": "rus"}

    for attempt in range(1, retries + 1):
        try:
            file.seek(0)
            files = {"file": (file.filename, file.read())}

            response = requests.post(url, data=payload, files=files, timeout=timeout)
            result = response.json()

            print(f"OCR RESPONSE (attempt {attempt}):", result, flush=True)

            if result.get("IsErroredOnProcessing"):
                raise Exception(f"OCR Error: {result.get('ErrorMessage')}")

            parsed_results = result.get("ParsedResults")
            if not parsed_results:
                raise Exception("OCR не вернул ParsedResults")

            text = parsed_results[0].get("ParsedText", "")
            if not text.strip():
                raise Exception("OCR вернул пустой текст")

            return text

        except requests.exceptions.Timeout:
            print(f"⚠️ OCR тайм-аут (попытка {attempt})", flush=True)
            if attempt == retries:
                raise Exception("OCR Error: Тайм-аут после нескольких попыток")

        except Exception as e:
            raise e

# ========================
# Gemini анализ
# ========================
def analyze_with_gemini(text, age, gender):
    print("🧠 Gemini START", flush=True)

    prompt = f"""
Ты медицинский ассистент.

Пациент:
Возраст: {age}
Пол: {gender}
Анализы:
{text}

Задача:
1. дай расшифровку анализов в таком виде:
показатель - значение (норма/ выше нормы/ ниже нормы) (если выше или ниже нормы тогда кратко укажи почему это может быть)

2. дай краткий вывод по анализам и дай рекомендации
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        print("✅ Gemini RESPONSE:", response, flush=True)

        return response.text

    except Exception as e:
        print("🔥 GEMINI ERROR:", str(e), flush=True)
        traceback.print_exc()

        return f"Ошибка Gemini: {str(e)}"

# ========================
# API endpoint
# ========================
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        print("🔥 /analyze HIT", flush=True)

        if "file" not in request.files:
            return jsonify({"error": "Файл не найден"}), 400

        file = request.files["file"]
        age = request.form.get("age")
        gender = request.form.get("gender")

        print("📥 FILE:", file.filename, flush=True)
        print("📥 AGE:", age, flush=True)
        print("📥 GENDER:", gender, flush=True)

        if not age or not gender:
            return jsonify({"error": "Возраст или пол не указаны"}), 400

        # OCR
        text = ocr_space(file)
        print("📄 OCR TEXT:", text[:200], flush=True)

        # Gemini
        analysis = analyze_with_gemini(text, age, gender)

        return jsonify({"analysis": analysis})

    except Exception as e:
        print("🔥 ERROR:", flush=True)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ========================
# Запуск
# ========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)