import os
import requests
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# ========================
# Ключи из Render
# ========================
OCR_API_KEY = os.environ.get("OCR_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

print("OCR KEY:", "OK" if OCR_API_KEY else "MISSING")
print("GEMINI KEY:", "OK" if GEMINI_API_KEY else "MISSING")

# ========================
# Настройка Gemini
# ========================
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5")  # исправленная модель

# ========================
# Функция OCR с таймаутом и повторными попытками
# ========================
def ocr_space(file, retries=3, timeout=120):
    url = "https://api.ocr.space/parse/image"
    payload = {"apikey": OCR_API_KEY, "language": "rus"}

    for attempt in range(1, retries + 1):
        try:
            # Сбрасываем курсор файла на начало перед чтением
            file.seek(0)
            files = {"file": (file.filename, file.read())}

            response = requests.post(url, data=payload, files=files, timeout=timeout)
            result = response.json()
            print(f"OCR RESPONSE (attempt {attempt}):", result)

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
            print(f"⚠️ OCR тайм-аут на попытке {attempt}")
            if attempt == retries:
                raise Exception("OCR Error: Тайм-аут сервиса после нескольких попыток")
        except Exception as e:
            # Любая другая ошибка — сразу выходим
            raise e

# ========================
# Функция анализа через Gemini
# ========================
def analyze_with_gemini(text, age, gender):
    prompt = f"""
Ты медицинский ассистент.

Пациент:
Возраст: {age}
Пол: {gender}

Анализы:
{text}

Задача:
1. Объясни простым языком
2. Укажи отклонения
3. Дай рекомендации
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("GEMINI ERROR:", e)
        # Если модель не доступна, возвращаем заглушку
        return "Анализ временно недоступен. OCR текст обработан, но Gemini не ответил."

# ========================
# API endpoint
# ========================
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Файл не найден"}), 400

        file = request.files["file"]
        age = request.form.get("age")
        gender = request.form.get("gender")

        print("📥 FILE:", file.filename)
        print("📥 AGE:", age)
        print("📥 GENDER:", gender)

        if not age or not gender:
            return jsonify({"error": "Возраст или пол не указаны"}), 400

        # OCR
        text = ocr_space(file)
        print("📄 OCR TEXT (первые 200 символов):", text[:200])

        # Gemini
        analysis = analyze_with_gemini(text, age, gender)

        return jsonify({"analysis": analysis})

    except Exception as e:
        print("🔥 ERROR:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ========================
# Запуск
# ========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)