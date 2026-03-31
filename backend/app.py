import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# 🔐 Ключи из Render
OCR_API_KEY = os.environ.get("OCR_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 🔹 Настройка Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# 🔹 OCR функция
def ocr_space(file_bytes):
    url = "https://api.ocr.space/parse/image"
    payload = {
        "apikey": OCR_API_KEY,
        "language": "rus"
    }
    files = {
        "file": ("file", file_bytes)
    }

    response = requests.post(url, files=files, data=payload)
    result = response.json()

    print("OCR RESPONSE:", result)

    # Проверка ошибок OCR
    if result.get("IsErroredOnProcessing"):
        raise Exception(f"OCR Error: {result.get('ErrorMessage')}")

    if not result.get("ParsedResults"):
        raise Exception("OCR не вернул ParsedResults")

    text = result["ParsedResults"][0].get("ParsedText", "")

    if not text.strip():
        raise Exception("OCR вернул пустой текст")

    return text


# 🔹 Gemini анализ
def analyze_with_gemini(text, age, gender):
    prompt = f"""
Ты медицинский ассистент.

Расшифруй медицинские анализы:
Пол: {gender}
Возраст: {age} лет

Текст анализов:
{text}

1. Объясни показатели простым языком
2. Укажи возможные отклонения
3. Дай рекомендации

Пиши понятно, как для обычного человека.
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("GEMINI ERROR:", e)
        raise Exception("Ошибка при обращении к Gemini")


# 🔹 Главный endpoint
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # Проверка файла
        if "file" not in request.files:
            return jsonify({"error": "Файл не найден"}), 400

        file = request.files["file"]
        age = request.form.get("age")
        gender = request.form.get("gender")

        if not age or not gender:
            return jsonify({"error": "Возраст или пол не указаны"}), 400

        print("📥 Получен файл:", file.filename)

        # 1️⃣ OCR
        text = ocr_space(file.read())
        print("📄 OCR TEXT:", text[:200])

        # 2️⃣ Gemini
        analysis = analyze_with_gemini(text, age, gender)

        # 3️⃣ Ответ
        return jsonify({"analysis": analysis})

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)}), 500


# 🔹 Запуск
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)