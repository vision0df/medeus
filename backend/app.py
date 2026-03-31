import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# Инициализация Flask
app = Flask(__name__)
CORS(app)  # Разрешаем фронтенду делать запросы к API

# Получение ключей из Render Environment Variables
OCR_KEY = os.environ.get("OCR_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

# Маршрут для распознавания файлов через OCR.Space
@app.route("/ocr", methods=["POST"])
def ocr():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        response = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": (file.filename, file.read())},
            data={"apikey": OCR_KEY, "language": "eng"}  # "rus" для русского языка
        )
        result = response.json()
        text = result["ParsedResults"][0]["ParsedText"]
    except Exception as e:
        text = ""
        print(f"OCR error: {e}")

    return jsonify({"text": text})

# Маршрут для расшифровки анализов через Gemini
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text_from_ocr = data.get("text", "")
    age = data.get("age", "")
    gender = data.get("gender", "")

    # -----------------------------
    # Готовый промпт прямо в app.py
    # -----------------------------
    prompt = f"""
Расшифруй медицинские анализы для человека:
Пол: {gender}
Возраст: {age} лет

Текст анализов:
{text_from_ocr}

Выведи разбор анализов на понятном человеческом языке и дай рекомендации по здоровью.
    """

    try:
        gemini_response = requests.post(
            "https://api.generative-ai.google.com/v1beta2/models/text-bison-001:generate",
            headers={"Authorization": f"Bearer {GEMINI_KEY}"},
            json={"prompt": prompt}
        )
        analysis = gemini_response.json().get("output_text", "Ошибка генерации")
    except Exception as e:
        analysis = "Ошибка при обращении к LLM"
        print(f"Gemini error: {e}")

    return jsonify({"analysis": analysis})

# Запуск сервера
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)