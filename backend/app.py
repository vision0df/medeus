from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

OCR_API_KEY = os.getenv("OCR_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Функция для отправки файла на OCR.Space
def ocr_space(file_bytes):
    url = "https://api.ocr.space/parse/image"
    payload = {"apikey": OCR_API_KEY, "language": "rus"}
    files = {"file": ("file", file_bytes)}
    response = requests.post(url, files=files, data=payload)
    result = response.json()
    # Получаем распознанный текст
    text = result["ParsedResults"][0]["ParsedText"]
    return text

# Функция для отправки текста в Gemini
def analyze_with_gemini(text, age, gender):
    prompt = f"""
    Ты медицинский ассистент. Дай расшифровку результатов анализов для человека:
    Возраст: {age} лет
    Пол: {gender}
    Результаты анализов:
    {text}

    Дай объяснение простым языком и рекомендации.
    """
    url = "https://api.generativeai.google/v1beta2/models/text-bison-001:generateText"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}"
    }
    payload = {"prompt": prompt, "temperature": 0.2, "maxOutputTokens": 1000}
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    analysis = result.get("candidates", [{}])[0].get("content", "")
    return analysis

# Один эндпоинт для фронтенда
@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "Нет файла"}), 400

    file = request.files["file"]
    age = request.form.get("age")
    gender = request.form.get("gender")

    if not age or not gender:
        return jsonify({"error": "Не указан возраст или пол"}), 400

    try:
        # 1️⃣ Отправляем файл на OCR
        text = ocr_space(file.read())

        # 2️⃣ Отправляем текст + возраст + пол в Gemini
        analysis = analyze_with_gemini(text, age, gender)

        # 3️⃣ Возвращаем готовый результат фронтенду
        return jsonify({"analysis": analysis})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)