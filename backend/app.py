import os
import requests
import traceback
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from PIL import Image

app = Flask(__name__)
CORS(app)

# ========================
# Ключи
# ========================
OCR_API_KEY = os.environ.get("OCR_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

print("OCR KEY:", "OK" if OCR_API_KEY else "MISSING", flush=True)
print("GEMINI KEY:", "OK" if GEMINI_API_KEY else "MISSING", flush=True)

# ========================
# Gemini клиент
# ========================
client = genai.Client(api_key=GEMINI_API_KEY)

def get_mime_type(filename):
    ext = os.path.splitext(filename)[1].lower()
    mime_types = {
        ".pdf":  "application/pdf",
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    return mime_types.get(ext, "image/jpeg")

# ========================
# Сжатие изображения
# ========================
def compress_image(file, max_size_mb=1):
    max_bytes = max_size_mb * 1024 * 1024

    file.seek(0)
    image = Image.open(file)

    # Конвертация (важно для PNG)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    quality = 95
    width, height = image.size

    while True:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        size = buffer.tell()

        if size <= max_bytes:
            print(f"✅ Image compressed: {size/1024:.2f} KB", flush=True)
            buffer.seek(0)
            return buffer

        quality -= 5

        # если качество уже низкое — уменьшаем размер
        if quality < 30:
            width = int(width * 0.8)
            height = int(height * 0.8)
            image = image.resize((width, height))
            quality = 85

# ========================
# OCR функция
# ========================
def ocr_space(file, retries=3, timeout=120):
    url = "https://api.ocr.space/parse/image"
    payload = {"apikey": OCR_API_KEY, "language": "rus"}

    for attempt in range(1, retries + 1):
        try:
            compressed_file = compress_image(file)

            files = {
                "file": ("compressed.jpg", compressed_file)
            }

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
Ты — медицинский ассистент.
Задача:
- Проанализировать текст анализов
- Выдать результат в строго заданной структуре

ВАЖНЫЕ ПРАВИЛА: 
1. Если анализы есть — отвечай строго в формате:
Название - значение - статус("норма", "выше нормы", "ниже нормы") - комментарий( писать ТОЛЬКО если не норма (кратко))
2. после анализов напиши краткий вывод и дай рекомендации

Возраст: {age}
Пол: {gender}
"""

    try:
        file_bytes = file.read()
        mime_type = get_mime_type(file.filename)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            prompt
            ]
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