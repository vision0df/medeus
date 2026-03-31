# backend/app.py
from flask import Flask, request, jsonify
import requests
from PIL import Image
import io
import os

app = Flask(__name__)
OCR_API_KEY = os.getenv("OCR_API_KEY")
OCR_URL = "https://api.ocr.space/parse/image"

MAX_SIZE_KB = 1000  # лимит OCR.Space

def compress_image(file, max_size_kb=MAX_SIZE_KB):
    img = Image.open(file)
    buf = io.BytesIO()
    quality = 90
    img.save(buf, format="JPEG", optimize=True, quality=quality)
    
    # уменьшаем качество пока файл > лимит
    while buf.tell() > max_size_kb * 1024 and quality > 10:
        buf.seek(0)
        buf.truncate(0)
        quality -= 10
        img.save(buf, format="JPEG", optimize=True, quality=quality)
    
    buf.seek(0)
    return buf

@app.route("/ocr", methods=["POST"])
def ocr():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    
    # Сжимаем изображение
    compressed_file = compress_image(file.stream)
    
    files = {"file": (file.filename, compressed_file, "image/jpeg")}
    payload = {"apikey": OCR_API_KEY, "language": "eng"}
    
    response = requests.post(OCR_URL, data=payload, files=files)
    result = response.json()
    
    try:
        text = result["ParsedResults"][0]["ParsedText"]
    except:
        text = ""
    
    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)