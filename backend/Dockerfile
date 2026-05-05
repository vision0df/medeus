# 1. Базовый образ Python
FROM python:3.11-slim
 
# 2. Рабочая директория внутри контейнера
WORKDIR /app
 
# 3. Копируем все файлы проекта в контейнер
COPY . /app
 
# 4. Обновляем pip и устанавливаем зависимости
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
 
# 5. Открываем порт 5000
EXPOSE 5000
 
# 6. Команда запуска сервера (gunicorn — продакшн WSGI-сервер)
# --workers 2 — 2 воркера, оптимально для большинства хостингов с 1-2 CPU
# --threads 4 — 4 треда на воркер для I/O-bound задач (httpx, Gemini API)
# --timeout 120 — 120 сек таймаут (Gemini может отвечать долго)
# --bind 0.0.0.0:5000 — слушаем на всех интерфейсах
CMD ["gunicorn", "app:app", "--workers", "2", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:5000"]