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

# 6. Команда запуска сервера
CMD ["python", "app.py"]