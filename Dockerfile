FROM python:3.9-slim
WORKDIR /app
COPY src/gpt-bot.py ./
COPY src/roles/ ./roles/
COPY .env ./
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN chmod +x gpt-bot.py
CMD ["python3", "gpt-bot.py"]
