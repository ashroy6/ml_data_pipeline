FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p input output archive failed logs models data/raw artifacts vectorstore rag/knowledge_base

CMD ["python", "src/predict.py"]
