FROM node:20-alpine AS frontend
WORKDIR /web
COPY web/package*.json ./
RUN npm install
COPY web/ ./
RUN npm run build
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api/app ./app
COPY --from=frontend /web/dist ./app/static
EXPOSE 8000
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
