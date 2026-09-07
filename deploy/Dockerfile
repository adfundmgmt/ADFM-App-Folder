FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080 ADFM_ENV=production
WORKDIR /app
COPY requirements-native.txt ./
RUN pip install --no-cache-dir -r requirements-native.txt && useradd --create-home --uid 10001 adfm
COPY adfm_engine ./adfm_engine
COPY adfm_api ./adfm_api
USER adfm
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8080')+'/health/live',timeout=4)"
CMD ["sh", "-c", "exec uvicorn adfm_api.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --limit-concurrency 16 --timeout-keep-alive 5"]
