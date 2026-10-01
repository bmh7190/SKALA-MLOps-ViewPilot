FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serving_app/ serving_app/
COPY data/ data/
COPY scripts/ scripts/
# 데이터와 모델은 실행 후 준비한다. 이미지 빌드에서 주가 모델을 만들지 않는다.
ENV MODEL_SOURCE=local LOADING_MODE=lazy
EXPOSE 8000
CMD ["uvicorn", "serving_app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
