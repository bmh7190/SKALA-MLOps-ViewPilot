"""ViewPilot API 진입점. 정적 화면은 실습 원본을 유지한다."""
import logging
import os
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from serving_app import model_loader
from serving_app.config import DEMO_MODE, LOG_DIR, PROJECT_DIR, baseline_rmsle, optional_number
from serving_app.routers import data, health, logs, predict, simulations, training

logger = logging.getLogger("aiops")


def configure_logging():
    # 시연 폴더 복원 후 파일을 연다. 이전 실행의 열린 핸들은 먼저 닫는다.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_DIR / "aiops.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())


request_logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if DEMO_MODE:
        from serving_app.demo import reset_demo_runtime
        reset_demo_runtime()
        model_loader.clear_cache()
    configure_logging()
    if DEMO_MODE:
        logger.info("demo reset: 원본 모델과 기준값을 복원했습니다. 정상·드리프트 시나리오를 실행할 수 있습니다.")
    # 잘못된 숫자 설정은 시작 시 알린다. 미설정은 허용해 CSV 업로드부터 진행할 수 있다.
    baseline_rmsle()
    optional_number("DEPLOY_RMSLE_GATE")
    if os.getenv("LOADING_MODE", "lazy") == "eager":
        model_loader.load_eager()
    yield


app = FastAPI(title="ViewPilot · 조회수 예측과 모델 운영", lifespan=lifespan)


@app.middleware("http")
async def log_response_time(request: Request, call_next):
    # 요청 수신부터 응답 준비까지의 시간을 성공·실패 모두 콘솔에 남긴다.
    started_at = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        request_logger.info("%s %s status=%d duration=%.2fms", request.method,
                            request.url.path, status_code, (perf_counter() - started_at) * 1000)


@app.exception_handler(ValueError)
async def invalid_data(request: Request, error: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(error)})


@app.exception_handler(FileNotFoundError)
async def missing_model_or_data(request: Request, error: FileNotFoundError):
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.exception_handler(LookupError)
async def missing_prediction(request: Request, error: LookupError):
    return JSONResponse(status_code=404, content={"detail": str(error)})


for router in (predict.router, data.router, training.router, simulations.router, health.router, logs.router):
    app.include_router(router)

# API보다 뒤에 등록해야 /api/v1 경로가 정적 파일로 처리되지 않는다.
app.mount("/", StaticFiles(directory=PROJECT_DIR / "serving_app" / "static", html=True), name="static")
