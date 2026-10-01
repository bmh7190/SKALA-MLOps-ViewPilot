"""신규 영상 예측, 실제값 등록, 과거 CSV를 이용한 별도 시뮬레이션."""
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, File, UploadFile
from starlette.concurrency import run_in_threadpool

from data import observations
from data.baselines import baseline_video_ids
from serving_app import model_loader
from serving_app.monitoring.retrain_trigger import check_and_trigger
from serving_app.routers.data import read_csv_file
from serving_app.schemas import BatchTestRequest, ObservationRequest, PredictResponse, VideoInput

router = APIRouter(prefix="/api/v1")


def predict_rows(rows: list[dict], mode: str = "live") -> list[dict]:
    model = model_loader.get_model()
    if mode == "live":
        now = datetime.now(timezone.utc)
        for row in rows:
            video = VideoInput.model_validate({key: row[key] for key in VideoInput.model_fields})
            if video.published_at + timedelta(hours=72) > now:
                raise ValueError(f"{video.video_id}: 게시 후 72시간이 지난 영상만 예측할 수 있습니다.")
    # 학습·검증에 사용한 영상으로 운영 오차를 측정하면 성능이 과대평가된다.
    used_ids = set(model.metadata["used_video_ids"]) | baseline_video_ids()
    if any(row["video_id"] in used_ids for row in rows):
        raise ValueError("학습·검증·베이스라인에 사용한 영상은 운영 예측·드리프트 평가에서 제외하세요.")
    predictions = model.predict_many(rows)
    return observations.save_predictions(rows, predictions, model.version, mode)


@router.post("/videos/predictions", response_model=PredictResponse, tags=["영상 성과"], summary="영상의 7일 누적 조회수 예측")
def predict(request: VideoInput):
    return predict_rows([request.features_dict()])[0]


@router.post("/videos/predictions/csv", tags=["영상 성과"], summary="CSV의 여러 영상 조회수 예측")
async def predict_csv(file: UploadFile = File(...)):
    _, rows = await read_csv_file(file, require_target=False)
    predictions = await run_in_threadpool(predict_rows, rows)
    return {"predictions": predictions}


@router.post("/videos/actuals", tags=["영상 성과"], summary="실제 7일 조회수 등록 및 드리프트 확인")
def observe(request: ObservationRequest):
    record = observations.save_actual(request.prediction_id, request.target_views_day7)
    check = check_and_trigger(record["model_version"], "live")
    return {"prediction_id": request.prediction_id, "drift_check": check}


def evaluate_videos(rows: list[dict], validation_rows=None) -> dict:
    # 정답을 알고 있는 과거 CSV 테스트는 운영 예측·평가 기록과 분리해 보관한다.
    predictions = predict_rows(rows, mode="simulation")
    version = predictions[0]["model_version"]
    check = check_and_trigger(version, "simulation", validation_rows=validation_rows)
    return {"predictions": predictions, "drift_check": check}


@router.post("/simulations/drift", tags=["시뮬레이션"], summary="과거 영상으로 드리프트·재학습 시나리오 실행",
             description="videos는 드리프트 감지에 사용합니다. 재학습 비교용 영상은 retraining_validation에 따로 전달합니다.")
def batch_test(request: BatchTestRequest):
    validation = ([v.model_dump(mode="json") for v in request.retraining_validation]
                  if request.retraining_validation is not None else None)
    return evaluate_videos([v.model_dump(mode="json") for v in request.videos], validation)


@router.post("/simulations/drift/csv", tags=["시뮬레이션"], summary="정답 포함 CSV로 드리프트 평가")
async def batch_test_csv(file: UploadFile = File(...)):
    _, rows = await read_csv_file(file)
    return await run_in_threadpool(evaluate_videos, rows)
