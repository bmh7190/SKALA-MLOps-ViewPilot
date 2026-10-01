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

router = APIRouter()


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


@router.post("/predict", response_model=PredictResponse)
def predict(request: VideoInput):
    return predict_rows([request.features_dict()])[0]


@router.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    _, rows = await read_csv_file(file, require_target=False)
    predictions = await run_in_threadpool(predict_rows, rows)
    return {"predictions": predictions}


@router.post("/observations")
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


@router.post("/predict/batch-test")
def batch_test(request: BatchTestRequest):
    validation = ([v.model_dump(mode="json") for v in request.retraining_validation]
                  if request.retraining_validation is not None else None)
    return evaluate_videos([v.model_dump(mode="json") for v in request.videos], validation)


@router.post("/predict/batch-test/csv")
async def batch_test_csv(file: UploadFile = File(...)):
    _, rows = await read_csv_file(file)
    return await run_in_threadpool(evaluate_videos, rows)
