"""대시보드에서 새로고침 후에도 저장된 예측·드리프트 이력을 조회한다."""
from typing import Annotated, Literal

from fastapi import APIRouter, Query

from data.observations import drift_history, prediction_history
from serving_app.schemas import DriftHistoryResponse, PredictionHistoryResponse

router = APIRouter(prefix="/api/v1")
Page = Annotated[int, Query(ge=1)]
PageSize = Annotated[int, Query(ge=1, le=100)]
Mode = Literal["live", "simulation"] | None
ModelVersion = Annotated[str | None, Query(min_length=1, max_length=100)]


@router.get("/videos/predictions", response_model=PredictionHistoryResponse,
            tags=["영상 성과"], summary="영상별 예측·실제값 이력 조회",
            description="최신 예측부터 반환합니다. 필터를 생략하면 모든 모드·모델 버전을 조회하며 미등록 실제값은 null입니다.")
def list_predictions(page: Page = 1, page_size: PageSize = 20,
                     mode: Mode = None, model_version: ModelVersion = None):
    return prediction_history(page=page, page_size=page_size, mode=mode, model_version=model_version)


@router.get("/monitoring/drift", response_model=DriftHistoryResponse,
            tags=["모델 운영"], summary="완료된 드리프트 평가 묶음 조회",
            description="최신 block_id부터 반환합니다. 당시 기준값과 판정을 그대로 읽으며, 대기 중인 영상이나 재학습 결과는 포함하지 않습니다.")
def list_drift(page: Page = 1, page_size: PageSize = 20,
               mode: Mode = None, model_version: ModelVersion = None):
    return drift_history(page=page, page_size=page_size, mode=mode, model_version=model_version)
