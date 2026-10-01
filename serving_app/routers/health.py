"""모델을 로드하지 않고 서버와 현재 버전의 설정을 확인한다."""
import json

from fastapi import APIRouter

from serving_app.config import PRODUCTION_FILE, baseline_rmsle, deployment_gate
from serving_app.monitoring.drift_detector import drift_threshold

router = APIRouter()


@router.get("/health", tags=["상태 확인"], summary="서버와 현재 모델 설정 확인")
def health():
    pointer = json.loads(PRODUCTION_FILE.read_text(encoding="utf-8")) if PRODUCTION_FILE.exists() else None
    version = pointer["version"] if pointer else None
    baseline = baseline_rmsle(version) if version is not None else None
    return {"status": "ok", "model_ready": pointer is not None, "model_version": version,
            "baseline_rmsle": baseline,
            "drift_threshold": drift_threshold(baseline) if baseline is not None else None,
            "deployment_gate": deployment_gate(version)}
