"""프로젝트 경로와 운영 설정. 실습 원본의 모델·DB와 저장 위치를 분리한다."""
import math
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
BASE_RUNTIME_DIR = Path(os.getenv("VIEWPILOT_RUNTIME_DIR", PROJECT_DIR / "runtime")).resolve()
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
# 시연 중 모델이 교체되어도 원본 모델·기준값·관측 DB는 보존한다.
RUNTIME_DIR = BASE_RUNTIME_DIR / "demo" if DEMO_MODE else BASE_RUNTIME_DIR
MODEL_DIR = RUNTIME_DIR / "models"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
LOG_DIR = RUNTIME_DIR / "logs"
OBSERVATION_DB = RUNTIME_DIR / "observations.db"
PRODUCTION_FILE = MODEL_DIR / "production.json"
BASELINE_DIR = RUNTIME_DIR / "baselines"
MODEL_NAME = "ViewPilot_Predictor"


def optional_number(name: str) -> float | None:
    text = os.getenv(name, "").strip()
    if not text:
        return None
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}은 0 이상의 유한한 숫자여야 합니다.")
    return value


def baseline_rmsle(version: str | None = None) -> float | None:
    # 모델 교체 후 이전 모델의 B를 재사용하지 않도록 버전도 함께 설정한다.
    value = optional_number("BASELINE_RMSLE")
    if version is not None and os.getenv("BASELINE_MODEL_VERSION", "").strip() != version:
        return None
    return value


def deployment_gate(version: str | None = None) -> float | None:
    # 선택 상한은 최초 모델에만 적용한다. 재학습은 동일 검증에서 개선 여부로 판단한다.
    return optional_number("DEPLOY_RMSLE_GATE") if version is None else None
