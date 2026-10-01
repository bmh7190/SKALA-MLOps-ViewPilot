"""수동으로 계산한 베이스라인의 모델 버전과 영상 ID를 보관한다."""
import json
import math

from data.observations import recorded_video_ids
from serving_app.config import BASELINE_DIR
from serving_app.monitoring.drift_detector import compute_rmsle, drift_threshold


def read_baseline(version: str) -> dict | None:
    path = BASELINE_DIR / f"{version}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def baseline_video_ids() -> set[str]:
    ids = set()
    for path in BASELINE_DIR.glob("*.json"):
        ids.update(json.loads(path.read_text(encoding="utf-8"))["video_ids"])
    return ids


def calculate_baseline(model, rows: list[dict]) -> dict:
    ids = {row["video_id"] for row in rows}
    if len(rows) != 30 or len(ids) != 30:
        raise ValueError("베이스라인은 서로 다른 영상 정확히 30개로 계산하세요.")
    if ids & (set(model.metadata["used_video_ids"]) | recorded_video_ids()):
        raise ValueError("학습·검증 또는 운영 평가에 사용한 영상은 베이스라인에 사용할 수 없습니다.")
    predictions = model.predict_many(rows)
    actual = [row["target_views_day7"] for row in rows]
    score = compute_rmsle(actual, predictions)
    return {"model_version": model.version, "run_id": model.metadata["run_id"],
            "baseline_rmsle": score, "threshold": drift_threshold(score),
            "mae": sum(abs(a - p) for a, p in zip(actual, predictions)) / len(rows),
            "video_ids": sorted(ids)}


def save_baseline(result: dict):
    existing = read_baseline(result["model_version"])
    if existing is not None and existing != result:
        raise ValueError("이 모델의 기준값은 이미 저장되어 있습니다. 다른 기준으로 덮어쓸 수 없습니다.")
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINE_DIR / f"{result['model_version']}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def check_configured_baseline(version: str, value: float):
    # 수동 복사 시 다른 버전의 결과나 오타를 넣었는지 확인한다.
    saved = read_baseline(version)
    if saved is None:
        raise ValueError("먼저 calculate_baseline.py로 해당 모델의 기준값을 계산하세요.")
    if not math.isclose(value, saved["baseline_rmsle"], rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("BASELINE_RMSLE가 저장된 계산 결과와 다릅니다. 출력된 값을 그대로 입력하세요.")
