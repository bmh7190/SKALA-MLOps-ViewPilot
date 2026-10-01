"""RMSLE와 30개 영상 단위의 성능 저하 판정."""
import math

WINDOW_SIZE = 30


def compute_rmsle(actual: list[float], predicted: list[float]) -> float:
    if not actual or len(actual) != len(predicted):
        raise ValueError("실제값과 예측값은 같은 길이의 비어 있지 않은 목록이어야 합니다.")
    if any(not math.isfinite(v) or v < 0 for v in [*actual, *predicted]):
        raise ValueError("RMSLE는 0 이상의 유한한 값으로 계산합니다.")
    errors = [(math.log1p(a) - math.log1p(p)) ** 2 for a, p in zip(actual, predicted)]
    return math.sqrt(sum(errors) / len(errors))


def drift_threshold(baseline: float) -> float:
    if not math.isfinite(baseline) or baseline < 0:
        raise ValueError("기준 RMSLE는 0 이상의 유한한 값이어야 합니다.")
    return max(baseline * 1.3, baseline + 0.1)


def evaluate_block(actual, predicted, baseline: float, previous_streak: int) -> dict:
    if len(actual) != WINDOW_SIZE:
        raise ValueError("평가 묶음은 정확히 30개 영상이어야 합니다.")
    score = compute_rmsle(actual, predicted)
    threshold = drift_threshold(baseline)
    streak = previous_streak + 1 if score > threshold else 0
    if streak >= 2:
        status = "retrain_review"
    elif streak == 1:
        status = "warning"
    else:
        status = "ok"
    return {"rmsle": score, "threshold": threshold, "consecutive_exceeds": streak, "status": status}
