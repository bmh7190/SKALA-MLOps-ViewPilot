"""단순 예측과 현재 모델을 같은 검증 영상에서 비교하는 배포 게이트."""
from serving_app.monitoring.drift_detector import compute_rmsle

NAIVE_METHOD = "first_3_days_daily_mean_x7"


def naive_predictions(rows: list[dict]) -> list[float]:
    # 첫 3일의 일평균 조회수가 7일까지 유지된다고 가정한다. 정답은 사용하지 않는다.
    return [sum(row[f"day{day}_views"] for day in range(1, 4)) / 3 * 7 for row in rows]


def evaluate_deployment(validation, candidate_predictions, current_predictions=None, absolute_gate=None):
    actual = [row["target_views_day7"] for row in validation]
    score = compute_rmsle(actual, candidate_predictions)
    naive_score = compute_rmsle(actual, naive_predictions(validation))
    current_score = compute_rmsle(actual, current_predictions) if current_predictions is not None else None

    # 모든 후보는 단순 예측보다 좋아야 한다. 동점도 통과시키지 않는다.
    checks = {"beats_naive": score < naive_score}
    if current_score is not None:
        checks["beats_current"] = score < current_score
    elif absolute_gate is not None:
        # 선택적 절대 상한은 기존 정책대로 최초 배포에만 추가 적용한다.
        checks["within_absolute_gate"] = score <= absolute_gate

    return {
        "promoted": all(checks.values()),
        "rmsle": score,
        "naive_rmsle": naive_score,
        "current_rmsle": current_score,
        "deployment_checks": checks,
        "deployment_rule": "beats_naive_and_current" if current_score is not None else "beats_naive_initial_selection",
        "naive_method": NAIVE_METHOD,
    }
