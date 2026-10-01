"""대시보드의 정상·드리프트 버튼이 각각 호출하는 시연 API."""
from threading import Lock

from fastapi import APIRouter, HTTPException

from data import observations
from data.baselines import check_configured_baseline
from data.features import load_rows, validate_splits
from serving_app import model_loader
from serving_app.config import DEMO_MODE, PROJECT_DIR, baseline_rmsle
from serving_app.monitoring.drift_detector import drift_threshold
from serving_app.routers.predict import evaluate_videos

router = APIRouter(prefix="/api/v1/simulations", tags=["시뮬레이션"])
_scenario_lock = Lock()


def run_scenario(name: str):
    if not DEMO_MODE:
        raise HTTPException(409, "버튼 시연은 DEMO_MODE=true로 서버를 시작한 뒤 실행하세요.")
    if not _scenario_lock.acquire(blocking=False):
        raise HTTPException(409, "다른 시나리오가 실행 중입니다. 완료될 때까지 기다리세요.")
    try:
        model = model_loader.get_model()
        baseline = baseline_rmsle(model.version)
        if baseline is None:
            raise HTTPException(409, "모델이 교체되었습니다. 서버를 재시작하면 초기 시연 상태로 돌아갑니다.")
        check_configured_baseline(model.version, baseline)
        rows = sorted(load_rows(PROJECT_DIR / "data" / "synthetic" / f"test_{name}.csv"),
                      key=lambda row: row["published_at"])
        validation = None
        if name == "drift":
            if len(rows) != 120:
                raise ValueError("드리프트 시나리오는 영상 120개가 필요합니다.")
            rows, validation = validate_splits(rows[:60], rows[60:], retrospective=True)
        recorded = observations.recorded_video_ids()
        if any(row["video_id"] in recorded for row in rows):
            raise HTTPException(409, "이미 실행한 시나리오입니다. 서버를 재시작한 뒤 다시 실행하세요.")
        report = {"scenario": name, "model_version": model.version,
                  "baseline_rmsle": baseline, "threshold": drift_threshold(baseline), "events": []}
        for start in range(0, len(rows), 30):
            # 마지막 드리프트 묶음에만 검증 60개를 전달해 한 번 재학습한다.
            held_out = validation if start == 30 else None
            check = evaluate_videos(rows[start:start + 30], held_out)["drift_check"]
            report["events"].append({"window": start // 30 + 1, **check})
            if check.get("retraining", {}).get("promoted"):
                break
        report["active_model_version"] = model_loader.get_model().version
        report["baseline_required"] = baseline_rmsle(report["active_model_version"]) is None
        return report
    finally:
        _scenario_lock.release()


@router.post("/normal/run", summary="정상 시나리오 실행",
             description="시연 모드에서 정상 영상 120개를 30개씩 평가합니다. 같은 시나리오의 재실행은 서버 재시작 후 가능합니다.")
def run_normal():
    return run_scenario("normal")


@router.post("/drift/run", summary="드리프트·재학습 시나리오 실행",
             description="시연 모드에서 앞 60개로 감지·재학습하고 뒤 60개로 비교합니다. 학습 완료까지 응답을 기다려야 합니다.")
def run_drift():
    return run_scenario("drift")
