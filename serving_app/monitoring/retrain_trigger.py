"""평가 결과를 알리고, 데이터가 준비됐을 때 후보 모델을 재학습한다."""
import logging
import os
from pathlib import Path

from data import observations
from data.features import load_rows
from data.storage import latest_upload
from serving_app import model_loader
from serving_app.config import PROJECT_DIR, baseline_rmsle
from data.baselines import check_configured_baseline

logger = logging.getLogger("aiops")


def retraining_data():
    """일반 재학습 API에서 지정한 학습·검증 파일을 읽는다."""
    paths = []
    for setting, purpose in [("RETRAIN_TRAIN_CSV", "retrain_train"),
                             ("RETRAIN_VALIDATION_CSV", "retrain_validation")]:
        value = os.getenv(setting, "").strip()
        try:
            path = Path(value) if value else latest_upload(purpose)
        except FileNotFoundError as error:
            raise FileNotFoundError("재학습용 CSV와 검증용 CSV를 지정하세요. 시나리오는 배치 요청에 검증 영상을 따로 전달합니다.") from error
        paths.append(path if path.is_absolute() else PROJECT_DIR / path)
    return load_rows(paths[0]), load_rows(paths[1])


def check_and_trigger(version: str, mode: str, validation_rows=None) -> dict:
    baseline = baseline_rmsle(version)
    if baseline is not None:
        check_configured_baseline(version, baseline)
    check = observations.evaluate_pending(version, mode, baseline)
    for block in check["new_blocks"]:
        logger.log(logging.INFO if block["status"] == "ok" else logging.WARNING,
                   "[%s] mode=%s model=v%s RMSLE=%.4f threshold=%.4f streak=%s videos=%s",
                   block["status"], mode, version, block["rmsle"], block["threshold"],
                   block["consecutive_exceeds"], ",".join(block["video_ids"]))
    if check["status"] != "retrain_review" or not check["new_blocks"]:
        return check

    check["message"] = "기존 예측과 다른 성과가 반복되고 있으니, 관련 영상과 지표의 변화를 검토하세요."
    if os.getenv("AUTO_RETRAIN", "true").lower() != "true":
        check["retraining"] = {"status": "manual_review"}
        return check
    try:
        # 이전 버전의 늦게 도착한 실제값이 현재 모델의 재학습을 유발하지 않도록 한다.
        if model_loader.get_model().version != version:
            check["retraining"] = {"status": "skipped_old_version"}
            return check
        from serving_app.train_and_register import fine_tune

        logger.info("retrain started: mode=%s model=v%s", mode, version)
        if mode == "simulation" and validation_rows is not None:
            # 앞 60개는 감지 후 학습에 사용하고, 요청에 따로 전달한 뒤 60개는 검증에만 사용한다.
            training = observations.recent_drift_rows(version, mode)
            result = fine_tune(training, validation_rows, expected_version=version, retrospective=True)
        else:
            training, validation = retraining_data()
            result = fine_tune(training, validation, expected_version=version)
        check["retraining"] = {"status": "completed", **result,
                               "baseline_required": result["promoted"]}
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        logger.warning("retrain deferred: %s", error)
        check["retraining"] = {"status": "deferred", "reason": str(error)}
    except Exception:
        # 관측 기록과 경보는 유지한다. 학습 장애를 예측 성공으로 숨기지 않는다.
        logger.exception("retrain failed")
        check["retraining"] = {"status": "failed", "reason": "학습 중 오류가 발생했습니다. 서버 로그를 확인하세요."}
    return check
