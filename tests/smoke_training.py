"""임시 데이터·DB에서 실제 TensorFlow/MLflow 저장과 배포 연결을 검증한다.

실제 예측 품질을 평가하는 실험이 아니다. 마지막 배포 테스트는 결과를 확실하게
재현하기 위해 후보 모델의 출력 가중치를 고정한다.
"""
import math
import os
from unittest.mock import patch

import tests  # 운영 저장소 대신 임시 런타임을 먼저 설정한다.
from tests.support import video

os.environ["DEPLOY_RMSLE_GATE"] = ""
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from serving_app import model_loader
from serving_app.config import PRODUCTION_FILE
from serving_app.train_and_register import fine_tune, train_and_register


def fixed_candidate(value):
    """게이트 분기만 검증할 수 있도록 후보의 예측값을 고정한다."""
    def fit(model, training, validation, preprocessor, epochs, patience):
        final = model.layers[-1]
        kernel, bias = final.get_weights()
        final.set_weights([np.zeros_like(kernel), np.full_like(bias, math.log1p(value))])
        return {"epochs_completed": 1, "best_epoch": 1, "best_validation_loss": 0.0}
    return fit


def main():
    # 단순 예측보다 나쁜 최초 후보는 모델 파일을 기록해도 활성화하지 않는다.
    with patch("serving_app.train_and_register.train_model", side_effect=fixed_candidate(8400)):
        blocked = train_and_register(
            training_rows=[video(i, target_views_day7=25000) for i in range(70)],
            validation_rows=[video(i, target_views_day7=25000) for i in range(80, 100)], epochs=1)
    assert not blocked["promoted"] and not blocked["deployment_checks"]["beats_naive"], blocked
    assert not PRODUCTION_FILE.exists()

    initial = train_and_register(training_rows=[video(i) for i in range(70)],
                                 validation_rows=[video(i) for i in range(80, 100)], epochs=1)
    assert initial["promoted"], initial
    assert initial["rmsle"] < initial["naive_rmsle"]
    original = model_loader.get_model()
    pointer = PRODUCTION_FILE.read_text()
    original_weights = [w.copy() for w in original.keras_model.get_weights()]

    os.environ["DEPLOY_RMSLE_GATE"] = "0"
    # 가중치를 그대로 둔 후보는 기존 모델과 동점이므로 교체하지 않는다.
    with patch("serving_app.train_and_register.train_model", return_value={
        "epochs_completed": 0, "best_epoch": 0, "best_validation_loss": 0.0
    }):
        rejected = fine_tune([video(i) for i in range(120, 190)], [video(i) for i in range(200, 220)], epochs=1)
    assert not rejected["promoted"], rejected
    assert PRODUCTION_FILE.read_text() == pointer
    assert all(np.array_equal(before, after) for before, after in zip(original_weights, original.keras_model.get_weights()))

    # 기존 모델보다 좋아도 단순 예측보다 나쁜 후보는 서빙 객체·포인터를 바꾸지 않는다.
    with patch("serving_app.train_and_register.train_model", side_effect=fixed_candidate(15000)):
        naive_rejected = fine_tune(
            [video(i, target_views_day7=25000) for i in range(120, 190)],
            [video(i, target_views_day7=25000) for i in range(200, 220)], epochs=1)
    assert naive_rejected["deployment_checks"] == {"beats_naive": False, "beats_current": True}, naive_rejected
    assert not naive_rejected["promoted"]
    assert PRODUCTION_FILE.read_text() == pointer
    assert all(np.array_equal(before, after) for before, after in zip(original_weights, original.keras_model.get_weights()))

    # 두 비교 대상을 모두 이기는 후보로 승격·캐시·MLflow 로딩 연결을 확인한다.
    os.environ["DEPLOY_RMSLE_GATE"] = "0"
    with patch("serving_app.train_and_register.train_model", side_effect=fixed_candidate(12000)):
        promoted = fine_tune([video(i) for i in range(120, 190)], [video(i) for i in range(200, 220)], epochs=1)
    assert promoted["promoted"], promoted
    assert promoted["rmsle"] < min(promoted["current_rmsle"], promoted["naive_rmsle"])
    assert model_loader.get_model().version != original.version
    assert model_loader.get_model().predict_one(video(200)) == 12000

    with patch.dict(os.environ, {"MODEL_SOURCE": "mlflow"}), patch.object(model_loader, "_cache_pointer", None):
        restored = model_loader.get_model()
        assert restored.predict_one(video(200)) == 12000
    print("PASS: naive gate blocks initial and improved-but-worse-than-naive candidates; real initial fit, tie retention, untouched serving weights, promotion, local and MLflow reload")


if __name__ == "__main__":
    main()
