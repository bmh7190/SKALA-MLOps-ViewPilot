"""분리된 train/validation으로 학습하고, 검증된 모델만 활성 버전으로 선택한다."""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.baselines import baseline_video_ids
from data.features import VideoPreprocessor, build_sequences, load_rows, validate_splits
from data.observations import recorded_video_ids
from data.storage import latest_upload
from serving_app import model_loader
from serving_app.config import MODEL_DIR, MODEL_NAME, PRODUCTION_FILE, deployment_gate
from serving_app.monitoring.drift_detector import compute_rmsle

logger = logging.getLogger("aiops")
_training_lock = Lock()


def make_dataset(rows, preprocessor, shuffle=False):
    import numpy as np
    import tensorflow as tf

    X, y = build_sequences(rows, preprocessor)
    dataset = tf.data.Dataset.from_tensor_slices((np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32")))
    if shuffle:
        dataset = dataset.shuffle(len(rows), seed=42)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    return dataset.batch(32).with_options(options)


def train_model(model, training, validation, preprocessor, epochs, patience):
    import numpy as np
    from tensorflow import keras

    # validation은 가중치 갱신에 사용하지 않고 조기 종료와 최적 가중치 선택에만 사용한다.
    stopper = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = model.fit(
        make_dataset(training, preprocessor, shuffle=True),
        validation_data=make_dataset(validation, preprocessor),
        epochs=epochs, callbacks=[stopper], verbose=0, shuffle=False,
    )
    return {"epochs_completed": len(history.history["loss"]),
            "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
            "best_validation_loss": float(min(history.history["val_loss"]))}


def prepare_data(training_rows, validation_rows, current=None):
    training, validation = validate_splits(training_rows, validation_rows)
    ids = {row["video_id"] for row in training + validation}
    if ids & (baseline_video_ids() | recorded_video_ids()):
        raise ValueError("기준값 계산 또는 운영 테스트에 사용한 영상은 학습·검증에 재사용할 수 없습니다.")
    if current is not None:
        if ids & set(current.metadata["used_video_ids"]):
            raise ValueError("재학습에는 기존 모델의 학습·검증에 쓰지 않은 새 영상을 사용하세요.")
        last_publication = datetime.fromisoformat(current.metadata["last_published_at"])
        if any(datetime.fromisoformat(row["published_at"]) <= last_publication for row in training + validation):
            raise ValueError("재학습 파일은 기존 모델의 학습·검증 범위보다 이후 영상이어야 합니다.")
    return training, validation


def promote_model(bundle_id, model_uri, sample_row):
    import mlflow
    from mlflow.tracking import MlflowClient

    registered = mlflow.register_model(model_uri, MODEL_NAME)
    version = str(registered.version)
    # 모델과 전처리 파일을 실제로 다시 읽은 다음 활성 포인터를 바꾼다.
    model_loader.load_bundle(bundle_id, version).predict_one(sample_row)
    MlflowClient().set_registered_model_alias(MODEL_NAME, "production", version)
    temporary = PRODUCTION_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps({"bundle_id": bundle_id, "version": version}), encoding="utf-8")
    os.replace(temporary, PRODUCTION_FILE)
    model_loader.get_model()
    return version


def train_and_register(train_path=None, validation_path=None, *, training_rows=None,
                       validation_rows=None, mode="initial", epochs=100, patience=10,
                       expected_version=None):
    if mode not in {"initial", "fine_tune"} or not 1 <= epochs <= 300 or not 1 <= patience <= 100:
        raise ValueError("mode, epochs 또는 patience 설정을 확인하세요.")
    if not _training_lock.acquire(blocking=False):
        raise ValueError("다른 학습이 진행 중입니다. 완료 후 다시 시도하세요.")
    try:
        return _train(train_path, validation_path, training_rows, validation_rows,
                      mode, epochs, patience, expected_version)
    finally:
        _training_lock.release()


def _train(train_path, validation_path, training_rows, validation_rows, mode, epochs, patience, expected_version):
    current = model_loader.get_model() if PRODUCTION_FILE.exists() else None
    if expected_version is not None and (current is None or current.version != expected_version):
        raise ValueError("학습 준비 중 모델 버전이 변경되었습니다. 새 버전의 평가를 기다리세요.")
    if mode == "initial" and current is not None:
        raise ValueError("이미 최초 모델이 선택되어 있습니다. 추가 학습은 fine_tune을 사용하세요.")
    if mode == "fine_tune" and current is None:
        raise ValueError("재학습 전에 최초 모델을 선택해야 합니다.")
    gate = deployment_gate(current.version if current else None)
    if current is not None and gate is None:
        raise ValueError("재학습 배포 상한 또는 현재 모델의 베이스라인을 먼저 설정하세요.")

    train_purpose = "retrain_train" if current else "train"
    validation_purpose = "retrain_validation" if current else "validation"
    if training_rows is None:
        training_rows = load_rows(train_path or latest_upload(train_purpose))
    if validation_rows is None:
        validation_rows = load_rows(validation_path or latest_upload(validation_purpose))
    training, validation = prepare_data(training_rows, validation_rows, current)

    import mlflow
    import mlflow.tensorflow
    import numpy as np
    from tensorflow import keras
    from serving_app.lstm_model import build_model

    keras.utils.set_random_seed(42)
    preprocessor = current.preprocessor if current else VideoPreprocessor.fit(training)
    if current:
        # 서빙 객체를 수정하지 않는다. 복제한 후보에만 fit한다.
        model = keras.models.clone_model(current.keras_model)
        model.set_weights(current.keras_model.get_weights())
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="mse")
    else:
        model = build_model(preprocessor.n_features)

    model_loader.configure_mlflow()
    with mlflow.start_run(run_name=mode) as run:
        selection = train_model(model, training, validation, preprocessor, epochs, patience)
        candidate = model_loader.LoadedModel(model, preprocessor, {}, "candidate")
        actual = [row["target_views_day7"] for row in validation]
        predictions = candidate.predict_many(validation)
        score = compute_rmsle(actual, predictions)
        current_score = compute_rmsle(actual, current.predict_many(validation)) if current else None
        # 최초 선택은 B 계산 전이다. 별도 상한이 없으면 최적 validation 가중치를 활성화한다.
        promoted = (gate is None or score <= gate) and (current_score is None or score < current_score)
        result = {"promoted": promoted, "rmsle": score, "current_rmsle": current_score,
                  "mae": sum(abs(a - p) for a, p in zip(actual, predictions)) / len(actual),
                  "gate": gate, "training_count": len(training), "validation_count": len(validation),
                  "run_id": run.info.run_id, **selection}
        mlflow.log_params({"mode": mode, "max_epochs": epochs, "patience": patience,
                           "training_count": len(training), "validation_count": len(validation),
                           "sequence_length": 3, "n_features": preprocessor.n_features,
                           "deployment_gate": gate if gate is not None else "validation_selection"})
        mlflow.log_metrics({"rmsle": score, "mae": result["mae"], **selection})
        if current_score is not None:
            mlflow.log_metric("current_rmsle", current_score)

        directory = MODEL_DIR / run.info.run_id
        directory.mkdir(parents=True, exist_ok=False)
        model.save(directory / "model.keras")
        preprocessor.save(directory / "preprocessor.json")
        used_ids = set(current.metadata["used_video_ids"]) if current else set()
        used_ids.update(row["video_id"] for row in training + validation)
        metadata = {"used_video_ids": sorted(used_ids), "trained_at": datetime.now(timezone.utc).isoformat(),
                    "last_published_at": max(row["published_at"] for row in training + validation),
                    "training_video_ids": [r["video_id"] for r in training],
                    "validation_video_ids": [r["video_id"] for r in validation], **result}
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        mlflow.log_artifacts(str(directory), artifact_path="bundle")
        example = np.asarray([preprocessor.transform(training[0])], dtype="float32")
        logged = mlflow.tensorflow.log_model(model, name="model", input_example=example)
        if promoted:
            result["version"] = promote_model(run.info.run_id, logged.model_uri, validation[0])
            logger.info("[OK] RMSLE=%.4f, production=v%s", score, result["version"])
        else:
            logger.warning("[WARN] RMSLE=%.4f, gate failed; existing model retained", score)
        mlflow.set_tag("promoted", promoted)
        return result


def fine_tune(training_rows, validation_rows, epochs=10, expected_version=None):
    return train_and_register(training_rows=training_rows, validation_rows=validation_rows,
                              mode="fine_tune", epochs=epochs, expected_version=expected_version)


def main():
    parser = argparse.ArgumentParser(description="ViewPilot 분리된 CSV 학습")
    parser.add_argument("--train", help="학습 CSV. 생략 시 해당 용도의 최신 업로드")
    parser.add_argument("--validation", help="검증 CSV. 생략 시 해당 용도의 최신 업로드")
    parser.add_argument("--mode", choices=["initial", "fine_tune"], default="initial")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    result = train_and_register(args.train, args.validation, mode=args.mode,
                                epochs=args.epochs, patience=args.patience)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
