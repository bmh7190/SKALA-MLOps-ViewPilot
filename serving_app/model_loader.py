"""모델과 전처리 설정을 같은 버전으로 로드하고 캐시한다."""
import json
import logging
import os
from threading import RLock

from data.features import VideoPreprocessor, restore_prediction
from serving_app.config import DEMO_MODE, MODEL_DIR, MODEL_NAME, PRODUCTION_FILE, RUNTIME_DIR

logger = logging.getLogger("aiops")
_model_cache = None
_cache_pointer = None
_cache_lock = RLock()


def configure_mlflow():
    import mlflow

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    local_uri = f"sqlite:///{RUNTIME_DIR / 'mlflow.db'}"
    tracking_uri = local_uri if DEMO_MODE else os.getenv("MLFLOW_TRACKING_URI", local_uri)
    mlflow.set_tracking_uri(tracking_uri)
    if DEMO_MODE:
        mlflow.set_registry_uri(local_uri)
    experiment = mlflow.get_experiment_by_name("viewpilot")
    if experiment is None:
        mlflow.create_experiment("viewpilot", artifact_location=(RUNTIME_DIR / "mlruns").as_uri())
    mlflow.set_experiment("viewpilot")


class LoadedModel:
    def __init__(self, keras_model, preprocessor: VideoPreprocessor, metadata: dict, version: str):
        self.keras_model = keras_model
        self.preprocessor = preprocessor
        self.metadata = metadata
        self.version = version

    def predict_many(self, rows: list[dict]) -> list[int]:
        import numpy as np

        inputs = np.asarray([self.preprocessor.transform(row) for row in rows], dtype="float32")
        # 작은 배치이므로 직접 호출한다. 평가와 API가 같은 역변환 함수를 공유한다.
        outputs = np.asarray(self.keras_model(inputs, training=False)).reshape(-1)
        return [restore_prediction(float(value), row) for value, row in zip(outputs, rows)]

    def predict_one(self, row: dict) -> int:
        return self.predict_many([row])[0]


def load_bundle(bundle_id: str, version: str, source: str = "local") -> LoadedModel:
    from tensorflow import keras

    directory = MODEL_DIR / bundle_id
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    preprocessor = VideoPreprocessor.load(directory / "preprocessor.json")
    if source == "mlflow":
        import mlflow.tensorflow

        configure_mlflow()
        model = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/{version}")
    elif source == "local":
        model = keras.models.load_model(directory / "model.keras", compile=False)
    else:
        raise ValueError("MODEL_SOURCE는 local 또는 mlflow여야 합니다.")
    if tuple(model.input_shape[1:]) != (3, preprocessor.n_features):
        raise ValueError("모델과 전처리 설정의 입력 크기가 일치하지 않습니다.")
    return LoadedModel(model, preprocessor, metadata, version)


def get_model() -> LoadedModel:
    global _model_cache, _cache_pointer
    with _cache_lock:
        if not PRODUCTION_FILE.exists():
            raise FileNotFoundError("배포된 ViewPilot 모델이 없습니다. 영상 CSV로 최초 학습을 진행하세요.")
        pointer_text = PRODUCTION_FILE.read_text(encoding="utf-8")
        if _model_cache is None or _cache_pointer != pointer_text:
            pointer = json.loads(pointer_text)
            loaded = load_bundle(pointer["bundle_id"], pointer["version"], os.getenv("MODEL_SOURCE", "local"))
            # 새 모델을 완전히 로드한 후 캐시를 교체한다. 로딩 도중 기존 객체를 수정하지 않는다.
            _model_cache = loaded
            _cache_pointer = pointer_text
            logger.info("model loaded: ViewPilot v%s", loaded.version)
        return _model_cache


def clear_cache():
    global _model_cache, _cache_pointer
    with _cache_lock:
        _model_cache = None
        _cache_pointer = None


def load_eager() -> LoadedModel:
    return get_model()
