"""서버 시작 시 원본 모델을 시연 전용 runtime으로 복원한다."""
import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

from serving_app.config import BASE_RUNTIME_DIR


def reset_demo_runtime(source: Path = BASE_RUNTIME_DIR):
    source = source.resolve()
    destination = source / "demo"
    marker = ".viewpilot-demo"
    # 임의의 기존 폴더를 지우지 않는다. 이 기능이 만든 시연 폴더만 재설정한다.
    if destination.is_symlink() or (destination.exists() and not (destination / marker).is_file()):
        raise ValueError("runtime/demo가 시연 전용 폴더가 아닙니다. 다른 저장 위치를 사용하세요.")
    pointer_path = source / "models" / "production.json"
    if not pointer_path.is_file():
        raise ValueError("DEMO_MODE=false에서 최초 학습과 기준값 계산을 먼저 완료하세요.")
    pointer = json.loads(pointer_path.read_text())
    bundle_id, version = pointer["bundle_id"], pointer["version"]
    if any(Path(value).name != value or value in (".", "..") for value in (bundle_id, version)):
        raise ValueError("원본 모델의 저장 경로가 올바르지 않습니다.")
    bundle = source / "models" / bundle_id
    baseline_path = source / "baselines" / f"{version}.json"
    required = [bundle / name for name in ("model.keras", "preprocessor.json", "metadata.json")]
    required += [baseline_path, source / "mlflow.db"]
    if any(not path.is_file() for path in required):
        raise ValueError("시연 원본의 모델·전처리·MLflow DB·기준값 파일을 확인하세요.")
    baseline = json.loads(baseline_path.read_text())
    metadata = json.loads((bundle / "metadata.json").read_text())
    if baseline["model_version"] != version or baseline["run_id"] != metadata["run_id"]:
        raise ValueError("시연 원본 모델과 기준값이 일치하지 않습니다.")

    # 새 복사본이 준비된 후 교체한다. 관측 DB·업로드·로그·후보 모델은 복사하지 않는다.
    with tempfile.TemporaryDirectory(prefix=".demo-prepare-", dir=source) as temporary:
        prepared = Path(temporary) / "demo"
        shutil.copytree(bundle, prepared / "models" / bundle_id)
        shutil.copy2(pointer_path, prepared / "models" / "production.json")
        (prepared / "baselines").mkdir()
        shutil.copy2(baseline_path, prepared / "baselines" / baseline_path.name)
        with sqlite3.connect(f"{(source / 'mlflow.db').as_uri()}?mode=ro", uri=True) as original:
            with sqlite3.connect(prepared / "mlflow.db") as copied:
                original.backup(copied)
                # 새 실험 산출물이 원본 폴더에 쓰이지 않도록 목적지를 변경한다.
                copied.execute("UPDATE experiments SET artifact_location=? WHERE name=?",
                               ((destination / "mlruns").as_uri(), "viewpilot"))
        (prepared / marker).write_text("ViewPilot disposable demo runtime\n")
        if destination.exists():
            shutil.rmtree(destination)
        prepared.replace(destination)

    # 기준값은 복원한 모델에 맞춘다. 사용자의 .env 파일은 수정하지 않는다.
    os.environ["BASELINE_RMSLE"] = str(baseline["baseline_rmsle"])
    os.environ["BASELINE_MODEL_VERSION"] = version
    os.environ["MODEL_SOURCE"] = "local"
    return {"model_version": version, "baseline_rmsle": baseline["baseline_rmsle"]}
