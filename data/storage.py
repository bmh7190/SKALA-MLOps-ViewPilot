"""용도별 CSV 보관. 검증·테스트 파일이 학습 파일을 덮어쓰지 않는다."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from serving_app.config import UPLOAD_DIR

PURPOSES = {"train", "validation", "retrain_train", "retrain_validation"}


def save_upload(text: str, purpose: str = "train") -> Path:
    if purpose not in PURPOSES:
        raise ValueError("CSV 용도를 확인하세요.")
    directory = UPLOAD_DIR / purpose
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = directory / f"videos_{stamp}_{uuid4().hex[:8]}.csv"
    path.write_text(text, encoding="utf-8")
    return path


def latest_upload(purpose: str = "train") -> Path:
    if purpose not in PURPOSES:
        raise ValueError("CSV 용도를 확인하세요.")
    files = list((UPLOAD_DIR / purpose).glob("videos_*.csv"))
    if not files:
        raise FileNotFoundError(f"{purpose} 용도의 영상 CSV를 먼저 업로드하세요.")
    return max(files, key=lambda p: p.stat().st_mtime_ns)
