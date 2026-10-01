"""학습용 CSV 업로드와 최신 파일 요약."""
from typing import Literal

from fastapi import APIRouter, File, UploadFile

from data.features import load_rows, parse_csv
from data.storage import latest_upload, save_upload

router = APIRouter(prefix="/data")
Purpose = Literal["train", "validation", "retrain_train", "retrain_validation"]


async def read_csv_file(file: UploadFile, require_target: bool = True) -> tuple[str, list[dict]]:
    raw = await file.read(10 * 1024 * 1024 + 1)
    if len(raw) > 10 * 1024 * 1024:
        raise ValueError("CSV는 10MB 이하로 업로드하세요.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("UTF-8 CSV만 업로드할 수 있습니다.") from error
    return text, parse_csv(text, require_target)


@router.post("/upload")
async def upload(file: UploadFile = File(...), purpose: Purpose = "train"):
    text, rows = await read_csv_file(file)
    path = save_upload(text, purpose)
    return {"filename": path.name, "rows": len(rows), "purpose": purpose}


@router.get("/status")
def status(purpose: Purpose = "train"):
    try:
        path = latest_upload(purpose)
    except FileNotFoundError:
        return {"exists": False}
    rows = load_rows(path)
    published = sorted(row["published_at"] for row in rows)
    return {"exists": True, "filename": path.name, "rows": len(rows),
            "start_date": published[0], "end_date": published[-1]}
