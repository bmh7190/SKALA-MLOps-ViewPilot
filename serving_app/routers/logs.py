"""학습·드리프트 로그를 대시보드에 표시한다."""
from fastapi import APIRouter, HTTPException

from serving_app.config import LOG_DIR

router = APIRouter(prefix="/api/v1/logs", tags=["모델 운영"])


@router.get("", summary="운영 로그 파일 목록")
def list_logs():
    return [{"name": p.name, "size": p.stat().st_size} for p in sorted(LOG_DIR.glob("*.log")) if p.is_file()]


@router.get("/{filename}", summary="학습·드리프트 로그 조회")
def read_log(filename: str):
    if filename != "aiops.log":
        raise HTTPException(404, "로그 파일을 찾을 수 없습니다.")
    path = LOG_DIR / filename
    if not path.is_file():
        raise HTTPException(404, "로그 파일을 찾을 수 없습니다.")
    return {"name": filename, "content": path.read_text(encoding="utf-8")[-50000:]}
