"""최초 예측과 나중에 도착한 실제값을 SQLite에 보관한다."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from serving_app.config import OBSERVATION_DB
from serving_app.monitoring.drift_detector import WINDOW_SIZE, evaluate_block
from serving_app.schemas import VideoInput


@contextmanager
def connection():
    OBSERVATION_DB.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(OBSERVATION_DB, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS forecasts (
                id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                mode TEXT NOT NULL,
                inputs TEXT NOT NULL,
                predicted INTEGER NOT NULL,
                predicted_at TEXT NOT NULL,
                actual INTEGER,
                observed_at TEXT,
                block_id INTEGER,
                UNIQUE(video_id, model_version, mode)
            );
            CREATE TABLE IF NOT EXISTS evaluation_blocks (
                id INTEGER PRIMARY KEY,
                model_version TEXT NOT NULL,
                mode TEXT NOT NULL,
                baseline REAL NOT NULL,
                result TEXT NOT NULL
            );
        """)
        # 같은 순간에 들어온 요청도 같은 영상을 중복 저장·평가하지 못하게 한다.
        db.execute("BEGIN IMMEDIATE")
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def response_for(record) -> dict:
    return {
        "prediction_id": record["id"], "video_id": record["video_id"],
        "model_version": record["model_version"], "predicted_views_day7": record["predicted"],
    }


def save_predictions(rows: list[dict], predictions: list[int], version: str, mode: str) -> list[dict]:
    if len(rows) != len(predictions):
        raise ValueError("영상과 예측값 개수가 다릅니다.")
    now = datetime.now(timezone.utc).isoformat()
    responses = []
    with connection() as db:
        for row, predicted in zip(rows, predictions):
            inputs = json.dumps(VideoInput.model_validate({k: row[k] for k in VideoInput.model_fields}).features_dict(), sort_keys=True)
            existing = db.execute(
                "SELECT * FROM forecasts WHERE video_id=? AND model_version=? AND mode=?",
                (row["video_id"], version, mode),
            ).fetchone()
            actual = row["target_views_day7"] if mode == "simulation" else None
            if existing:
                if existing["inputs"] != inputs or (actual is not None and existing["actual"] != actual):
                    raise ValueError(f"{row['video_id']}: 이미 저장된 입력 또는 실제값과 다릅니다.")
                responses.append(response_for(existing))
                continue
            record = {"id": uuid4().hex, "video_id": row["video_id"], "model_version": version, "predicted": predicted}
            db.execute(
                "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (record["id"], row["video_id"], version, mode, inputs, predicted, now, actual, now if actual is not None else None),
            )
            responses.append(response_for(record))
    return responses


def save_actual(prediction_id: str, actual: int) -> dict:
    with connection() as db:
        record = db.execute("SELECT * FROM forecasts WHERE id=? AND mode='live'", (prediction_id,)).fetchone()
        if record is None:
            raise LookupError("해당 prediction_id의 운영 예측을 찾을 수 없습니다.")
        video = VideoInput.model_validate_json(record["inputs"])
        if datetime.now(timezone.utc) < video.published_at + timedelta(hours=168):
            raise ValueError("게시 후 168시간이 지난 영상만 실제값을 등록할 수 있습니다.")
        if actual < video.initial_views:
            raise ValueError("7일 누적 조회수는 첫 3일 조회수 합계보다 작을 수 없습니다.")
        if record["actual"] is not None and record["actual"] != actual:
            raise ValueError("이미 등록된 실제값과 다릅니다. 평가 완료 값은 덮어쓰지 않습니다.")
        db.execute(
            "UPDATE forecasts SET actual=?, observed_at=COALESCE(observed_at, ?) WHERE id=?",
            (actual, datetime.now(timezone.utc).isoformat(), prediction_id),
        )
        return dict(record)


def evaluate_pending(version: str, mode: str, baseline: float | None) -> dict:
    with connection() as db:
        last = db.execute(
            "SELECT * FROM evaluation_blocks WHERE model_version=? AND mode=? ORDER BY id DESC LIMIT 1",
            (version, mode),
        ).fetchone()
        if last and baseline is not None and last["baseline"] != baseline:
            raise ValueError("같은 모델 버전의 기준 RMSLE는 고정해야 합니다.")
        pending = db.execute(
            "SELECT * FROM forecasts WHERE model_version=? AND mode=? AND actual IS NOT NULL AND block_id IS NULL ORDER BY observed_at, rowid",
            (version, mode),
        ).fetchall()
        result = json.loads(last["result"]) if last else {"status": "collecting", "consecutive_exceeds": 0}
        blocks = []
        if baseline is not None:
            # 처리한 30개에는 block_id를 붙인다. 다음 평가에서 다시 사용하지 않는다.
            for start in range(0, len(pending) - WINDOW_SIZE + 1, WINDOW_SIZE):
                block = pending[start:start + WINDOW_SIZE]
                result = evaluate_block(
                    [r["actual"] for r in block], [r["predicted"] for r in block],
                    baseline, result["consecutive_exceeds"],
                )
                result["video_ids"] = [r["video_id"] for r in block]
                cursor = db.execute(
                    "INSERT INTO evaluation_blocks (model_version, mode, baseline, result) VALUES (?, ?, ?, ?)",
                    (version, mode, baseline, json.dumps(result)),
                )
                block_id = cursor.lastrowid
                db.executemany("UPDATE forecasts SET block_id=? WHERE id=?", [(block_id, r["id"]) for r in block])
                blocks.append({"block_id": block_id, **result})
        return {
            **result, "status": result["status"] if baseline is not None else "baseline_required",
            "model_version": version, "mode": mode, "baseline_rmsle": baseline,
            "pending_count": len(pending) - len(blocks) * WINDOW_SIZE,
            "new_blocks": blocks,
        }


def labeled_rows(mode: str) -> list[dict]:
    with connection() as db:
        records = db.execute(
            "SELECT inputs, actual FROM forecasts WHERE mode=? AND actual IS NOT NULL ORDER BY observed_at",
            (mode,),
        ).fetchall()
    # 같은 영상이 여러 모델 버전으로 예측됐더라도 학습 데이터는 한 행만 사용한다.
    rows = {}
    for record in records:
        row = json.loads(record["inputs"])
        row["target_views_day7"] = record["actual"]
        rows[row["video_id"]] = row
    return list(rows.values())


def recorded_video_ids() -> set[str]:
    """이미 운영·시뮬레이션 평가에 사용한 영상은 별도 학습 데이터로 재사용하지 않는다."""
    if not OBSERVATION_DB.exists():
        return set()
    with connection() as db:
        records = db.execute("SELECT DISTINCT video_id FROM forecasts").fetchall()
    return {row["video_id"] for row in records}
