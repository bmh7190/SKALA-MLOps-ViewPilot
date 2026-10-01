"""영상 한 행을 3일 시퀀스로 변환한다. 학습과 예측이 이 함수를 공유한다."""
import csv
import io
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import ValidationError

from serving_app.schemas import LabeledVideo, VideoInput

SEQ_LEN = 3
INPUT_COLUMNS = list(VideoInput.model_fields)
CSV_COLUMNS = list(LabeledVideo.model_fields)
MIN_TRAIN_ROWS = 30
MIN_VALIDATION_ROWS = 10


def parse_csv(text: str, require_target: bool = True) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    columns = reader.fieldnames or []
    required = CSV_COLUMNS if require_target else INPUT_COLUMNS
    if len(columns) != len(set(columns)):
        raise ValueError("CSV 헤더에 중복 컬럼이 있습니다.")
    if set(required) - set(columns) or set(columns) - set(CSV_COLUMNS):
        raise ValueError(f"CSV 컬럼을 확인하세요. 필수 컬럼: {', '.join(required)}")

    rows, seen = [], set()
    for line, raw in enumerate(reader, start=2):
        try:
            if require_target:
                video = LabeledVideo.model_validate(raw)
            else:
                # 추론에서는 정답 컬럼이 있어도 입력 피처에 포함하지 않는다.
                raw.pop("target_views_day7", None)
                video = VideoInput.model_validate(raw)
        except ValidationError as error:
            first = error.errors()[0]
            field = ".".join(str(x) for x in first["loc"]) or "row"
            raise ValueError(f"CSV {line}행 {field}: {first['msg']}") from error
        if video.video_id in seen:
            raise ValueError(f"CSV {line}행: video_id가 중복되었습니다 ({video.video_id}).")
        seen.add(video.video_id)
        rows.append(video.model_dump(mode="json"))
    if not rows:
        raise ValueError("CSV에 영상 데이터가 없습니다.")
    return rows


def load_rows(csv_path: str | Path, require_target: bool = True) -> list[dict]:
    return parse_csv(Path(csv_path).read_text(encoding="utf-8-sig"), require_target)


def validate_splits(training_rows: list[dict], validation_rows: list[dict], *, retrospective=False):
    """지정된 두 파일을 그대로 사용하되 중복·시간 누수를 검사한다."""
    training = [LabeledVideo.model_validate(row) for row in training_rows]
    validation = [LabeledVideo.model_validate(row) for row in validation_rows]
    if len(training) < MIN_TRAIN_ROWS or len(validation) < MIN_VALIDATION_ROWS:
        raise ValueError("학습 영상 30개, 검증 영상 10개 이상이 필요합니다.")
    combined = training + validation
    if len({v.video_id for v in combined}) != len(combined):
        raise ValueError("학습·검증 파일 안이나 파일 사이에 중복 video_id가 있습니다.")
    now = datetime.now(timezone.utc)
    if any(v.published_at + timedelta(hours=168) > now for v in combined):
        raise ValueError("게시 후 168시간이 지나지 않은 영상은 학습·검증할 수 없습니다.")
    first_validation = min(v.published_at for v in validation)
    last_training = max(v.published_at for v in training)
    if last_training >= first_validation:
        raise ValueError("검증 영상은 모든 학습 영상보다 나중에 게시되어야 합니다.")
    # 과거 CSV 시나리오는 모든 정답을 확보한 뒤 비교한다. 실시간 재생에서는 4일 간격이 필요하다.
    if not retrospective and last_training + timedelta(hours=168) > first_validation + timedelta(hours=72):
        raise ValueError("첫 검증 예측 시점에 학습 정답을 알 수 있도록 두 파일의 게시 시각을 분리하세요.")
    return ([v.model_dump(mode="json") for v in training],
            [v.model_dump(mode="json") for v in validation])


class VideoPreprocessor:
    """훈련 데이터로만 스케일과 카테고리 사전을 정하고 JSON으로 저장한다."""

    def __init__(self, categories: list[str], means: list[float], scales: list[float]):
        self.categories = categories
        self.means = means
        self.scales = scales

    @staticmethod
    def daily_values(row: dict, day: int) -> list[float]:
        return [
            math.log1p(row[f"day{day}_views"]),
            math.log1p(row[f"day{day}_impressions"]),
            row[f"day{day}_avg_view_percentage"] / 100.0,
            math.log1p(row["subscriber_count_at_publish"]),
        ]

    @classmethod
    def fit(cls, training_rows: list[dict]) -> "VideoPreprocessor":
        if not training_rows:
            raise ValueError("전처리 학습 데이터가 없습니다.")
        points = [cls.daily_values(row, day) for row in training_rows for day in range(1, 4)]
        means = [sum(p[i] for p in points) / len(points) for i in range(4)]
        scales = [math.sqrt(sum((p[i] - means[i]) ** 2 for p in points) / len(points)) or 1.0 for i in range(4)]
        known = {row["category"] for row in training_rows}
        preferred = ["gaming", "education", "entertainment", "lifestyle"]
        categories = [name for name in preferred if name in known]
        categories += sorted(known - set(preferred))
        return cls(categories, means, scales)

    @property
    def n_features(self) -> int:
        return 4 + len(self.categories)

    def transform(self, row: dict) -> list[list[float]]:
        # 구독자 수와 카테고리는 3일 모두 반복한다. 새 카테고리는 all-zero로 처리한다.
        category = [float(row["category"] == name) for name in self.categories]
        sequence = []
        for day in range(1, 4):
            values = self.daily_values(row, day)
            scaled = [(value - mean) / scale for value, mean, scale in zip(values, self.means, self.scales)]
            sequence.append(scaled + category)
        return sequence

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VideoPreprocessor":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def build_sequences(rows: list[dict], preprocessor: VideoPreprocessor):
    # 한 행이 한 샘플이다. 영상 사이에 슬라이딩 윈도우를 적용하지 않는다.
    X = [preprocessor.transform(row) for row in rows]
    y = [math.log1p(row["target_views_day7"]) for row in rows]
    return X, y


def restore_prediction(log_views: float, row: dict) -> int:
    if not math.isfinite(log_views):
        raise ValueError("모델이 유한하지 않은 예측값을 반환했습니다.")
    try:
        predicted = math.expm1(max(0.0, log_views))
    except OverflowError as error:
        raise ValueError("모델 예측값이 허용 범위를 초과했습니다.") from error
    if not math.isfinite(predicted):
        raise ValueError("모델 예측값이 허용 범위를 초과했습니다.")
    initial = sum(row[f"day{day}_views"] for day in range(1, 4))
    # 누적 조회수의 하한을 보장하며, 평가와 서빙에서 같은 후처리를 사용한다.
    return max(initial, round(predicted))
