"""CSV와 API가 함께 사용하는 영상 데이터 계약."""
from datetime import datetime, timezone
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator


class VideoInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)

    video_id: str = Field(min_length=1, max_length=200)
    published_at: AwareDatetime
    day1_views: int = Field(ge=0)
    day1_impressions: int = Field(ge=0)
    day1_avg_view_percentage: float = Field(ge=0)
    day2_views: int = Field(ge=0)
    day2_impressions: int = Field(ge=0)
    day2_avg_view_percentage: float = Field(ge=0)
    day3_views: int = Field(ge=0)
    day3_impressions: int = Field(ge=0)
    day3_avg_view_percentage: float = Field(ge=0)
    subscriber_count_at_publish: int = Field(ge=0)
    category: str = Field(min_length=1, max_length=100)

    @field_validator("published_at")
    @classmethod
    def use_utc(cls, value: datetime) -> datetime:
        return value.astimezone(timezone.utc)

    @field_validator("category")
    @classmethod
    def normalize_category(cls, value: str) -> str:
        return value.lower()

    @property
    def initial_views(self) -> int:
        return self.day1_views + self.day2_views + self.day3_views

    def features_dict(self) -> dict:
        # 정답이 있는 객체라도 모델에는 입력 컬럼만 전달한다.
        return self.model_dump(mode="json", include=set(VideoInput.model_fields))


class LabeledVideo(VideoInput):
    target_views_day7: int = Field(ge=0)

    @model_validator(mode="after")
    def check_cumulative_views(self):
        if self.target_views_day7 < self.initial_views:
            raise ValueError("7일 누적 조회수는 첫 3일 조회수 합계보다 작을 수 없습니다.")
        return self


class PredictResponse(BaseModel):
    prediction_id: str
    video_id: str
    predicted_views_day7: int
    model_version: str


class BatchTestRequest(BaseModel):
    videos: list[LabeledVideo] = Field(min_length=1, max_length=1000)
    # 시나리오에서만 사용한다. 이 영상들은 드리프트 평가나 가중치 학습에 넣지 않는다.
    retraining_validation: list[LabeledVideo] | None = Field(default=None, min_length=10, max_length=1000)

    @field_validator("videos")
    @classmethod
    def unique_videos(cls, videos):
        if len({v.video_id for v in videos}) != len(videos):
            raise ValueError("같은 요청에 video_id가 중복되었습니다.")
        return videos


class ObservationRequest(BaseModel):
    prediction_id: str
    target_views_day7: int = Field(ge=0)


class TrainingRequest(BaseModel):
    mode: str = Field(default="initial", pattern="^(initial|fine_tune)$")
    epochs: int = Field(default=100, ge=1, le=300)
    patience: int = Field(default=10, ge=1, le=100)


class PredictionHistoryItem(PredictResponse):
    mode: Literal["live", "simulation"]
    target_views_day7: int | None
    predicted_at: AwareDatetime
    observed_at: AwareDatetime | None
    block_id: int | None


class DriftHistoryItem(BaseModel):
    block_id: int
    model_version: str
    mode: Literal["live", "simulation"]
    baseline_rmsle: float
    rmsle: float
    threshold: float
    status: Literal["ok", "warning", "retrain_review"]
    consecutive_exceeds: int
    video_ids: list[str]


class HistoryPage(BaseModel):
    total: int
    page: int
    page_size: int


class PredictionHistoryResponse(HistoryPage):
    items: list[PredictionHistoryItem]


class DriftHistoryResponse(HistoryPage):
    items: list[DriftHistoryItem]
