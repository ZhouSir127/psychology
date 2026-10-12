"""主线一、二只需遵守这些数据契约；维度和模型架构没有写死。"""
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Submission(BaseModel):
    subject_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    collection_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    collected_at: datetime
    text: str = Field(default="", max_length=4000)
    scenario: str | None = None
    source: Literal["demo", "uploaded"] = "uploaded"

    @field_validator("collected_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("采集时间必须带时区，例如 2026-10-06T10:00:00+08:00")
        return value


class Sample(BaseModel):
    subject_id: str
    collection_id: str
    collected_at: datetime
    text: str = ""
    files: dict[str, Path] = Field(default_factory=dict)
    source: str


class ModalityFeature(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    values: list[float]
    names: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class FeatureBundle(BaseModel):
    schema_version: str = "1.0"
    extractor_version: str
    modalities: dict[str, ModalityFeature]
    missing_modalities: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Prediction(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    score: float = Field(ge=0, le=100)
    model_version: str
    is_simulated: bool
    explanation: str


class FeatureExtractor(Protocol):
    def extract(self, sample: Sample) -> FeatureBundle: ...


class Predictor(Protocol):
    def predict(self, features: FeatureBundle, sample: Sample,
                history: list[dict], scenario: str | None) -> Prediction: ...
