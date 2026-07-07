"""SQLAlchemy ORM models and Pydantic schemas."""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.ext.asyncio import AsyncAttrs, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# SQLAlchemy ORM
# ---------------------------------------------------------------------------


class Base(AsyncAttrs, DeclarativeBase):
    pass


class DatasetRow(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hf_id: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    license: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    languages: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    row_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    byte_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    downloads: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    likes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    trending_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_scored: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    scores: Mapped[list[ScoreRow]] = relationship(back_populates="dataset")
    contamination: Mapped[list[ContaminationRow]] = relationship(
        back_populates="dataset"
    )
    training_runs: Mapped[list[TrainingRunRow]] = relationship(
        back_populates="dataset"
    )


class ScoreRow(Base):
    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    scorer_name: Mapped[str] = mapped_column(String(128), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    warnings: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    sample_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    dataset: Mapped[DatasetRow] = relationship(back_populates="scores")


class ContaminationRow(Base):
    __tablename__ = "contamination"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    benchmark_name: Mapped[str] = mapped_column(String(128), nullable=False)
    overlap_rate: Mapped[float] = mapped_column(Float, nullable=False)
    overlap_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_eval: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    method: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    dataset: Mapped[DatasetRow] = relationship(back_populates="contamination")


class TrainingRunRow(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    model_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    tokens_seen: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    final_val_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    loss_curve: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    convergence_steps: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    loss_stability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    eval_scores: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    gpu_hours: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    dataset: Mapped[DatasetRow] = relationship(back_populates="training_runs")


class LeaderboardRow(Base):
    __tablename__ = "leaderboard"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id"), unique=True, nullable=False
    )
    auto_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    training_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    contamination_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    combined_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ---------------------------------------------------------------------------
# Pydantic schemas (API / CLI output)
# ---------------------------------------------------------------------------


class ScoreResult(BaseModel):
    name: str
    score: float = Field(ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ContaminationResult(BaseModel):
    benchmark: str
    overlap_rate: float = Field(ge=0.0, le=1.0)
    overlap_count: int = 0
    total_eval: int = 0
    method: str = "ngram_13"


class TrainingResult(BaseModel):
    final_val_loss: float
    loss_curve: list[float] = Field(default_factory=list)
    convergence_steps: int = 0
    loss_stability: float = 0.0
    relative_quality: float = 0.0


class DatasetSummary(BaseModel):
    hf_id: str
    name: str
    license: Optional[str] = None
    languages: list[str] = Field(default_factory=list)
    row_count: Optional[int] = None
    downloads: Optional[int] = None
    likes: Optional[int] = None
    auto_score: Optional[float] = None
    training_score: Optional[float] = None
    combined_score: Optional[float] = None
    rank: Optional[int] = None


class DatasetDetail(DatasetSummary):
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    scores: list[ScoreResult] = Field(default_factory=list)
    contamination: list[ContaminationResult] = Field(default_factory=list)
    training: Optional[TrainingResult] = None
