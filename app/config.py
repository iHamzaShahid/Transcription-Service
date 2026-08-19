"""Application configuration, loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables live here so nothing reads os.environ directly."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Gemini -------------------------------------------------------------
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.1-flash-lite"
    # audio_timestamp is accepted only on Vertex AI; the Developer API rejects
    # it outright. Set these to run against Vertex and get the better
    # timestamps. See the README, "A note on audio_timestamp".
    gemini_use_vertex: bool = False
    google_cloud_project: str | None = None
    google_cloud_location: str = "global"

    # --- Upload limits ------------------------------------------------------
    max_upload_bytes: int = Field(default=100 * 1024 * 1024, ge=1)

    # --- Chunking -----------------------------------------------------------
    # Audio shorter than this is transcribed whole, with no VAD pass at all.
    chunk_size_sec: float = Field(default=120.0, gt=0)
    # How far either side of the target we hunt for a silence to cut in.
    chunk_search_window_sec: float = Field(default=30.0, ge=0)
    # A silence gap must be at least this long to be considered a cut point.
    min_silence_sec: float = Field(default=0.25, gt=0)

    # --- Concurrency & retries ---------------------------------------------
    # Ceiling on chunks in flight to Gemini at once, for any single request.
    max_parallel_chunks: int = Field(default=4, ge=1)
    max_retries: int = Field(default=3, ge=1)
    retry_base_delay_sec: float = Field(default=0.5, gt=0)
    retry_max_delay_sec: float = Field(default=8.0, gt=0)
    request_timeout_sec: float = Field(default=180.0, gt=0)

    # --- Audio --------------------------------------------------------------
    target_sample_rate: int = 16_000
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    ffmpeg_timeout_sec: float = Field(default=300.0, gt=0)

    # --- Silero VAD ---------------------------------------------------------
    vad_model_path: str | None = None
    vad_model_url: str = (
        "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2"
        "/src/silero_vad/data/silero_vad.onnx"
    )
    vad_model_cache_dir: str = "/tmp/silero-vad"
    vad_speech_threshold: float = Field(default=0.5, gt=0, lt=1)
    # Padding kept around detected speech so cuts do not clip word onsets.
    vad_speech_pad_sec: float = Field(default=0.1, ge=0)

    # --- Misc ---------------------------------------------------------------
    log_level: str = "INFO"
    temp_dir: str | None = None

    @property
    def skip_vad_below_sec(self) -> float:
        """Below this the planner cannot produce more than one chunk anyway."""
        return self.chunk_size_sec + self.chunk_search_window_sec


@lru_cache
def get_settings() -> Settings:
    return Settings()
