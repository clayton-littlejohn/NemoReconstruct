from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "NemoReconstruct"
    api_prefix: str = "/api/v1"
    docs_url: str = "/docs"
    openapi_url: str = "/openapi.json"
    redoc_url: str = "/redoc"
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    base_dir: Path = Path(__file__).resolve().parents[3]
    storage_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "storage")
    database_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "nemo_reconstruct.db")

    ffmpeg_bin: str = "ffmpeg"
    colmap_bin: str = "colmap"
    frgs_bin: str = "frgs"
    fvdb_conda_env: str = "fvdb"
    fvdb_conda_root: Path = Field(default_factory=lambda: Path.home() / "miniconda3" / "envs")

    frame_rate: float = 2.0
    max_upload_size_mb: int = 4096
    sequential_matcher_overlap: int = 12
    colmap_mapper_type: str = "incremental"
    colmap_max_num_features: int = 8192
    fvdb_max_epochs: int = 40
    fvdb_sh_degree: int = 3
    fvdb_image_downsample_factor: int = 6
    splat_only_mode: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="NEMO_RECONSTRUCT_",
        extra="ignore",
    )


settings = Settings()
