from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    pass


settings.database_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"sqlite:///{settings.database_path}",
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def ensure_runtime_schema() -> None:
    if engine.dialect.name != "sqlite":
        return

    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info(reconstructions)")).mappings().all()
        columns = {row["name"] for row in rows}
        if "processing_params_json" not in columns:
            conn.execute(text("ALTER TABLE reconstructions ADD COLUMN processing_params_json TEXT"))

        # Create iteration_records table if it doesn't exist
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS iteration_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reconstruction_id VARCHAR(36) NOT NULL,
                iteration INTEGER NOT NULL,
                params_json TEXT,
                metrics_json TEXT,
                ply_path TEXT,
                verdict VARCHAR(64),
                reason TEXT,
                loss FLOAT,
                ssim FLOAT,
                num_gaussians INTEGER,
                started_at DATETIME,
                completed_at DATETIME,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_iteration_records_reconstruction_id
            ON iteration_records (reconstruction_id)
        """))
