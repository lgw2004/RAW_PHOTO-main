from __future__ import annotations

import os
from typing import Any


def normalize_sync_database_url(value: object) -> str:
    """Adapt asyncpg URLs for the project's synchronous SQLAlchemy sessions."""
    database_url = str(value or "").strip()
    if "://" not in database_url:
        return database_url
    scheme, remainder = database_url.split("://", 1)
    if scheme in {"postgres+asyncpg", "postgresql+asyncpg"}:
        scheme = "postgresql+psycopg2"
    return f"{scheme}://{remainder}"


def resolve_database_url(*fallback_env_names: str, default: str = "") -> str:
    """Resolve shared DATABASE_URL before service-specific compatibility names."""
    for env_name in ("DATABASE_URL", *fallback_env_names):
        value = os.getenv(env_name)
        if value and value.strip():
            return normalize_sync_database_url(value)
    return normalize_sync_database_url(default)


def create_sync_engine(database_url: object, **kwargs: Any):
    from sqlalchemy import create_engine

    return create_engine(normalize_sync_database_url(database_url), **kwargs)
