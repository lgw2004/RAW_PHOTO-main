import os
import unittest
from unittest.mock import patch

from services.database_utils import normalize_sync_database_url, resolve_database_url


class DatabaseUtilsTests(unittest.TestCase):
    def test_normalizes_asyncpg_for_sync_sqlalchemy(self) -> None:
        url = "postgresql+asyncpg://dev_sh_ai:secret@127.0.0.1:5432/dev_sh_ai_db"
        self.assertEqual(
            normalize_sync_database_url(url),
            "postgresql+psycopg2://dev_sh_ai:secret@127.0.0.1:5432/dev_sh_ai_db",
        )

    def test_database_url_has_precedence(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql+asyncpg://shared@db/shared",
                "IMAGE_TASK_DATABASE_URL": "sqlite:///task.db",
            },
            clear=False,
        ):
            self.assertEqual(
                resolve_database_url("IMAGE_TASK_DATABASE_URL"),
                "postgresql+psycopg2://shared@db/shared",
            )


if __name__ == "__main__":
    unittest.main()
