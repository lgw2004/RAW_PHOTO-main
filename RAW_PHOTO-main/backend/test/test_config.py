import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT_DIR = Path(__file__).resolve().parents[2]
ROOT_CONFIG_FILE = ROOT_DIR / "config.json"


class ConfigLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._created_root_config = False
        if not ROOT_CONFIG_FILE.exists():
            ROOT_CONFIG_FILE.write_text(json.dumps({"auth-key": "test-auth"}), encoding="utf-8")
            cls._created_root_config = True

        from services import config as config_module

        cls.config_module = config_module

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._created_root_config and ROOT_CONFIG_FILE.exists():
            ROOT_CONFIG_FILE.unlink()

    def test_load_settings_ignores_directory_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            data_dir = base_dir / "data"
            config_dir = base_dir / "config.json"
            os_auth_key = "env-auth"

            config_dir.mkdir()

            module = self.config_module
            old_base_dir = module.BASE_DIR
            old_data_dir = module.DATA_DIR
            old_config_file = module.CONFIG_FILE
            old_env_auth_key = module.os.environ.get("LGWRAW_AUTH_KEY")
            try:
                module.BASE_DIR = base_dir
                module.DATA_DIR = data_dir
                module.CONFIG_FILE = config_dir
                module.os.environ["LGWRAW_AUTH_KEY"] = os_auth_key

                settings = module._load_settings()

                self.assertEqual(settings.auth_key, os_auth_key)
                self.assertEqual(settings.refresh_account_interval_minute, 5)
            finally:
                module.BASE_DIR = old_base_dir
                module.DATA_DIR = old_data_dir
                module.CONFIG_FILE = old_config_file
                if old_env_auth_key is None:
                    module.os.environ.pop("LGWRAW_AUTH_KEY", None)
                else:
                    module.os.environ["LGWRAW_AUTH_KEY"] = old_env_auth_key

    def test_minio_uses_standard_environment_names_and_session_token(self) -> None:
        module = self.config_module
        with mock.patch.dict(
            module.os.environ,
            {
                "LGWRAW_IMAGE_STORAGE_ENABLED": "true",
                "LGWRAW_IMAGE_STORAGE_MODE": "minio",
                "MINIO_ENDPOINT": "http://minio.example.test:9000",
                "MINIO_ACCESS_KEY": "access",
                "MINIO_SECRET_KEY": "secret",
                "MINIO_SESSION_TOKEN": "session",
                "MINIO_BUCKET": "raw-photo",
                "MINIO_SECURE": "false",
                "MINIO_REGION": "cn-beijing",
            },
            clear=True,
        ):
            storage = module._normalize_image_storage_settings({})

        self.assertEqual(storage["provider"], "minio")
        self.assertEqual(storage["minio_endpoint"], "http://minio.example.test:9000")
        self.assertEqual(storage["minio_session_token"], "session")
        self.assertFalse(storage["minio_secure"])

    def test_reference_upload_uses_minio_configuration(self) -> None:
        module = self.config_module
        reference_config = {
            "enabled": True,
            "provider": "minio",
            "minio_endpoint": "http://minio.example.test:9000",
            "minio_access_key": "ak",
            "minio_secret_key": "sk",
            "minio_session_token": "session",
            "minio_bucket": "bucket",
            "minio_root_path": "reference",
        }

        with mock.patch.dict(module.os.environ, {}, clear=True):
            reference = module._normalize_image_reference_upload_settings(reference_config)

        self.assertEqual(reference["provider"], "minio")
        self.assertEqual(reference["minio_access_key"], "ak")
        self.assertEqual(reference["minio_secret_key"], "sk")
        self.assertEqual(reference["minio_session_token"], "session")
        self.assertEqual(reference["minio_root_path"], "reference")

    def test_openai_relay_accepts_environment_api_key_pool(self) -> None:
        module = self.config_module
        with mock.patch.dict(
            module.os.environ,
            {
                "LGWRAW_OPENAI_RELAY_API_KEY": "legacy-key",
                "LGWRAW_OPENAI_RELAY_API_KEYS": "pool-a\npool-b,pool-a",
                "LGWRAW_OPENAI_RELAY_POOL_DISTRIBUTED": "true",
            },
            clear=False,
        ):
            settings = module._normalize_openai_relay_settings({"base_url": "https://relay.example/v1"})

        self.assertEqual(settings["api_key"], "legacy-key")
        self.assertEqual(settings["api_keys"], ["pool-a", "pool-b"])
        self.assertTrue(settings["api_key_pool_distributed"])


if __name__ == "__main__":
    unittest.main()
