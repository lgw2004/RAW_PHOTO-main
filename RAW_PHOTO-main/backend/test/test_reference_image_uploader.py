import base64
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest
from unittest.mock import patch

from services import reference_image_uploader


class FakeMinIOClient:
    uploaded: dict[str, bytes] = {}
    settings: dict[str, object] = {}

    def __init__(self, settings):
        self.__class__.settings = dict(settings)

    def put(self, rel: str, payload: bytes, content_type: str = "image/png") -> str:
        self.uploaded[rel] = payload
        return f"minio://bucket/{rel}"

    def public_url(self, rel: str, expires=None) -> str:
        return f"https://minio.example.test/{rel}"


class MinIOReferenceUploadTests(unittest.TestCase):
    def setUp(self):
        reference_image_uploader._upload_url_cache.clear()
        reference_image_uploader._upload_inflight.clear()
        reference_image_uploader._upload_serial_until = 0.0
        FakeMinIOClient.uploaded = {}
        FakeMinIOClient.settings = {}

    @staticmethod
    def minio_settings():
        return {
            "enabled": True,
            "provider": "minio",
            "minio_endpoint": "http://minio.example.test:9000",
            "minio_access_key": "ak",
            "minio_secret_key": "sk",
            "minio_session_token": "session",
            "minio_bucket": "bucket",
            "minio_region": "us-east-1",
            "minio_secure": False,
            "minio_root_path": "reference",
            "public_base_url": "https://cdn.example.test",
            "timeout_sec": 45,
        }

    def test_minio_key_is_stable_for_identical_content(self):
        digest = "a" * 64
        first = reference_image_uploader._minio_key("one.png", digest, "image/png")
        second = reference_image_uploader._minio_key("two.png", digest, "image/png")

        self.assertEqual(first, second)
        self.assertEqual(first, f"sha256/aa/{digest}.png")

    def test_minio_upload_passes_session_token_and_returns_public_url(self):
        with (
            patch.object(reference_image_uploader, "settings", side_effect=self.minio_settings),
            patch("services.image_storage_service.MinIOClient", FakeMinIOClient),
        ):
            url = reference_image_uploader.upload_to_minio(b"image", "image.png", "image/png")

        uploaded_key = next(iter(FakeMinIOClient.uploaded))
        self.assertEqual(url, f"https://minio.example.test/{uploaded_key}")
        self.assertEqual(FakeMinIOClient.settings["minio_session_token"], "session")

    def test_upload_images_reuses_cached_identical_image(self):
        with (
            patch.object(reference_image_uploader, "settings", side_effect=self.minio_settings),
            patch.object(
                reference_image_uploader,
                "upload_to_minio",
                return_value="https://minio.example.test/reference/a.png",
            ) as upload,
        ):
            urls = reference_image_uploader.upload_images([
                (b"same-image", "one.png", "image/png"),
                (b"same-image", "two.png", "image/png"),
            ])

        self.assertEqual(urls, ["https://minio.example.test/reference/a.png"])
        upload.assert_called_once()

    def test_identical_upload_waiter_survives_wait_slice_timeout(self):
        started = threading.Event()
        release = threading.Event()

        def slow_upload(*_args, **_kwargs):
            started.set()
            release.wait(1)
            return "https://minio.example.test/reference/slow.png"

        with (
            patch.object(reference_image_uploader, "settings", side_effect=self.minio_settings),
            patch.object(reference_image_uploader, "upload_to_minio", side_effect=slow_upload) as upload,
            patch.object(reference_image_uploader, "_UPLOAD_INFLIGHT_WAIT_SLICE_SECONDS", 0.01),
            patch.object(reference_image_uploader, "_UPLOAD_INFLIGHT_MAX_WAIT_SECONDS", 1),
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                owner = executor.submit(
                    reference_image_uploader._upload_one_detailed,
                    b"slow-identical",
                    "one.png",
                    "image/png",
                )
                self.assertTrue(started.wait(1))
                waiter = executor.submit(
                    reference_image_uploader._upload_one_detailed,
                    b"slow-identical",
                    "two.png",
                    "image/png",
                )
                time.sleep(0.05)
                self.assertFalse(waiter.done())
                release.set()
                owner_result = owner.result(timeout=2)
                waiter_result = waiter.result(timeout=2)

        self.assertEqual(owner_result.url, waiter_result.url)
        self.assertTrue(waiter_result.cached)
        upload.assert_called_once()

    def test_upload_falls_back_to_data_url_when_requested(self):
        with (
            patch.object(reference_image_uploader, "settings", side_effect=self.minio_settings),
            patch.object(reference_image_uploader, "upload_to_minio", side_effect=RuntimeError("offline")),
        ):
            urls = reference_image_uploader.upload_images(
                [(b"image", "image.png", "image/png")],
                fallback_to_data_url=True,
            )

        self.assertEqual(urls, [f"data:image/png;base64,{base64.b64encode(b'image').decode('ascii')}"])

    def test_two_small_uploads_can_use_both_upload_slots(self):
        first_entered = threading.Event()
        second_entered = threading.Event()
        release = threading.Event()

        def hold_capacity(entered):
            with reference_image_uploader._upload_capacity(1024):
                entered.set()
                release.wait(1)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(hold_capacity, first_entered)
            self.assertTrue(first_entered.wait(1))
            second = executor.submit(hold_capacity, second_entered)
            self.assertTrue(second_entered.wait(1))
            release.set()
            first.result(timeout=2)
            second.result(timeout=2)

    def test_large_upload_exclusively_uses_upload_capacity(self):
        large_entered = threading.Event()
        small_entered = threading.Event()
        release_large = threading.Event()

        def hold_large():
            with reference_image_uploader._upload_capacity(reference_image_uploader._UPLOAD_LARGE_FILE_THRESHOLD):
                large_entered.set()
                release_large.wait(1)

        def hold_small():
            with reference_image_uploader._upload_capacity(1024):
                small_entered.set()

        with ThreadPoolExecutor(max_workers=2) as executor:
            large = executor.submit(hold_large)
            self.assertTrue(large_entered.wait(1))
            small = executor.submit(hold_small)
            self.assertFalse(small_entered.wait(0.05))
            release_large.set()
            large.result(timeout=2)
            small.result(timeout=2)

        self.assertTrue(small_entered.is_set())


if __name__ == "__main__":
    unittest.main()
