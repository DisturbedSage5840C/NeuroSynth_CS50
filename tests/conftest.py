from __future__ import annotations

import os

# Must be set before any C-extension import (LightGBM, torch, sklearn) to avoid
# the OpenMP duplicate-library segfault when they're loaded in the same process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Activates the fast-path in backend.api.lifespan — skips DB/Redis connections,
# pretrain subprocess, and all heavy ML imports.  Individual fixtures inject the
# specific app.state they need after TestClient starts.
os.environ.setdefault("TESTING", "1")
# Prevent locust.__init__ from calling gevent.monkey.patch_all() if locust is
# ever imported during the test session.
os.environ.setdefault("LOCUST_SKIP_MONKEY_PATCH", "1")

# Belt-and-suspenders: stub out gevent.monkey patch functions so they become
# no-ops for the entire session.  gevent.monkey.patch_all() replaces
# threading.Lock with a gevent semaphore; without a running gevent hub,
# anyio's TestClient then deadlocks at start_blocking_portal → thread.join().
try:
    import gevent.monkey as _gm  # type: ignore[import-untyped]
    _gm.patch_all = lambda *_a, **_kw: None  # type: ignore[assignment]
    _gm.patch_thread = lambda *_a, **_kw: None  # type: ignore[assignment]
    _gm.patch_socket = lambda *_a, **_kw: None  # type: ignore[assignment]
    _gm.patch_os = lambda *_a, **_kw: None  # type: ignore[assignment]
    _gm.patch_time = lambda *_a, **_kw: None  # type: ignore[assignment]
    _gm.patch_sys = lambda *_a, **_kw: None  # type: ignore[assignment]
    del _gm
except ImportError:
    pass
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def synthetic_volume_3x3x3() -> np.ndarray:
    return np.arange(27, dtype=np.uint16).reshape(3, 3, 3)


@pytest.fixture
def synthetic_dicom_file(tmp_path: Path, synthetic_volume_3x3x3: np.ndarray) -> Path:
    pydicom_dataset = pytest.importorskip("pydicom.dataset")
    pydicom_uid = pytest.importorskip("pydicom.uid")

    Dataset = pydicom_dataset.Dataset
    FileDataset = pydicom_dataset.FileDataset
    ExplicitVRLittleEndian = pydicom_uid.ExplicitVRLittleEndian
    MRImageStorage = pydicom_uid.MRImageStorage
    generate_uid = pydicom_uid.generate_uid

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dicom_path = tmp_path / "synthetic_3x3x3.dcm"
    ds = FileDataset(str(dicom_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = "MR"
    ds.Manufacturer = "NeuroSynthQA"
    ds.MagneticFieldStrength = 3.0
    ds.SeriesDescription = "Synthetic 3x3x3"
    ds.Rows = synthetic_volume_3x3x3.shape[1]
    ds.Columns = synthetic_volume_3x3x3.shape[2]
    ds.NumberOfFrames = synthetic_volume_3x3x3.shape[0]
    ds.ImagesInAcquisition = synthetic_volume_3x3x3.shape[0]
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PatientName = "Jane^Doe"
    ds.PatientID = "PHI-12345"
    ds.BurnedInAnnotation = "NO"
    ds.PixelData = synthetic_volume_3x3x3.tobytes()
    ds.save_as(dicom_path)
    return dicom_path


@pytest.fixture
def synthetic_vcf_text() -> str:
    return "\n".join(
        [
            "##fileformat=VCFv4.2",
            "##INFO=<ID=GENE,Number=1,Type=String,Description=\"Gene\">",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
            "1\t1000\t.\tA\tG\t.\tPASS\tGENE=APOE",
            "1\t1100\t.\tC\tT\t.\tPASS\tGENE=TREM2",
        ]
    )


@pytest.fixture
def fake_iceberg() -> object:
    class _FakeIceberg:
        def __init__(self) -> None:
            self.records: dict[str, pd.DataFrame] = {}

        def append_dataframe(self, table_name: str, frame: pd.DataFrame) -> None:
            self.records[table_name] = frame.copy()

    return _FakeIceberg()


@pytest.fixture
def fake_redis():
    fakeredis = pytest.importorskip("fakeredis")
    return fakeredis.FakeStrictRedis(decode_responses=True)


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    from unittest.mock import AsyncMock, MagicMock, patch

    from backend import api as api_module

    async def _no_drain(timeout_seconds: int = 20) -> None:
        _ = timeout_seconds

    monkeypatch.setattr(api_module, "_drain_celery_queue", _no_drain)

    with (
        patch("backend.db.Database.connect", new_callable=AsyncMock),
        patch("backend.db.Database.disconnect", new_callable=AsyncMock),
        patch("backend.api.Redis") as mock_redis_cls,
        # Prevent the lifespan from spawning a pretrain subprocess — on a cold CI
        # runner, Python startup + importing torch/lightgbm alone takes 30-50s.
        patch("backend.api._manifest_valid", return_value=True),
        # Prevent model-file I/O and heavy ML lib imports during test startup.
        patch("backend.model_registry.ModelRegistry"),
        patch("backend.report_generator_v4.ClinicalReportGeneratorV4"),
    ):
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping = AsyncMock()
        mock_redis_instance.close = AsyncMock()
        mock_redis_instance.llen = AsyncMock(return_value=0)
        mock_redis_cls.from_url.return_value = mock_redis_instance

        with TestClient(api_module.app) as client:
            yield client
