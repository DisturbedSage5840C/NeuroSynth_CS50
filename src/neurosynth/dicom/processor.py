# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import AsyncGenerator, Any
from uuid import UUID

import boto3
import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import Dataset
from tenacity import retry, stop_after_attempt, wait_exponential

from neurosynth.core.config import NeuroSynthSettings
from neurosynth.core.exceptions import DicomValidationError
from neurosynth.core.logging import get_logger
from neurosynth.core.models import DicomResult, DicomValidationResult

SAFE_HARBOR_TAGS = [
    (0x0010, 0x0010),
    (0x0010, 0x0020),
    (0x0010, 0x0030),
    (0x0010, 0x0040),
    (0x0008, 0x0080),
    (0x0008, 0x0090),
    (0x0008, 0x1010),
]


def _validate_single(path: str) -> DicomValidationResult:
    ds = pydicom.dcmread(path, stop_before_pixels=False, force=True)
    has_private_tags = any(getattr(elem.tag, "is_private", False) for elem in ds.iterall())
    return DicomValidationResult(
        is_valid=True,
        modality=getattr(ds, "Modality", None),
        manufacturer=getattr(ds, "Manufacturer", None),
        field_strength=float(getattr(ds, "MagneticFieldStrength", 0) or 0) or None,
        series_description=getattr(ds, "SeriesDescription", None),
        pixel_spacing=[float(x) for x in getattr(ds, "PixelSpacing", [])],
        slice_thickness=float(getattr(ds, "SliceThickness", 0) or 0) or None,
        n_slices=int(getattr(ds, "ImagesInAcquisition", 0) or 0) or None,
        has_private_tags=has_private_tags,
    )


class DICOMProcessor:
    def __init__(self, settings: NeuroSynthSettings) -> None:
        self._settings = settings
        self._logger = get_logger(__name__)
        self._s3 = boto3.client(
            "s3",
            endpoint_url=self._settings.minio_endpoint,
            aws_access_key_id=self._settings.minio_access_key,
            aws_secret_access_key=self._settings.minio_secret_key,
            region_name=self._settings.minio_region,
        )

    def validate_dicom(self, path: Path) -> DicomValidationResult:
        try:
            return _validate_single(str(path))
        except Exception as exc:
            raise DicomValidationError(f"Validation failed for {path}: {exc}") from exc

    def _detect_burned_in_regions(self, ds: Dataset) -> list[tuple[int, int, int, int]]:
        # Fallback detector: inspect image corners for high-intensity overlays
        # often used by scanner overlays when BurnedInAnnotation=YES.
        if not hasattr(ds, "pixel_array"):
            return []

        arr = ds.pixel_array
        if arr.ndim > 2:
            arr = arr[..., 0]
        arr = arr.astype(np.float32)

        h, w = arr.shape[:2]
        hh = max(8, h // 10)
        ww = max(8, w // 10)
        corners = [
            (0, 0, ww, hh),
            (w - ww, 0, w, hh),
            (0, h - hh, ww, h),
            (w - ww, h - hh, w, h),
        ]

        p995 = float(np.percentile(arr, 99.5))
        regions: list[tuple[int, int, int, int]] = []
        for x0, y0, x1, y1 in corners:
            patch = arr[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            hot_ratio = float((patch >= p995).mean())
            if hot_ratio > 0.01:
                regions.append((x0, y0, x1, y1))
        return regions

    def anonymize_dicom(self, path: Path, patient_uuid: UUID) -> Path:
        ds = pydicom.dcmread(path, force=True)
        for tag in SAFE_HARBOR_TAGS:
            if tag in ds:
                ds[tag].value = ""
        ds.PatientID = str(patient_uuid)

        if getattr(ds, "BurnedInAnnotation", "NO") == "YES" and hasattr(ds, "pixel_array"):
            arr = ds.pixel_array.copy()
            for x0, y0, x1, y1 in self._detect_burned_in_regions(ds):
                arr[y0:y1, x0:x1] = 0
            ds.PixelData = arr.tobytes()

        out_path = path.with_name(f"anon_{path.name}")
        ds.save_as(out_path)
        return out_path

    def extract_metadata(self, path: Path) -> dict[str, Any]:
        ds = pydicom.dcmread(path, force=True, stop_before_pixels=True)
        keys = [
            "Modality",
            "Manufacturer",
            "SeriesDescription",
            "StudyDate",
            "RepetitionTime",
            "EchoTime",
            "FlipAngle",
            "MagneticFieldStrength",
        ]
        return {k: getattr(ds, k, None) for k in keys}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    def compress_and_store(self, path: Path, s3_bucket: str) -> str:
        image = sitk.ReadImage(str(path))
        out_path = path.with_suffix(".jp2")
        sitk.WriteImage(image, str(out_path), useCompression=True, compressionLevel=0)

        object_key = out_path.name
        self._s3.upload_file(str(out_path), s3_bucket, object_key)
        return f"s3://{s3_bucket}/{object_key}"

    async def batch_process(self, dicom_dir: Path, n_workers: int = 16) -> AsyncGenerator[DicomResult, None]:
        files = [p for p in dicom_dir.rglob("*") if p.is_file()]
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            tasks = [loop.run_in_executor(pool, _validate_single, str(path)) for path in files]
            for path, task in zip(files, tasks):
                try:
                    validation = await task
                    yield DicomResult(source_path=path, patient_uuid=UUID(int=0), validation=validation)
                except Exception as exc:
                    yield DicomResult(
                        source_path=path,
                        patient_uuid=UUID(int=0),
                        validation=DicomValidationResult(is_valid=False),
                        error=str(exc),
                    )
