from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from neurosynth.core.config import NeuroSynthSettings
from neurosynth.dicom.processor import DICOMProcessor


@patch("neurosynth.dicom.processor._validate_single")
def test_validate_dicom_returns_result(mock_validate: MagicMock, tmp_path: Path) -> None:
    file_path = tmp_path / "scan.dcm"
    file_path.write_bytes(b"dummy")

    mock_validate.return_value = type("Validation", (), {"is_valid": True})()
    processor = DICOMProcessor(NeuroSynthSettings())

    result = processor.validate_dicom(file_path)
    assert getattr(result, "is_valid") is True
