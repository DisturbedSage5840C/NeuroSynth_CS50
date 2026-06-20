from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from nilearn.input_data import NiftiLabelsMasker

from neurosynth.core.logging import get_logger
from neurosynth.data.iceberg_catalog import IcebergDomainCatalog

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

@dataclass
class ImagingQCResult:
    qc_pass: bool
    flags: list[str]
    voxel_size_mm: tuple[float, float, float]
    orientation: str
    field_strength_t: float | None


class DICOMIngestionPipeline:
    def __init__(self, iceberg: IcebergDomainCatalog) -> None:
        self.iceberg = iceberg
        self.log = get_logger(__name__)

    def ingest_series(
        self,
        dicom_dir: str | Path,
        patient_id: str,
        patient_cohort: str,
        mni_template_path: str | Path,
        atlas_labels_path: str | Path,
        output_dir: str | Path,
        field_strength_bounds: tuple[float, float] = (1.0, 7.0),
    ) -> dict[str, Any]:
        files = sorted(Path(dicom_dir).glob("*.dcm"))
        if not files:
            raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

        headers = [pydicom.dcmread(str(path), stop_before_pixels=True) for path in files]
        qc = self._run_qc(headers, field_strength_bounds=field_strength_bounds)

        image = sitk.ReadImage([str(p) for p in files])
        registered = self._register_to_mni(image, str(mni_template_path))
        nifti_img = self._to_nifti(registered)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        session_id = str(uuid4())
        conn_id = str(uuid4())
        nifti_path = output_dir / f"{session_id}_registered.nii.gz"
        matrix_path = output_dir / f"{conn_id}_connectivity.parquet"

        nib.save(nifti_img, str(nifti_path))
        conn = self._connectivity_matrix(nifti_img, str(atlas_labels_path))
        conn.to_parquet(matrix_path, index=False)

        imaging_row = pd.DataFrame(
            [
                {
                    "imaging_session_id": session_id,
                    "patient_id": patient_id,
                    "patient_cohort": patient_cohort,
                    "ingestion_date": date.today(),
                    "series_uid": str(getattr(headers[0], "SeriesInstanceUID", "unknown")),
                    "modality": str(getattr(headers[0], "Modality", "MR")),
                    "field_strength_t": qc.field_strength_t,
                    "voxel_size_mm": str(qc.voxel_size_mm),
                    "orientation": qc.orientation,
                    "qc_pass": qc.qc_pass,
                    "qc_flags": qc.flags,
                    "registered_nifti_uri": str(nifti_path),
                }
            ]
        )
        self.iceberg.append_dataframe("imaging_sessions", imaging_row)

        conn_row = pd.DataFrame(
            [
                {
                    "connectivity_id": conn_id,
                    "imaging_session_id": session_id,
                    "patient_id": patient_id,
                    "patient_cohort": patient_cohort,
                    "ingestion_date": date.today(),
                    "atlas_name": Path(atlas_labels_path).stem,
                    "n_regions": int(conn[["region_i", "region_j"]].max().max() + 1 if not conn.empty else 0),
                    "matrix_uri": str(matrix_path),
                    "mean_connectivity": float(conn["corr"].mean()) if not conn.empty else 0.0,
                }
            ]
        )
        self.iceberg.append_dataframe("connectivity_matrices", conn_row)

        self.log.info(
            "imaging.ingested",
            patient_id=patient_id,
            session_id=session_id,
            qc_pass=qc.qc_pass,
            qc_flags=qc.flags,
            matrix_uri=str(matrix_path),
        )

        return {
            "imaging_session_id": session_id,
            "connectivity_id": conn_id,
            "qc": qc,
            "registered_nifti_uri": str(nifti_path),
            "connectivity_uri": str(matrix_path),
        }

    def _run_qc(self, headers: list[pydicom.dataset.FileDataset], field_strength_bounds: tuple[float, float]) -> ImagingQCResult:
        first = headers[0]
        spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
        thickness = float(getattr(first, "SliceThickness", 1.0))
        voxel_size_mm = (float(spacing[0]), float(spacing[1]), thickness)

        orientation_arr = getattr(first, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
        orientation = "".join(str(int(x)) for x in orientation_arr[:6])
        field_strength = getattr(first, "MagneticFieldStrength", None)
        field_strength_f = float(field_strength) if field_strength is not None else None

        flags: list[str] = []
        if any(v <= 0 or v > 4.0 for v in voxel_size_mm):
            flags.append("voxel_size_outlier")
        if orientation in {"100010", "010001"}:
            flags.append("orientation_unexpected")
        if field_strength_f is not None and not (field_strength_bounds[0] <= field_strength_f <= field_strength_bounds[1]):
            flags.append("field_strength_outlier")

        return ImagingQCResult(
            qc_pass=len(flags) == 0,
            flags=flags,
            voxel_size_mm=voxel_size_mm,
            orientation=orientation,
            field_strength_t=field_strength_f,
        )

    def _register_to_mni(self, image: sitk.Image, mni_template_path: str) -> sitk.Image:
        fixed = sitk.ReadImage(mni_template_path, sitk.sitkFloat32)
        moving = sitk.Cast(image, sitk.sitkFloat32)

        initial_tx = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=60)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetInitialTransform(initial_tx, inPlace=False)

        transform = reg.Execute(fixed, moving)
        return sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, moving.GetPixelID())

    @staticmethod
    def _to_nifti(image: sitk.Image) -> nib.Nifti1Image:
        arr = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()
        affine = np.diag([float(spacing[0]), float(spacing[1]), float(spacing[2]), 1.0])
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = np.transpose(arr, (2, 1, 0))
        return nib.Nifti1Image(arr, affine)

    def _connectivity_matrix(self, nifti_img: nib.Nifti1Image, atlas_labels_path: str) -> pd.DataFrame:
        masker = NiftiLabelsMasker(labels_img=atlas_labels_path, standardize=True)
        series = masker.fit_transform(nifti_img)
        series = np.asarray(series)

        if series.ndim == 1:
            series = series.reshape(1, -1)
        if series.shape[0] < 2:
            corr = np.zeros((series.shape[1], series.shape[1]), dtype=float)
        else:
            corr = np.corrcoef(series.T)

        rows = []
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[1]):
                rows.append({"region_i": i, "region_j": j, "corr": float(corr[i, j])})
        return pd.DataFrame(rows)
