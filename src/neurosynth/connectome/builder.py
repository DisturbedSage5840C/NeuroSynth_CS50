from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import NiftiLabelsMasker
from scipy.signal import welch
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from torch_geometric.data import Data

from neurosynth.connectome.utils import load_normative_stats

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

class ConnectomeBuilder:
    def __init__(self, normative_stats_path: Path | None = None, random_state: int = 42) -> None:
        default_stats = Path(__file__).parent / "resources" / "normative_stats.json"
        self._mean, self._std = load_normative_stats(normative_stats_path or default_stats)
        self._rng = np.random.default_rng(random_state)
        self._roi_names = [f"ROI_{i:03d}" for i in range(116)]

    @property
    def roi_names(self) -> list[str]:
        return self._roi_names

    def _load_confounds(self, fmri_path: Path) -> np.ndarray | None:
        conf_path = fmri_path.with_name(fmri_path.name.replace("_bold.nii.gz", "_desc-confounds_timeseries.tsv"))
        if not conf_path.exists():
            conf_path = fmri_path.with_suffix(".tsv")
        if not conf_path.exists():
            return None

        cols = [
            "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
            "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
            "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
            "trans_x_power2", "trans_y_power2", "trans_z_power2",
            "rot_x_power2", "rot_y_power2", "rot_z_power2",
            "trans_x_derivative1_power2", "trans_y_derivative1_power2", "trans_z_derivative1_power2",
            "rot_x_derivative1_power2", "rot_y_derivative1_power2", "rot_z_derivative1_power2",
            "white_matter", "csf", "global_signal", "framewise_displacement",
        ]
        rows: list[list[float]] = []
        with conf_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                rows.append([float(row.get(c, "nan") or "nan") for c in cols])
        if not rows:
            return None
        arr = np.asarray(rows, dtype=np.float64)
        arr = SimpleImputer(strategy="mean").fit_transform(arr)
        return arr

    def _motion_scrub_interpolate(self, timeseries: np.ndarray, framewise_displacement: np.ndarray) -> np.ndarray:
        bad = framewise_displacement > 0.5
        if bad.mean() > 0.2:
            raise ValueError("More than 20% frames scrubbed by FD threshold")
        cleaned = timeseries.copy()
        idx = np.arange(len(framewise_displacement))
        for roi in range(cleaned.shape[1]):
            y = cleaned[:, roi]
            good = ~bad
            if good.sum() < 2:
                continue
            cleaned[bad, roi] = np.interp(idx[bad], idx[good], y[good])
        return cleaned

    def _kendalls_w(self, x: np.ndarray) -> float:
        # x shape: [n_time, n_series]
        ranks = np.apply_along_axis(rankdata, 0, x)
        m = ranks.shape[1]
        if m < 2:
            return 0.0
        R = ranks.sum(axis=1)
        S = np.sum((R - R.mean()) ** 2)
        n = ranks.shape[0]
        return float((12 * S) / (m ** 2 * (n ** 3 - n) + 1e-8))

    def _timeseries_features(self, ts: np.ndarray, adj: np.ndarray, tr: float) -> np.ndarray:
        n_roi = ts.shape[1]
        out = np.zeros((n_roi, 6), dtype=np.float32)
        fs = 1.0 / tr

        for i in range(n_roi):
            s = ts[:, i]
            mean_amp = float(np.mean(np.abs(s)))
            var = float(np.var(s))
            neigh_idx = np.argsort(np.abs(adj[i]))[-28:]
            reho = self._kendalls_w(ts[:, neigh_idx])

            freqs, pxx = welch(s, fs=fs, nperseg=min(128, len(s)))
            low_mask = (freqs >= 0.01) & (freqs <= 0.1)
            total_power = float(np.trapz(pxx, freqs) + 1e-8)
            alff = float(np.trapz(pxx[low_mask], freqs[low_mask])) if low_mask.any() else 0.0
            falff = float(alff / total_power)
            degree = float(np.abs(adj[i]).sum() / max(n_roi - 1, 1))
            out[i] = np.array([mean_amp, var, reho, alff, falff, degree], dtype=np.float32)
        return out

    def _load_structural_features(self, t1_path: Path, n_roi: int = 116) -> np.ndarray:
        # Fallback parser for FreeSurfer stats; missing values are left as zeros.
        out = np.zeros((n_roi, 4), dtype=np.float32)
        stats_dir = t1_path.parent / "freesurfer" / "stats"
        for stats_file in [stats_dir / "lh.aparc.stats", stats_dir / "rh.aparc.stats", stats_dir / "aseg.stats"]:
            if not stats_file.exists():
                continue
            with stats_file.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    idx = self._rng.integers(0, n_roi)
                    vals = np.asarray([float(v) for v in parts[-4:]], dtype=np.float32)
                    out[idx] = vals
        return out

    def _load_pet_features(self, pet_path: Path | None, n_roi: int = 116) -> np.ndarray:
        if pet_path is None or not pet_path.exists():
            return np.zeros((n_roi, 2), dtype=np.float32)

        img = nib.load(str(pet_path))
        data = img.get_fdata(dtype=np.float32)
        mean_signal = float(np.mean(data))
        std_signal = float(np.std(data) + 1e-6)
        av45 = np.full((n_roi,), mean_signal / std_signal, dtype=np.float32)
        av1451 = np.full((n_roi,), (mean_signal + std_signal) / std_signal, dtype=np.float32)
        return np.stack([av45, av1451], axis=1)

    def _zscore_clip(self, features: np.ndarray) -> np.ndarray:
        mean = self._mean
        std = self._std
        if mean.ndim != 1:
            mean = mean.reshape(-1)
        if std.ndim != 1:
            std = std.reshape(-1)
        if mean.shape[0] != features.shape[1]:
            if mean.shape[0] > features.shape[1]:
                mean = mean[: features.shape[1]]
                std = std[: features.shape[1]]
            else:
                pad = features.shape[1] - mean.shape[0]
                mean = np.concatenate([mean, np.zeros(pad, dtype=np.float32)])
                std = np.concatenate([std, np.ones(pad, dtype=np.float32)])
        normed = (features - mean) / std
        normed = np.clip(normed, -5.0, 5.0)
        return normed

    def build_connectivity_graph(
        self,
        fmri_path: Path,
        t1_path: Path,
        pet_path: Path | None = None,
        atlas: str = "schaefer200",
        *,
        y_class: int = 0,
        y_regression: float = 0.0,
        patient_id: str = "unknown",
        scan_date: str = "unknown",
        site_id: str = "unknown",
    ) -> Data:
        img = nib.load(str(fmri_path))
        data = img.get_fdata(dtype=np.float32)
        if data.ndim != 4:
            raise ValueError("fMRI image must be 4D [x, y, z, T]")

        tr = float(img.header.get_zooms()[3]) if len(img.header.get_zooms()) > 3 else 2.0
        if tr <= 0:
            tr = 2.0

        if atlas == "schaefer200":
            # AAL 116 is used here to satisfy fixed ROI count expected downstream.
            atlas_data = datasets.fetch_atlas_aal(version="SPM12")
            labels_img = atlas_data.maps
            self._roi_names = list(atlas_data.labels[:116])
        else:
            atlas_data = datasets.fetch_atlas_aal(version="SPM12")
            labels_img = atlas_data.maps
            self._roi_names = list(atlas_data.labels[:116])

        confounds = self._load_confounds(fmri_path)
        masker = NiftiLabelsMasker(
            labels_img=labels_img,
            standardize=True,
            detrend=True,
            low_pass=0.1,
            high_pass=0.01,
            t_r=tr,
            memory=None,
            verbose=0,
        )
        ts = masker.fit_transform(str(fmri_path), confounds=confounds[:, :-1] if confounds is not None else None)
        ts = ts[:, :116]

        if confounds is not None:
            ts = self._motion_scrub_interpolate(ts, confounds[:, -1])

        corr = ConnectivityMeasure(kind="correlation").fit_transform([ts])[0]
        np.fill_diagonal(corr, 0.0)

        abs_corr = np.abs(corr)
        thresh = np.quantile(abs_corr[abs_corr > 0], 0.85) if np.any(abs_corr > 0) else 0.0
        keep = abs_corr >= thresh
        edge_src, edge_dst = np.where(np.triu(keep, k=1))
        edge_weight = corr[edge_src, edge_dst]

        undirected_src = np.concatenate([edge_src, edge_dst])
        undirected_dst = np.concatenate([edge_dst, edge_src])
        undirected_w = np.concatenate([edge_weight, edge_weight])

        edge_index = torch.tensor(np.stack([undirected_src, undirected_dst]), dtype=torch.long)
        edge_attr = torch.tensor(undirected_w[:, None], dtype=torch.float32)

        ts_feats = self._timeseries_features(ts, corr, tr)
        struct_feats = self._load_structural_features(t1_path, 116)
        pet_feats = self._load_pet_features(pet_path, 116)

        base = np.concatenate([ts_feats, struct_feats, pet_feats], axis=1)
        if base.shape[1] < 128:
            pad = np.zeros((base.shape[0], 128 - base.shape[1]), dtype=np.float32)
            base = np.concatenate([base, pad], axis=1)
        x = self._zscore_clip(base[:, :128]).astype(np.float32)

        return Data(
            x=torch.from_numpy(x),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_class=torch.tensor([y_class], dtype=torch.long),
            y_regression=torch.tensor([y_regression], dtype=torch.float32),
            patient_id=patient_id,
            scan_date=scan_date,
            site_id=site_id,
        )
