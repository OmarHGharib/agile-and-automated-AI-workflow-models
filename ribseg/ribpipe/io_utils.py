"""I/O helpers: DICOM->NIfTI conversion, NIfTI + mask loading, QC."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import nibabel as nib

log = logging.getLogger("ribpipe.io")


def convert_dicom_to_nifti(dicom_dir: str, out_path: str) -> str:
    """Convert a DICOM series folder to a single .nii.gz using SimpleITK."""
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not names:
        raise FileNotFoundError(f"No DICOM series found in {dicom_dir}")
    reader.SetFileNames(names)
    image = reader.Execute()
    image = sitk.DICOMOrient(image, "LPS")
    sitk.WriteImage(image, str(out_path))
    log.info("Converted %d DICOM slices -> %s", len(names), out_path)
    return str(out_path)


def load_volume(path: str):
    """Load a NIfTI volume. Returns (data float32, affine, nib image)."""
    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float32), img.affine, img


def load_mask(path: str):
    """Load a binary mask NIfTI. Returns (bool array, affine)."""
    img = nib.load(str(path))
    return np.asarray(img.dataobj) > 0.5, img.affine


def qc_volume(data: np.ndarray, affine: np.ndarray, cfg) -> dict:
    """Basic quality-control checks on an input CT volume."""
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    warnings = []
    for k, s in enumerate(spacing):
        if not (cfg.spacing_min[k] <= s <= cfg.spacing_max[k]):
            warnings.append(f"axis {k} spacing {s:.2f} mm outside "
                            f"[{cfg.spacing_min[k]},{cfg.spacing_max[k]}]")
    hu_lo, hu_hi = float(np.percentile(data, 1)), float(np.percentile(data, 99))
    bone_frac = float((data > cfg.bone_hu_threshold).mean())
    if bone_frac < 1e-4:
        warnings.append("almost no bone-HU voxels -- check intensity units")
    return {"spacing_mm": [round(float(x), 3) for x in spacing],
            "shape": list(map(int, data.shape)),
            "hu_p1": round(hu_lo, 1), "hu_p99": round(hu_hi, 1),
            "bone_fraction": round(bone_frac, 5), "warnings": warnings}


def bone_mask(data: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold-based bony mask used for the no-label reconstruction fallback."""
    return data > threshold
