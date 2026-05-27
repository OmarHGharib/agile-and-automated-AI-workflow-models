"""Central configuration: thresholds, parameters, anatomical conventions.

Every tunable constant in the pipeline lives here so a reviewer can audit the
clinical/biomechanical assumptions in one place.  Distance thresholds follow
the reasoning documented in the project approach (chest_ct_pipeline_approach):
TotalSegmentator rib masks include the rib head/neck, so the costo-vertebral
joint sits 5-10 mm from a vertebral *centroid*; the thresholds below allow for
that plus segmentation noise.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # --- Stage 01: conversion / QC -------------------------------------
    # Expected voxel-spacing envelope (mm). Outside -> QC warning.
    spacing_min: Tuple[float, float, float] = (0.3, 0.3, 0.3)
    spacing_max: Tuple[float, float, float] = (1.5, 1.5, 6.0)
    # Bone window (HU) used for QC sanity checks and for the threshold-based
    # reconstruction fallback when no labelled masks are available.
    bone_hu_threshold: float = 200.0

    # --- Stage 02: segmentation ---------------------------------------
    seg_task: str = "total"
    seg_fast: bool = False  # set True for CPU-only (3 mm model)

    # --- Stage 04: centerline -----------------------------------------
    centerline_n_points: int = 500          # RibSeg v2 resampling convention
    centerline_smooth_sigma: float = 1.5     # voxels
    # kimimaro TEASAR parameters (used when kimimaro is available)
    teasar_params: dict = field(default_factory=lambda: {
        "scale": 4, "const": 500, "pdrf_scale": 100000,
        "pdrf_exponent": 4, "soma_acceptance_threshold": 3500,
    })

    # --- Stage 06: mesh ------------------------------------------------
    mesh_smooth_iter: int = 30
    marching_level: float = 0.5

    # --- Costal cartilage bridges (rib -> sternum connection) ----------
    add_cartilage_bridge: bool = True
    cartilage_radius_mm: float = 5.0      # tube radius at the rib end
    cartilage_max_gap_mm: float = 90.0    # don't bridge implausibly large gaps

    # --- Costovertebral bone RECOVERY (real CT bone, not synthetic) ----
    # Recover the rib head/neck that TotalSegmentator truncates, by region-
    # growing real high-HU bone from the rib toward (but not into) the vertebra.
    # This closes the posterior gap with actual image data; the residual is the
    # true joint space. Fracture gaps are NOT crossed.
    recover_costovertebral_bone: bool = True
    recover_bone_threshold: float = 180.0     # HU; cortical bone is well above
    recover_max_grow_mm: float = 45.0         # cap on recovered rib-head reach

    # --- Costovertebral joints (rib head -> thoracic vertebra) ---------
    # OFF by default: the rib head and vertebra are both real segmentations and
    # already articulate; a synthetic connector tends to look fabricated. The
    # ribpipe.joints code remains available (set True) for biomechanics/FEA use.
    add_costovertebral_link: bool = False
    costovertebral_radius_mm: float = 4.5     # connector tube radius
    costovertebral_max_gap_mm: float = 35.0   # rib head sits close to the facet

    # --- Stage 07: mechanics thresholds (mm) --------------------------
    # Posterior centerline endpoint -> corresponding vertebra centroid.
    costo_vertebral_mm: float = 12.0
    # Anterior endpoint (true ribs 1-7) -> sternum/costal-cartilage surface.
    costo_sternal_mm: float = 25.0
    # Adjacent thoracic vertebra surfaces (disc space).
    intervertebral_mm: float = 10.0
    # Floating ribs (11-12): anterior end must be FAR from sternum.
    floating_min_mm: float = 30.0
    # Anatomical rib->sternum attachment map (true / false / floating).
    true_ribs: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    false_ribs: Tuple[int, ...] = (8, 9, 10)
    floating_ribs: Tuple[int, ...] = (11, 12)

    # --- Stage 08: report ---------------------------------------------
    write_excel: bool = True
    write_csv: bool = True
    write_json: bool = True


DEFAULT = Config()
