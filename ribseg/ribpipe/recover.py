"""Recover the real rib head/neck bone from the CT (no fabrication).

The costovertebral gap in a TotalSegmentator model is usually because the *rib*
mask is truncated -- it stops before the rib head/neck that reaches the
vertebra.  But that bone IS in the CT (high HU); it just wasn't labelled.  So
instead of drawing a synthetic strut, we recover the real voxels:

    bone = CT > threshold
    grow from the rib mask through connected bone, WITHOUT entering the
    vertebra, limited to a few cm of the rib head -> add to the rib.

The grow stops at the costovertebral joint space (radiolucent articular
cartilage), so the rib ends up reaching the vertebra with only the true joint
gap remaining.  Everything added is real image bone, and a genuine fracture
gap is NOT crossed (the fragments stay as separate components unless real bone
connects them).
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

from . import RIB_LABELS, VERTEBRA_LABELS
from .segment import find_mask

log = logging.getLogger("ribpipe.recover")


def recover_rib_head_bone(ct: np.ndarray, rib_mask: np.ndarray,
                          vert_union: np.ndarray, spacing: np.ndarray,
                          threshold: float = 180.0,
                          max_grow_mm: float = 45.0) -> np.ndarray:
    """Return rib_mask augmented with real CT bone reaching toward the spine."""
    bone = ct > threshold
    allowed = bone & ~vert_union          # don't grow into the vertebra
    allowed |= rib_mask                    # ensure the rib seeds are included
    lab, n = ndimage.label(allowed)
    if n == 0:
        return rib_mask
    seed_labels = np.unique(lab[rib_mask])
    seed_labels = seed_labels[seed_labels != 0]
    if len(seed_labels) == 0:
        return rib_mask
    grown = np.isin(lab, seed_labels)
    # limit how far from the original rib we are willing to recover
    dist = ndimage.distance_transform_edt(~rib_mask, sampling=spacing)
    grown &= dist <= max_grow_mm
    return rib_mask | grown


def augment_segmentation(ct_path: str, seg_dir: str, out_dir: str,
                         threshold: float = 180.0, max_grow_mm: float = 45.0
                         ) -> str:
    """Write a recovered copy of ``seg_dir`` with rib masks extended to the spine.

    Non-rib structures are copied unchanged.  Returns ``out_dir``.
    """
    seg_dir, out_dir = Path(seg_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # copy everything first (vertebrae, sternum, cartilage, etc. unchanged)
    for f in seg_dir.glob("*.nii*"):
        if not (out_dir / f.name).exists():
            shutil.copy2(f, out_dir / f.name)

    img = nib.load(str(ct_path))
    ct = np.asarray(img.dataobj, dtype=np.float32)
    spacing = np.array(img.header.get_zooms()[:3], float)

    # vertebra union (so grow stops at the spine)
    vert_union = None
    for vl in VERTEBRA_LABELS:
        p = find_mask(seg_dir, vl)
        if p is None:
            continue
        m = np.asarray(nib.load(str(p)).dataobj) > 0.5
        vert_union = m if vert_union is None else (vert_union | m)
    if vert_union is None:
        log.warning("No vertebrae -- skipping rib-head recovery.")
        return str(out_dir)
    # dilate the spine slightly so we truly stop at its surface
    vert_union = ndimage.binary_dilation(vert_union, iterations=1)

    n_aug = 0
    for rl in RIB_LABELS:
        p = find_mask(seg_dir, rl)
        if p is None:
            continue
        rib_img = nib.load(str(p))
        rib = np.asarray(rib_img.dataobj) > 0.5
        if rib.sum() == 0:
            continue
        aug = recover_rib_head_bone(ct, rib, vert_union, spacing,
                                    threshold, max_grow_mm)
        added = int(aug.sum() - rib.sum())
        if added > 0:
            n_aug += 1
        nib.save(nib.Nifti1Image(aug.astype(np.uint8), rib_img.affine),
                 out_dir / Path(p).name)
    log.info("Rib-head recovery: extended %d/%d ribs with real CT bone.",
             n_aug, len(RIB_LABELS))
    return str(out_dir)
