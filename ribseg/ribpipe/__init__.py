"""
ribpipe -- Automated rib morphometry, 3D bony-thorax reconstruction, and
biomechanical verification from chest CT.

Implements the methodology described in:
  * Wasserthal et al., "TotalSegmentator", Radiology: AI (2023)  -- segmentation
  * Jin et al., "RibSeg v2", IEEE TMI (2023)                     -- centerlines/labeling

The package is organised as eight sequential stages (stage_00 .. stage_08)
orchestrated by ``pipeline.py``.  All measurement code lives in
``ribpipe.geometry`` and is fully decoupled from the deep-learning
segmentation step so it can be unit-tested against synthetic phantoms with
known ground truth (see ``validation/``).
"""

__version__ = "1.0.0"

# Canonical structure-name lists used throughout the pipeline.  These match the
# label names emitted by TotalSegmentator's ``total`` task.
RIB_SIDES = ("left", "right")
RIB_NUMBERS = tuple(range(1, 13))  # 1..12

RIB_LABELS = tuple(
    f"rib_{side}_{n}" for side in RIB_SIDES for n in RIB_NUMBERS
)  # 24 individual ribs

VERTEBRA_LABELS = tuple(f"vertebrae_T{n}" for n in range(1, 13))  # T1..T12

OTHER_BONE_LABELS = (
    "sternum",
    "costal_cartilages",
    "clavicula_left",
    "clavicula_right",
    "scapula_left",
    "scapula_right",
)

ALL_THORAX_LABELS = RIB_LABELS + VERTEBRA_LABELS + OTHER_BONE_LABELS
