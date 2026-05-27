"""Stage 02 -- TotalSegmentator wrapper.

This is the only stage that requires the deep-learning model + (ideally) a GPU.
It is intentionally isolated: every downstream stage consumes the per-structure
NIfTI masks this stage writes, so the rest of the pipeline can be developed and
validated without it.

On a machine with TotalSegmentator installed this calls the Python API; if only
the CLI is present it shells out.  If neither is available it raises a clear
error telling the user where to run it.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("ribpipe.segment")


def totalseg_available() -> str | None:
    """Return 'api', 'cli', or None depending on what's installed."""
    try:
        import totalsegmentator.python_api  # noqa: F401
        return "api"
    except Exception:
        pass
    if shutil.which("TotalSegmentator"):
        return "cli"
    return None


def pick_device(prefer: str = "auto") -> str:
    """Choose a TotalSegmentator device string generically across hardware.

    Returns one of 'gpu' (CUDA), 'mps' (Apple Silicon), or 'cpu'.  ``prefer``
    may force a choice; 'auto' detects the best available backend.
    """
    if prefer in ("gpu", "mps", "cpu"):
        return prefer
    try:
        import torch
        if torch.cuda.is_available():
            return "gpu"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def run_segmentation(ct_path: str, out_dir: str, task: str = "total",
                     fast: bool = False, device: str = "auto") -> str:
    """Run TotalSegmentator, writing one NIfTI mask per structure into out_dir.

    ``device`` is 'auto' (detect CUDA/MPS/CPU), or an explicit 'gpu'/'mps'/'cpu'.
    On Apple Silicon nnU-Net's MPS path can be unstable; if the chosen device
    errors we automatically retry on CPU so the run still completes.
    Returns out_dir.  Raises RuntimeError only if TotalSegmentator is absent.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = totalseg_available()
    dev = pick_device(device)
    log.info("TotalSegmentator: mode=%s task=%s fast=%s device=%s(req=%s)",
             mode, task, fast, dev, device)

    def _run(d):
        if mode == "api":
            import nibabel as nib
            from totalsegmentator.python_api import totalsegmentator
            totalsegmentator(nib.load(str(ct_path)), str(out_dir),
                             task=task, fast=fast, device=d)
        elif mode == "cli":
            cmd = ["TotalSegmentator", "-i", str(ct_path), "-o", str(out_dir),
                   "--task", task, "--device", d]
            if fast:
                cmd.append("--fast")
            log.info("Running: %s", " ".join(cmd))
            subprocess.run(cmd, check=True)
        else:
            raise RuntimeError("notools")

    if mode in ("api", "cli"):
        try:
            _run(dev)
        except Exception as e:
            if dev != "cpu":
                log.warning("Device '%s' failed (%s); retrying on CPU.", dev, e)
                _run("cpu")
            else:
                raise
        return str(out_dir)

    raise RuntimeError(
        "TotalSegmentator is not installed in this environment. Install it on a "
        "machine with a CUDA GPU (`pip install TotalSegmentator`) or use the "
        "SlicerTotalSegmentator extension (see slicer/slicer_pipeline.py), then "
        "re-run with --skip-seg pointing at the produced masks.")


def find_mask(seg_dir: str, label: str) -> Path | None:
    """Locate a structure mask file in a TotalSegmentator output folder."""
    seg_dir = Path(seg_dir)
    for ext in (".nii.gz", ".nii"):
        p = seg_dir / f"{label}{ext}"
        if p.exists():
            return p
    return None
