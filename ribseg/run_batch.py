#!/usr/bin/env python3
"""Batch-run the pipeline over a folder of NIfTI volumes (e.g. RibFrac test set).

Example
-------
  python run_batch.py --in-dir /path/to/ribfrac-test-images \
                      --out runs_ribfrac --pattern "*-image.nii.gz" --limit 5

Each case gets its own subfolder under --out. A combined ``batch_summary.csv``
aggregates per-case status so you can triage failures across the cohort.
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd

from ribpipe.config import Config
from ribpipe import io_utils, segment, stages, report, meshing


def process_case(nifti: Path, out: Path, cfg: Config, skip_seg: bool,
                 seg_dir: Path | None, fast: bool, device: str = "auto") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    case_id = nifti.name.replace(".nii.gz", "").replace(".nii", "")
    sdir = seg_dir if seg_dir else out / "segmentations"
    if not skip_seg:
        segment.run_segmentation(str(nifti), sdir, task=cfg.seg_task, fast=fast,
                                 device=device)
    data, affine, _ = io_utils.load_volume(str(nifti))
    qc = io_utils.qc_volume(data, affine, cfg)
    frame, vc = stages.stage_03_spine(sdir, cfg)
    cls = stages.stage_04_centerlines(sdir, frame, vc, cfg)
    morph = stages.stage_05_morphometry(sdir, frame, cls, cfg)
    meshes = stages.stage_06_mesh(sdir, out / "meshes", cfg)
    mech = stages.stage_07_mechanics(cls, vc, meshes, cfg)
    report.write_reports(morph, mech, out / "reports", cfg, case_id=case_id, qc=qc)
    n_fail = sum(1 for r in mech if r.get("status") == "FAIL")
    return {"case_id": case_id, "n_ribs": len(morph),
            "n_checks": len(mech), "n_fail": n_fail, "status": "OK"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pattern", default="*-image.nii.gz")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--skip-seg", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--device", default="auto",
                    help="auto | gpu | mps | cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    cfg = Config()
    files = sorted(Path(args.in_dir).glob(args.pattern))
    if args.limit:
        files = files[:args.limit]
    rows = []
    for f in files:
        case_out = Path(args.out) / f.name.replace(".nii.gz", "").replace(".nii", "")
        try:
            rows.append(process_case(f, case_out, cfg, args.skip_seg, None,
                                     args.fast, args.device))
        except Exception as e:                       # keep the batch going
            logging.error("FAILED %s: %s", f.name, e)
            logging.debug(traceback.format_exc())
            rows.append({"case_id": f.name, "status": f"ERROR: {e}"})
    df = pd.DataFrame(rows)
    outp = Path(args.out); outp.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp / "batch_summary.csv", index=False)
    logging.info("Batch done: %d cases -> %s", len(rows), outp / "batch_summary.csv")


if __name__ == "__main__":
    main()
