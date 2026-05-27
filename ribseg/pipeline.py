#!/usr/bin/env python3
"""End-to-end chest-CT rib analysis pipeline.

Stages
------
  00 preflight        check dependencies
  01 convert          DICOM -> NIfTI (skipped if input is already .nii/.nii.gz)
  02 segment          TotalSegmentator -> per-structure masks   (needs GPU)
  03 spine            PCA spinal axis + anatomical frame
  04 centerlines      TEASAR/kimimaro rib centerlines
  05 morphometry      length / max-min width / max-min height per rib
  06 mesh             marching-cubes STL meshes of the bony thorax
  07 mechanics        costo-vertebral / costo-sternal / inter-vertebral checks
  08 report           CSV + Excel + JSON + 3D render

Usage
-----
  # from a DICOM folder
  python pipeline.py --dicom /path/to/dicom --out runs/patientA

  # from an existing NIfTI volume
  python pipeline.py --nifti ct.nii.gz --out runs/patientA

  # re-use existing TotalSegmentator masks (skip the GPU step)
  python pipeline.py --nifti ct.nii.gz --seg-dir existing_masks --skip-seg \
                     --out runs/patientA
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from ribpipe.config import Config
from ribpipe import (io_utils, segment, stages, report, meshing, cartilage,
                     joints, raster, recover)


def setup_logging(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(out / "pipeline.log")])
    return logging.getLogger("ribpipe.pipeline")


def run(args) -> dict:
    out = Path(args.out)
    log = setup_logging(out)
    cfg = Config(seg_fast=args.fast)
    t0 = time.time()

    # --- 00 preflight ---
    mode = segment.totalseg_available()
    log.info("Preflight: TotalSegmentator=%s, kimimaro=%s",
             mode or "MISSING", stages.geo._HAS_KIMIMARO)

    # --- 01 convert ---
    if args.dicom:
        ct_path = io_utils.convert_dicom_to_nifti(args.dicom, out / "ct.nii.gz")
    else:
        ct_path = args.nifti
    data, affine, _img = io_utils.load_volume(ct_path)
    qc = io_utils.qc_volume(data, affine, cfg)
    (out / "qc.json").write_text(json.dumps(qc, indent=2))
    log.info("QC: %s", qc)

    # --- 02 segment ---
    seg_dir = Path(args.seg_dir) if args.seg_dir else out / "segmentations"
    if not args.skip_seg:
        segment.run_segmentation(ct_path, seg_dir, task=cfg.seg_task,
                                 fast=args.fast, device=args.device)
    else:
        log.info("Skipping segmentation; using masks in %s", seg_dir)

    # --- recover real rib-head bone from the CT (closes the costovertebral
    #     gap with actual image voxels, not a synthetic strut) ---
    if cfg.recover_costovertebral_bone:
        seg_dir = recover.augment_segmentation(
            ct_path, str(seg_dir), str(out / "segmentations_recovered"),
            threshold=cfg.recover_bone_threshold,
            max_grow_mm=cfg.recover_max_grow_mm)

    # --- coverage diagnostics (which structures are actually in the model) ---
    cov = stages.coverage_report(seg_dir)
    (out / "coverage.json").write_text(json.dumps(cov, indent=2))

    # --- 03-07 ---
    frame, vert_centroids = stages.stage_03_spine(seg_dir, cfg)
    centerlines = stages.stage_04_centerlines(seg_dir, frame, vert_centroids, cfg)
    morph_rows = stages.stage_05_morphometry(seg_dir, frame, centerlines, cfg)
    meshes = stages.stage_06_mesh(seg_dir, out / "meshes", cfg)

    # Costal-cartilage bridges: connect true ribs (and false ribs) to the
    # sternum so the model is anatomically continuous (CT bone alone shows a gap
    # because cartilage is radiolucent).
    if cfg.add_cartilage_bridge:
        sv = meshes.get("sternum", (None, None))[0]
        cvv = meshes.get("costal_cartilages", (None, None))[0]
        bridges, cart_report = cartilage.build_bridges(
            centerlines, sv, cvv, cfg, vert_centroids=vert_centroids)
        for (side, n), (bv, bf) in bridges.items():
            meshes[f"cartilage_bridge_{side}_{n}"] = (bv, bf)
            meshing.save_stl(bv, bf, out / "meshes" / f"cartilage_bridge_{side}_{n}.stl")
        (out / "cartilage_report.json").write_text(json.dumps(cart_report, indent=2))
        if bridges:
            meshing.export_combined(meshes, out, "bony_thorax")  # front connected

    # Costovertebral joints: connect each rib head to its thoracic vertebra
    # (the posterior articulation), closing the gap at the back of the cage.
    if cfg.add_costovertebral_link:
        vsurf = {lbl: v for lbl, (v, _f) in meshes.items()
                 if lbl.startswith("vertebrae_")}
        links, cv_report = joints.build_costovertebral_links(
            centerlines, vsurf, vert_centroids, cfg)
        for (side, n), (lv, lf) in links.items():
            meshes[f"costovertebral_{side}_{n}"] = (lv, lf)
            meshing.save_stl(lv, lf, out / "meshes" / f"costovertebral_{side}_{n}.stl")
        (out / "costovertebral_report.json").write_text(json.dumps(cv_report, indent=2))
        if links:
            meshing.export_combined(meshes, out, "bony_thorax")  # fully connected

    mech_rows = stages.stage_07_mechanics(centerlines, vert_centroids, meshes, cfg)

    # --- 08 report + render ---
    case_id = Path(ct_path).stem.replace(".nii", "")
    paths = report.write_reports(morph_rows, mech_rows, out / "reports", cfg,
                                 case_id=case_id, qc=qc)
    render = {lbl: (v, f, meshing.GROUP_COLORS[meshing.group_of(lbl)])
              for lbl, (v, f) in meshes.items()}
    if render:
        png = str(out / "thorax_3d.png")
        ttl = f"Bony thorax -- {case_id}"
        try:
            if args.render == "vtk":
                meshing.render_meshes_vtk(render, png, title=ttl)
            elif args.render == "mpl":
                meshing.render_meshes(render, png, title=ttl)
            else:                                      # software z-buffer
                raster.render(render, png, title=ttl, views=("anterior", "left"))
        except Exception as e:
            log.warning("render '%s' failed (%s); using software rasteriser",
                        args.render, e)
            raster.render(render, png, title=ttl, views=("anterior", "left"))
        paths["render"] = png

    log.info("Pipeline complete in %.1fs. Outputs in %s", time.time() - t0, out)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dicom", help="DICOM series folder")
    src.add_argument("--nifti", help="pre-converted .nii/.nii.gz volume")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seg-dir", help="existing TotalSegmentator masks folder")
    ap.add_argument("--skip-seg", action="store_true",
                    help="reuse masks in --seg-dir instead of running segmentation")
    ap.add_argument("--fast", action="store_true",
                    help="3mm low-res model (faster, lower quality). Omit for "
                         "full-resolution high-quality segmentation.")
    ap.add_argument("--device", default="auto",
                    help="auto | gpu (CUDA) | mps (Apple Silicon) | cpu")
    ap.add_argument("--render", choices=["vtk", "raster", "mpl"], default="vtk",
                    help="vtk = PyVista/OpenGL (best, needs GPU); "
                         "raster = software z-buffer (high-quality, headless); "
                         "mpl = legacy matplotlib")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
