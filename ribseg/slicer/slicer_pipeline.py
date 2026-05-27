"""
slicer_pipeline.py -- run/visualise the rib pipeline INSIDE 3D Slicer.

This is the interactive companion to the headless ``pipeline.py``.  It is meant
to be executed from Slicer's Python console (View > Python Console) or via
``Slicer --no-main-window --python-script slicer/slicer_pipeline.py -- <args>``.

What it does
------------
  1. Loads a CT volume (NIfTI or a DICOM directory).
  2. Runs the SlicerTotalSegmentator extension (task "total") to segment the
     ribs / vertebrae / sternum, then exports each segment to NIfTI.
  3. Calls the headless ``ribpipe`` package on those masks to compute the
     spinal frame, rib centerlines, morphometry, meshes and mechanics report.
  4. Loads the resulting STL meshes + rib centerlines back into the scene,
     colour-codes them, and prints any mechanics FAIL/WARN flags so you can
     jump straight to the offending joint in the 3D view.

Prerequisites
-------------
  * 3D Slicer 5.x
  * Extensions Manager > install "SlicerTotalSegmentator"
  * The ``rib_pipeline`` folder on disk (this file lives in it).

Usage
-----
  Slicer --python-script slicer/slicer_pipeline.py -- \
         --input /path/ct.nii.gz --out /path/run_dir

  # or, inside the Slicer Python console:
  #   exec(open('.../slicer/slicer_pipeline.py').read())
  #   run('/path/ct.nii.gz', '/path/run_dir')
"""
import os
import sys
import glob

import slicer            # provided by the Slicer Python environment
from slicer.util import loadVolume, loadSegmentation

# Make the sibling ``ribpipe`` package importable.
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)


# --------------------------------------------------------------------------- #
def load_ct(input_path):
    """Load a NIfTI file or a DICOM directory; return the volume node."""
    if os.path.isdir(input_path):
        from DICOMLib import DICOMUtils
        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(input_path, db)
            patients = db.patients()
            loaded = DICOMUtils.loadPatientByUID(patients[0])
        return slicer.mrmlScene.GetNodeByID(loaded[0])
    return loadVolume(input_path)


def run_totalsegmentator(volume_node, seg_dir, fast=False):
    """Run the SlicerTotalSegmentator extension and export masks to NIfTI.

    Falls back with a clear message if the extension is not installed.
    """
    os.makedirs(seg_dir, exist_ok=True)
    try:
        import TotalSegmentator
        logic = TotalSegmentator.TotalSegmentatorLogic()
    except Exception as exc:                          # pragma: no cover
        raise RuntimeError(
            "SlicerTotalSegmentator is not installed. Open Extensions Manager, "
            "install 'SlicerTotalSegmentator', restart Slicer, and retry. (%s)"
            % exc)

    seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    # The extension API has shifted across versions; try the common signatures.
    logic.setupPythonRequirements()
    try:
        logic.process(volume_node, seg_node, fast=fast, task="total")
    except TypeError:
        logic.process(volume_node, seg_node, fast, "total")

    # Export every segment as an individual binary labelmap NIfTI.
    seg = seg_node.GetSegmentation()
    ref = volume_node
    for i in range(seg.GetNumberOfSegments()):
        seg_id = seg.GetNthSegmentID(i)
        name = seg.GetSegment(seg_id).GetName()
        lm = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
            seg_node, [seg_id], lm, ref)
        slicer.util.saveNode(lm, os.path.join(seg_dir, "%s.nii.gz" % name))
        slicer.mrmlScene.RemoveNode(lm)
    return seg_dir


def run_analysis(seg_dir, out_dir):
    """Run the headless geometry/mechanics stages on exported masks."""
    from ribpipe.config import Config
    from ribpipe import stages, report
    cfg = Config()
    frame, vc = stages.stage_03_spine(seg_dir, cfg)
    cls = stages.stage_04_centerlines(seg_dir, frame, vc, cfg)
    morph = stages.stage_05_morphometry(seg_dir, frame, cls, cfg)
    meshes = stages.stage_06_mesh(seg_dir, os.path.join(out_dir, "meshes"), cfg)
    mech = stages.stage_07_mechanics(cls, vc, meshes, cfg)
    report.write_reports(morph, mech, os.path.join(out_dir, "reports"), cfg,
                         case_id="slicer_case")
    return cls, mech


def load_results_into_scene(out_dir, centerlines, mechanics_rows):
    """Load STL meshes + centerlines back into Slicer and flag mechanics issues."""
    colors = {"rib": (0.91, 0.79, 0.49), "vertebra": (0.66, 0.82, 0.86),
              "sternum": (0.96, 0.65, 0.65), "cartilage": (0.72, 0.88, 0.78)}

    for stl in glob.glob(os.path.join(out_dir, "meshes", "*.stl")):
        node = slicer.util.loadModel(stl)
        name = os.path.basename(stl)
        grp = ("rib" if "rib" in name else "vertebra" if "vertebra" in name
               else "sternum" if "sternum" in name else "cartilage"
               if "cartilage" in name else "rib")
        node.GetDisplayNode().SetColor(*colors.get(grp, (0.8, 0.8, 0.8)))
        node.GetDisplayNode().SetOpacity(0.9)

    # rib centerlines as markups curves
    for (side, n), cl in centerlines.items():
        curve = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode",
                                                   "centerline_%s_%d" % (side, n))
        for p in cl[::20]:                            # subsample for display
            curve.AddControlPoint(*[float(x) for x in p])
        curve.GetDisplayNode().SetSelectedColor(1, 0, 0)

    flags = [r for r in mechanics_rows if r.get("status") in ("FAIL", "WARN")]
    print("\n=== MECHANICS FLAGS (%d) ===" % len(flags))
    for r in flags:
        print("  ", r)
    print("Open the 3D view and inspect the flagged joints above.")


def run(input_path, out_dir, fast=False, skip_seg=False, seg_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    seg_dir = seg_dir or os.path.join(out_dir, "segmentations")
    vol = load_ct(input_path)
    if not skip_seg:
        run_totalsegmentator(vol, seg_dir, fast=fast)
    cls, mech = run_analysis(seg_dir, out_dir)
    load_results_into_scene(out_dir, cls, mech)
    print("Done. Reports in", os.path.join(out_dir, "reports"))


if __name__ == "__main__":
    import argparse
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-seg", action="store_true")
    ap.add_argument("--seg-dir", default=None)
    a = ap.parse_args(argv)
    run(a.input, a.out, fast=a.fast, skip_seg=a.skip_seg, seg_dir=a.seg_dir)
