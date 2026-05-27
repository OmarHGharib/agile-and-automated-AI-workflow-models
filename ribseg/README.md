# Automated Rib Morphometry, 3D Bony-Thorax Reconstruction & Biomechanical Verification

An end-to-end, automated pipeline that takes chest-CT slices of one individual and produces:

1. **Rib morphometry** — for all 24 ribs (12 pairs): maximum **length**, maximum/minimum **width**, and maximum/minimum **height**, all measured in a patient-specific coordinate frame whose vertical axis is the detected spinal column axis.
2. **3D reconstruction** of the bony thorax from the 2D slices, plus a **biomechanical / anatomical consistency check** that verifies the ribs articulate with the correct thoracic vertebrae and that true ribs reach the sternum, adjacent vertebrae stay in contact, and floating ribs stay free.
3. A **single automated pipeline** (`pipeline.py`) and a **batch runner** (`run_batch.py`) that generate all of the above, plus an interactive **3D Slicer** companion script.

The methodology follows the two reference papers and the project approach documents:

- Wasserthal et al., *TotalSegmentator*, Radiology: AI (2023) — deep-learning segmentation (nnU-Net) of the 24 individual ribs, T1–T12, sternum and costal cartilages.
- Jin et al., *RibSeg v2*, IEEE TMI (2023) — rib labeling and TEASAR/L1-medial anatomical centerline extraction (the basis for the length/curvature measurements).

---

## Why the design is split the way it is

Only **one** stage (segmentation) needs the deep-learning model and a GPU. Every other stage is **pure geometry**. The code is deliberately split along that line:

```
CT ──▶ [02 TotalSegmentator]──▶ per-structure masks ──▶ [03..08 pure geometry] ──▶ measurements + 3D model + report
        (GPU, run on your machine)                        (CPU, runs anywhere, fully unit-tested)
```

That separation is what lets the measurement engine be **validated against a synthetic phantom with analytically-known dimensions** (see *Validation* below) — independent of any segmentation.

---

## Pipeline stages

| Stage | Module | What it does |
|------:|--------|--------------|
| 00 | `pipeline.py` (preflight) | check dependencies (TotalSegmentator, kimimaro) |
| 01 | `ribpipe/io_utils.py` | DICOM → NIfTI (SimpleITK), QC (spacing, HU range, orientation) |
| 02 | `ribpipe/segment.py` | TotalSegmentator → one binary NIfTI mask per structure |
| 03 | `ribpipe/geometry.py` `build_anatomical_frame` | **spinal axis** by PCA over T1–T12 centroids → right-handed anatomical frame (Z = spine, X = left-right, Y = posterior→anterior) |
| 04 | `ribpipe/geometry.py` `extract_centerline` | rib **centerlines** (kimimaro TEASAR, with a scipy thinning + graph-diameter fallback), oriented posterior→anterior, resampled to 500 points (RibSeg v2 convention) |
| 05 | `ribpipe/geometry.py` `rib_measurements` | **length** (centerline arc length), **max width/height** (global L-R / S-I extent in the anatomical frame), **min width/height** (smallest cross-section measured in the plane ⟂ to the local tangent), curvature |
| 06 | `ribpipe/meshing.py` | marching-cubes **surface meshes** → STL + shaded PNG render |
| 07 | `ribpipe/mechanics.py` | **biomechanics checks**: costo-vertebral, costo-sternal (true ribs 1–7), inter-vertebral (disc contact), floating-rib (11–12 must be far from sternum) |
| 08 | `ribpipe/report.py` | CSV + Excel + JSON reports |

### How the measurements are defined (Task 1)

All extents are taken **after** rotating every voxel into the anatomical frame, so they are independent of how the patient was tilted in the scanner:

- **length_mm** — arc length of the rib centerline (true curved length, not a chord).
- **max_width_mm** — left-right (anatomical X) span of the whole rib.
- **max_height_mm** — superior-inferior (anatomical Z) span of the whole rib.
- **min_width_mm / min_height_mm** — the smallest cross-sectional width/height, computed on rings of voxels in the plane perpendicular to the local centerline tangent (this is the rib *bone* thickness, and is robust to rib orientation — the naive "spread along X within a slice" collapses where a rib runs parallel to X, which this avoids).
- **max_depth_mm**, **curvature_max** — bonus posterior-anterior span and peak curvature (a discontinuity/fracture indicator, per the curvature analysis in the approach deck).

### How the biomechanics check works (Task 2)

| Check | Rule | Default threshold |
|-------|------|-------------------|
| Costo-vertebral | posterior centerline endpoint near the matching Tn centroid | ≤ 12 mm |
| Costo-sternal (ribs 1–7) | anterior endpoint near sternum / costal-cartilage surface | ≤ 25 mm |
| Inter-vertebral | adjacent Tn / Tn+1 surfaces in near contact (disc space) | ≤ 10 mm |
| Floating rib (11–12) | anterior end must be **far** from the sternum | ≥ 30 mm |

Thresholds live in `ribpipe/config.py` so the clinical assumptions are auditable in one place.

---

## Installation

```bash
pip install -r requirements.txt
# Segmentation (GPU machine), installed separately:
pip install TotalSegmentator           # or use the SlicerTotalSegmentator extension
```

`kimimaro` is optional; without it the centerline stage uses a scipy fallback that the validation suite confirms is equivalent for measurement purposes.

---

## Usage

```bash
# Full pipeline from a DICOM folder
python pipeline.py --dicom /path/to/dicom --out runs/patientA

# From an existing NIfTI volume (e.g. a RibFrac scan) -- full-resolution,
# auto-selecting the best device (CUDA gpu / Apple mps / cpu):
python pipeline.py --nifti RibFrac501-image.nii.gz --out runs/RibFrac501 --device auto

# Force a device, or low-res fast mode:
python pipeline.py --nifti ct.nii.gz --out runs/x --device mps      # Apple Silicon
python pipeline.py --nifti ct.nii.gz --out runs/x --device cpu --fast

# Re-use existing TotalSegmentator masks (skip the GPU step)
python pipeline.py --nifti ct.nii.gz --seg-dir masks/ --skip-seg --out runs/patientA

# Batch over the whole RibFrac test set, one subfolder per case
python run_batch.py --in-dir /path/to/ribfrac-test-images \
                    --out runs_ribfrac --pattern "*-image.nii.gz" --limit 5
```

Outputs per case: `reports/<case>_rib_morphometry.csv`, `<case>_mechanics_report.csv`, `<case>_report.xlsx`, `<case>_summary.json`, `coverage.json` (which structures are present/missing/empty), `meshes/*.stl` (one per structure), a combined **`bony_thorax.stl`** + colour-coded **`.glb`/`.obj`**, and `thorax_3d.png`.

### Rendering & a photorealistic view

`pipeline.py --render vtk` (default) uses PyVista/OpenGL for clean, solid, lit surfaces; `--render mpl` is a headless Matplotlib fallback. To re-render an existing meshes folder at high quality:

```bash
python render_stls.py --in-dir runs/RibFrac501/meshes --out runs/RibFrac501/thorax_hq.png --backend vtk
```

The surface mesh is the geometry/measurement model. For the **clinical volume-rendered bone look** (the warm "VRT" image), use `slicer/volume_render.py` inside 3D Slicer (Volume Rendering module, `CT-Bone` preset) — that renders the CT voxels directly and preserves trabecular texture, which a surface mesh intentionally discards.

### Why a model might look incomplete

`coverage.json` tells you exactly which structures made it into the model. Two things are expected, not bugs: (1) **costal cartilage** is largely radiolucent on CT, so it is faint/absent unless calcified; (2) the **HU-threshold demo** (not the real pipeline) also picks up the **CT table** and arm bones — `demo_reconstruct.py` removes those with a flat-slab filter. The real TotalSegmentator pipeline segments named anatomy only, so it never contains the table.

### Ribs not touching the sternum — and the cartilage bridge

This is anatomy, not a bug: ribs do **not** articulate with the sternum directly. True ribs (1–7) join it through their **costal cartilage**; false ribs (8–10) join the cartilage of the rib above; floating ribs (11–12) are free. Costal cartilage is radiolucent on CT, so a pure bone model always shows a gap between each anterior rib end and the sternum.

The pipeline closes that gap (`ribpipe/cartilage.py`, on by default — `Config.add_cartilage_bridge`): if TotalSegmentator produced a `costal_cartilages` mask it is meshed directly; otherwise a **synthetic tapered cartilage bridge** is generated from each true/false-rib anterior endpoint to the nearest sternum/cartilage surface. Bridges are written as `meshes/cartilage_bridge_*.stl`, folded into the combined `bony_thorax.stl`, and every bridged gap is logged in `cartilage_report.json`. These synthetic bridges are a **model element** (clearly named), not measured bone, so the morphometry numbers are unaffected.

### Quality / device

Run **full-resolution** (omit `--fast`; `--fast` is the coarse 3 mm model). `--device auto` picks CUDA → Apple MPS → CPU; if MPS errors on nnU-Net it auto-retries on CPU so the run still finishes. Meshes are hole-filled, largest-component-cleaned, Gaussian-pre-smoothed and Taubin-smoothed, so surfaces look like bone rather than voxel staircases — and `--render vtk` on your Mac gives the clean lit result.

### Interactive 3D Slicer path

```bash
Slicer --python-script slicer/slicer_pipeline.py -- --input ct.nii.gz --out runs/patientA
```

Runs SlicerTotalSegmentator, exports masks, runs the same geometry engine, then loads the STL meshes + rib centerlines back into the Slicer scene (color-coded) and prints the mechanics FAIL/WARN flags so you can jump to the offending joint.

---

## Validation (proof the measurement engine is correct)

Because the per-rib labels need a GPU, the geometry engine is validated against a **synthetic thorax phantom** with closed-form dimensions: 24 ribs built as tubes of known radius around analytic arc centerlines, 12 vertebra cubes, and a sternum slab (`validation/make_phantom.py`).

```bash
python validation/test_geometry.py
```

The harness runs the **real** pipeline stages on the phantom masks and asserts the recovered values match ground truth. Result:

```
[frame]     recovered spinal Z within 0.00° of true axis            PASS
[length]    all 24 ribs within 7% of analytic arc length            PASS
[max W/H]   within 3 mm of analytic tube extents (incl. end caps)   PASS
[min W/H]   ~9.5 mm vs true 10 mm tube diameter                     PASS
[mechanics] 24/24 costo-vertebral PASS, 14/14 costo-sternal PASS,
            11/11 inter-vertebral PASS, 4/4 floating ribs flagged far  PASS
==> VALIDATION PASSED — geometry engine matches ground truth.
```

Deliverables from this run: `validation/phantom_reports/` (the validated morphometry Excel/CSV) and `validation/phantom_thorax_3d.png` (color-coded labelled reconstruction).

---

## Real-CT demonstration (RibFrac)

`demo/demo_reconstruct.py` runs the I/O + reconstruction path on **actual RibFrac scans** (HU thresholding feeds the same marching-cubes `stage_06` code, since per-rib labelling needs the GPU model):

```bash
python demo/demo_reconstruct.py --in-dir <ribfrac-test-images> --out demo/out \
       --cases RibFrac501 RibFrac502 --downsample 2
```

5 RibFrac cases (501–505) were reconstructed here: each produced an STL bony-thorax mesh and a shaded anterior+lateral render (`demo/out/<case>/`), with bone volumes of ~675–1017 cm³. See `demo/out/demo_summary.csv`.

---

## Surface quality, rendering & cartilage (literature-grounded redesign)

Three things were rebuilt around recent work in the field:

**1. Surface extraction (`ribpipe/surface.py`).** Raw marching cubes gives a voxel-staircase surface. Following the medical-mesh literature (Flying Edges + implicit/low-pass smoothing, the method 3D Slicer uses for closed-surface export), each mask is now iso-surfaced with **VTK `vtkFlyingEdges3D`** and smoothed with a **windowed-sinc low-pass filter** (`vtkWindowedSincPolyDataFilter`) — a true shrink-free smoother, so thin ribs keep their calibre — with optional quadric decimation. Masks are cropped to their bounding box first for speed. This is the default mesh path (skimage marching cubes + Taubin remains a fallback).

**2. Rendering (`ribpipe/raster.py`).** Matplotlib's 3D backend sorts whole polygons and cannot resolve occlusion, which is why dense bone looked like a scattered shell. There is now a **pure-numpy software z-buffer rasteriser** with per-vertex-normal (smooth) Phong shading and proper per-pixel depth testing — crisp, solid, lit images with no GPU. `--render` selects `vtk` (PyVista/OpenGL, best, needs a GPU — use on your Mac), `raster` (this software renderer, high-quality headless), or `mpl` (legacy).

**3. Costal cartilage (`ribpipe/cartilage.py`).** Recent work shows costal cartilage *is* segmentable on CT (e.g. the 2024 Topology-Guided Deformable Mamba benchmark, 165 cases; pediatric studies reconstructing 500+ cartilages). So the model now **prefers the real `costal_cartilages` mask**. Only when it is absent does it synthesise cartilage — and not as a straight strut: each bridge is a tapered tube along a **quadratic Bézier that leaves the rib along its own tangent (C1-continuous) and curves into the sternum**, enforcing the rib→cartilage→sternum topology, so it reads as a natural continuation of the rib.

Why ribs look "disconnected" in a bone-only model is itself anatomy: ribs join the sternum through cartilage, which is radiolucent on CT — see below.

**Posterior (costovertebral) connection.** Unlike the front, the rib head and the thoracic vertebra are *both* segmented bone that already articulate, so no synthetic geometry is added by default — drawing a strut there only looks fabricated. The rib head simply sits against the vertebra as real data (a small natural joint gap remains). A synthetic connector (`ribpipe/joints.py`, `Config.add_costovertebral_link`, **off by default**) is available for biomechanics/FEA use, but the realistic CT model relies on the segmentation itself.

> Meshing note: structure meshes use `keep_largest=False` so a multi-piece mask (the single `costal_cartilages` mask holds ~24 separate pieces; a fractured rib is >1 piece) keeps **all** components rather than only the largest.

## Limitations

- Costal cartilage is largely invisible on CT, so the costo-sternal check uses a tolerant threshold and the cartilage mask when TotalSegmentator provides it.
- The synthetic phantom validates *measurement correctness*, not segmentation accuracy; segmentation accuracy is TotalSegmentator's responsibility (Dice ≈ 0.94 on ribs per the paper).
- The threshold-based real-CT demo reconstructs the whole bony thorax (it cannot separate individual ribs without the DL labels) — run `pipeline.py` on a GPU machine for per-rib measurements on real scans.

## References

1. Wasserthal et al. (2023) *TotalSegmentator*. Radiology: AI 5(5).
2. Jin et al. (2023) *RibSeg v2*. IEEE TMI. arXiv:2210.09309 / 2208.05868.
3. Isensee et al. (2021) *nnU-Net*. Nature Methods 18.
4. Fedorov et al. (2012) *3D Slicer*. Magn. Reson. Imaging 30(9).
5. Costal Cartilage Segmentation with Topology-Guided Deformable Mamba: Method and Benchmark (2024). arXiv:2408.07444 — 165-case costal-cartilage CT benchmark; motivates using the real cartilage mask and enforcing topology.
6. Schroeder et al. *Flying Edges: A High-Performance Scalable Isocontouring Algorithm* (VTK) — used here for fast, high-quality iso-surfacing; windowed-sinc smoothing per Taubin (1995), as in 3D Slicer's surface export.
7. Deep-learning age estimation from costal-cartilage CT reconstructions, Eur. Radiol. (2023) — evidence costal cartilage is reliably reconstructable from CT.
