"""
volume_render.py -- produce a photorealistic CT bone *volume rendering* (VRT)
inside 3D Slicer, i.e. the warm orange-bone look of a clinical 3D workstation
(as opposed to the smooth surface mesh produced by stage_06).

This is direct volume rendering: each voxel is colour/opacity-mapped by its HU
value through a transfer function and GPU ray-cast, so the trabecular texture
and realistic bone colouring are preserved.

Run from Slicer's Python console:
    exec(open('.../slicer/volume_render.py').read())
    render('/path/RibFrac501-image.nii.gz', '/path/out.png')

or headless / batch:
    Slicer --no-main-window --python-script slicer/volume_render.py -- \
           --input ct.nii.gz --out bone_vrt.png --preset CT-Bone
"""
import os
import sys
import slicer
from slicer.util import loadVolume

# Bone-ish presets shipped with Slicer's volume rendering module. "CT-Bone"
# gives the warm bone VRT in the reference image; try the others to taste.
BONE_PRESETS = ["CT-Bone", "CT-Bones", "CT-AAA", "CT-AAA2", "CT-Cardiac3"]


def render(input_path, out_png, preset="CT-Bone", view="anterior",
           bg=(0, 0, 0), size=(1400, 1000)):
    """Load a CT, apply a bone VRT preset, frame it, and save a PNG."""
    vol = loadVolume(input_path)

    vr = slicer.modules.volumerendering.logic()
    disp = vr.CreateVolumeRenderingDisplayNode()
    slicer.mrmlScene.AddNode(disp)
    disp.UnRegister(vr)
    vol.AddAndObserveDisplayNodeID(disp.GetID())
    vr.UpdateDisplayNodeFromVolumeNode(disp, vol)

    # Apply the chosen transfer-function preset (HU -> colour + opacity).
    preset_node = vr.GetPresetByName(preset)
    if preset_node:
        disp.GetVolumePropertyNode().Copy(preset_node)
    disp.SetVisibility(True)

    # Enable GPU ray casting + shading for the cinematic, lit-bone look.
    try:
        disp.SetRaycastTechnique(slicer.vtkMRMLVolumeRenderingDisplayNode.Composite)
        vp = disp.GetVolumePropertyNode().GetVolumeProperty()
        vp.ShadeOn()
        vp.SetAmbient(0.25); vp.SetDiffuse(0.75); vp.SetSpecular(0.45)
        vp.SetSpecularPower(12)
    except Exception:
        pass

    # 3D view: black background, orientation marker (the "A" anterior letter).
    lm = slicer.app.layoutManager()
    w = lm.threeDWidget(0); v = w.threeDView()
    vnode = v.mrmlViewNode()
    vnode.SetBackgroundColor(*bg); vnode.SetBackgroundColor2(*bg)
    vnode.SetBoxVisible(False); vnode.SetAxisLabelsVisible(False)
    vnode.SetOrientationMarkerType(
        slicer.vtkMRMLAbstractViewNode.OrientationMarkerTypeAxes)

    v.resetFocalPoint()
    _set_camera(view)
    v.forceRender()

    # Save a high-res screenshot of just the 3D view.
    import ScreenCapture
    cap = ScreenCapture.ScreenCaptureLogic()
    cap.captureImageFromView(v, out_png)
    print("Saved volume rendering ->", out_png)
    return out_png


def _set_camera(view):
    """Point the camera at a standard anatomical view."""
    cams = slicer.util.getNodesByClass("vtkMRMLCameraNode")
    if not cams:
        return
    cam = cams[0].GetCamera()
    fp = cam.GetFocalPoint()
    dist = cam.GetDistance() or 500
    presets = {
        "anterior":  (fp[0], fp[1] - dist, fp[2], 0, 0, 1),   # looking from front
        "posterior": (fp[0], fp[1] + dist, fp[2], 0, 0, 1),
        "left":      (fp[0] - dist, fp[1], fp[2], 0, 0, 1),
        "right":     (fp[0] + dist, fp[1], fp[2], 0, 0, 1),
    }
    px, py, pz, ux, uy, uz = presets.get(view, presets["anterior"])
    cam.SetPosition(px, py, pz)
    cam.SetViewUp(ux, uy, uz)


def render_all_views(input_path, out_dir, preset="CT-Bone"):
    """Save anterior / posterior / left / right VRTs of one scan."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    for vw in ("anterior", "posterior", "left", "right"):
        render(input_path, os.path.join(out_dir, f"{stem}_{vw}.png"),
               preset=preset, view=vw)
        # remove the volume so the next view starts clean
        slicer.mrmlScene.Clear(0)


if __name__ == "__main__":
    import argparse
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="CT-Bone")
    ap.add_argument("--view", default="anterior")
    a = ap.parse_args(argv)
    render(a.input, a.out, preset=a.preset, view=a.view)
