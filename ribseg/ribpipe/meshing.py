"""Surface-mesh generation from binary masks + export + rendering.

Improvements over the first version:
  * the mask is hole-filled, reduced to its largest connected component, and
    lightly Gaussian-smoothed *before* marching cubes, then the surface is
    Taubin-smoothed -- so meshes look like real bone, not voxel staircases;
  * exports per-structure STL **and** a single combined ``bony_thorax.stl``
    plus a colour-coded GLB/OBJ for viewers;
  * the Matplotlib renderer now draws the full surface (no scatter decimation)
    with flat lighting, so structures read as solid;
  * an optional high-quality PyVista renderer (``render_meshes_vtk``) for
    machines with a GPU / OpenGL (e.g. the user's Mac).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .geometry import voxel_to_world


# --------------------------------------------------------------------------- #
# Mask -> mesh
# --------------------------------------------------------------------------- #
def mask_to_mesh(mask: np.ndarray, affine: np.ndarray, level: float = 0.5,
                 smooth_mask_sigma: float = 0.6, keep_largest: bool = True,
                 taubin_iter: int = 12, smooth_iter: int = 20,
                 pass_band: float = 0.10, decimate: float = 0.0):
    """Smooth world-space surface (verts mm, faces) of a binary mask.

    Primary path: VTK Flying Edges + windowed-sinc smoothing (medical-grade,
    the method used by 3D Slicer's closed-surface export). Falls back to
    skimage marching cubes + Taubin smoothing if VTK is unavailable.
    """
    try:
        from . import surface
        v, f = surface.mask_to_surface(
            mask, affine, smooth_iter=smooth_iter, pass_band=pass_band,
            sigma_vox=smooth_mask_sigma, keep_largest=keep_largest,
            decimate=decimate)
        if len(v) > 0:
            return v, f
    except Exception:
        pass

    # --- fallback: skimage marching cubes + Taubin ---
    from skimage.measure import marching_cubes
    m = mask.astype(bool)
    if m.sum() < 27 or min(m.shape) < 3:
        return np.empty((0, 3)), np.empty((0, 3), int)
    if keep_largest:
        lab, n = ndimage.label(m)
        if n > 1:
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            m = lab == sizes.argmax()
    m = ndimage.binary_fill_holes(m)
    vol = ndimage.gaussian_filter(m.astype(np.float32), smooth_mask_sigma)
    try:
        verts_vox, faces, _n, _v = marching_cubes(vol, level=level)
    except (ValueError, RuntimeError):
        return np.empty((0, 3)), np.empty((0, 3), int)
    verts = voxel_to_world(verts_vox, affine)
    if taubin_iter:
        verts, faces = _taubin(verts, faces, taubin_iter)
    return verts, faces.astype(int)


def _taubin(verts, faces, iterations):
    """Taubin (shrink-free) smoothing via trimesh, with a safe fallback."""
    try:
        import trimesh
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        trimesh.smoothing.filter_taubin(tm, iterations=iterations)
        return np.asarray(tm.vertices), np.asarray(tm.faces)
    except Exception:
        return verts, faces


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def save_stl(verts: np.ndarray, faces: np.ndarray, path: str) -> bool:
    from stl import mesh as stlmesh
    if len(verts) == 0 or len(faces) == 0:
        return False
    m = stlmesh.Mesh(np.zeros(len(faces), dtype=stlmesh.Mesh.dtype))
    m.vectors[:] = verts[faces]
    m.save(str(path))
    return True


def export_combined(meshes: dict, out_dir, stem: str = "bony_thorax") -> dict:
    """Merge {label: (verts, faces)} into a single STL + colour-coded GLB/OBJ.

    Returns paths actually written.
    """
    from pathlib import Path
    import trimesh
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for label, (v, f) in meshes.items():
        if len(f) == 0:
            continue
        tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
        rgb = (_hex_to_rgb(GROUP_COLORS[group_of(label)]) * 255).astype(np.uint8)
        tm.visual.face_colors = np.tile([*rgb, 255], (len(f), 1))
        parts.append(tm)
    if not parts:
        return {}
    combined = trimesh.util.concatenate(parts)
    paths = {}
    p = out_dir / f"{stem}.stl"; combined.export(p); paths["stl"] = str(p)
    try:
        p = out_dir / f"{stem}.glb"; combined.export(p); paths["glb"] = str(p)
    except Exception:
        pass
    try:
        p = out_dir / f"{stem}.obj"; combined.export(p); paths["obj"] = str(p)
    except Exception:
        pass
    return paths


# --------------------------------------------------------------------------- #
# Rendering -- Matplotlib (headless, no GPU) : solid full-surface
# --------------------------------------------------------------------------- #
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)]) / 255.0


def render_meshes(meshes: dict, out_png: str, title: str = "",
                  views=((10, -90), (10, -180)), dpi: int = 130,
                  max_faces_total: int = 700000, light=(0.4, 0.4, 0.9),
                  bg="white") -> None:
    """Render {label: (verts, faces, color)} to a PNG (headless, solid surface).

    Unlike the old version this does NOT skip every Nth face (which scattered
    the surface).  If the *total* face count exceeds ``max_faces_total`` the
    individual meshes are quadric-decimated (trimesh) so they stay watertight.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    light = np.array(light, float); light /= np.linalg.norm(light)
    total = sum(len(f) for _v, f, _c in meshes.values())
    keep = min(1.0, max_faces_total / max(total, 1))

    prepared = []
    for label, (verts, faces, color) in meshes.items():
        if len(faces) == 0:
            continue
        if keep < 1.0:
            verts, faces = _decimate(verts, faces, max(0.05, keep))
        prepared.append((verts, faces, color))

    n = len(views)
    fig = plt.figure(figsize=(6.5 * n, 7))
    allv = [v for v, f, _ in prepared if len(f)]
    if allv:
        V = np.vstack(allv); ctr = V.mean(0); rad = np.ptp(V, 0).max() / 2

    for vi, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, n, vi + 1, projection="3d")
        az, el = np.radians(azim), np.radians(elev)
        viewdir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                            np.sin(el)])
        for verts, faces, color in prepared:
            tris = verts[faces]
            e1 = tris[:, 1] - tris[:, 0]; e2 = tris[:, 2] - tris[:, 0]
            nrm = np.cross(e1, e2)
            nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
            shade = 0.45 + 0.55 * np.clip(np.abs(nrm @ light), 0, 1)
            cols = np.clip(shade[:, None] * _hex_to_rgb(color)[None, :], 0, 1)
            depth = tris.mean(1) @ viewdir
            order = np.argsort(depth)
            coll = Poly3DCollection(tris[order], linewidths=0, shade=False)
            coll.set_facecolor(cols[order]); coll.set_edgecolor("none")
            ax.add_collection3d(coll)
        if allv:
            ax.set_xlim(ctr[0]-rad, ctr[0]+rad); ax.set_ylim(ctr[1]-rad, ctr[1]+rad)
            ax.set_zlim(ctr[2]-rad, ctr[2]+rad)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
    if title:
        fig.suptitle(title, fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor=bg)
    plt.close(fig)


def _decimate(verts, faces, fraction):
    try:
        import trimesh
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        target = max(50, int(len(faces) * fraction))
        tm2 = tm.simplify_quadric_decimation(target)
        if len(tm2.faces) > 0:
            return np.asarray(tm2.vertices), np.asarray(tm2.faces)
    except Exception:
        pass
    return verts, faces


# --------------------------------------------------------------------------- #
# Rendering -- PyVista (GPU / OpenGL) : photorealistic. Use on the Mac.
# --------------------------------------------------------------------------- #
def render_meshes_vtk(meshes: dict, out_png: str, title: str = "",
                      views=("anterior", "left"), window=(1400, 1000),
                      bg="white") -> None:
    """High-quality colour render via PyVista (needs OpenGL).

    ``meshes`` = {label: (verts, faces, color_hex)}.  Saves one PNG, optionally
    a horizontal montage of multiple anatomical views.
    """
    import pyvista as pv
    pv.OFF_SCREEN = True

    cams = {"anterior": (0, -1, 0), "posterior": (0, 1, 0),
            "left": (-1, 0, 0), "right": (1, 0, 0), "superior": (0, 0, 1)}
    pl = pv.Plotter(off_screen=True, window_size=list(window),
                    shape=(1, len(views)))
    for i, vw in enumerate(views):
        pl.subplot(0, i)
        for label, (v, f, color) in meshes.items():
            if len(f) == 0:
                continue
            faces_vtk = np.c_[np.full(len(f), 3), f].ravel()
            mesh = pv.PolyData(v, faces_vtk)
            pl.add_mesh(mesh, color=color, smooth_shading=True,
                        specular=0.3, specular_power=15)
        pl.set_background(bg)
        d = cams.get(vw, (0, -1, 0))
        pl.camera_position = [(d[0]*1e3, d[1]*1e3, d[2]*1e3), (0, 0, 0),
                              (0, 0, 1)]
        pl.reset_camera()
    if title:
        pl.add_text(title, position="upper_edge", font_size=10, color="black")
    pl.screenshot(out_png)
    pl.close()


# --------------------------------------------------------------------------- #
GROUP_COLORS = {
    "rib": "#E8C97E", "vertebra": "#A8D0DB", "sternum": "#F4A7A7",
    "cartilage": "#B8E0C8", "clavicle": "#D0B8E8", "scapula": "#E8D8B0",
    "joint": "#C98BB0", "bone": "#E8C97E",
}


def group_of(label: str) -> str:
    if "costovertebral" in label or label.startswith("cv_") or "joint" in label:
        return "joint"
    if "cartilage" in label:        # incl. cartilage_bridge_*
        return "cartilage"
    if "rib" in label:
        return "rib"
    if "vertebra" in label:
        return "vertebra"
    if "sternum" in label:
        return "sternum"
    if "clavicula" in label:
        return "clavicle"
    if "scapula" in label:
        return "scapula"
    return "bone"
