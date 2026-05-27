"""Headless software renderer: a numpy z-buffer triangle rasteriser with
smooth (per-vertex-normal) Phong shading.

Matplotlib's 3D backend sorts whole polygons and cannot resolve occlusion, so
dense bony surfaces come out looking like scattered shells.  This module does
proper per-pixel depth testing and interpolated-normal shading entirely in
numpy, so the headless preview looks like a real lit surface -- without needing
OpenGL/GPU (the production path is still PyVista/Slicer on a GPU machine).
"""
from __future__ import annotations

import numpy as np

# anatomical view directions (camera sits along +dir, looks toward origin).
# World axes (NIfTI RAS-like): X=L-R, Y=post->ant, Z=inf->sup.
VIEWS = {
    "anterior":  (np.array([0, 1.0, 0]), np.array([0, 0, 1.0])),   # from front
    "posterior": (np.array([0, -1.0, 0]), np.array([0, 0, 1.0])),
    "left":      (np.array([-1.0, 0, 0]), np.array([0, 0, 1.0])),
    "right":     (np.array([1.0, 0, 0]), np.array([0, 0, 1.0])),
    "superior":  (np.array([0, 0, 1.0]), np.array([0, 1.0, 0])),
}


def _hex(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)]) / 255.0


def _vertex_normals(verts, faces):
    n = np.zeros_like(verts)
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # area-weighted
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1
    return n / ln


def _decimate_to(verts, faces, target):
    """Fast VTK quadric decimation to ~target faces (keeps shape)."""
    if len(faces) <= target:
        return verts, faces
    try:
        from . import surface
        pd = surface._polydata_from_arrays(verts, faces)
        import vtk
        dec = vtk.vtkQuadricDecimation()
        dec.SetInputData(pd)
        dec.SetTargetReduction(min(0.95, 1 - target / len(faces)))
        dec.Update()
        return surface._arrays_from_polydata(dec.GetOutput())
    except Exception:
        return verts, faces


def render(meshes: dict, out_png: str, title: str = "",
           views=("anterior", "left"), size=720, margin=0.06,
           bg=(1, 1, 1), light_dir=(0.3, 0.5, 1.0), dpi=130,
           max_faces_total=90000):
    """Render {label: (verts, faces, color_hex)} to a PNG via software z-buffer.

    Meshes are quadric-decimated to ``max_faces_total`` so the pure-numpy
    rasteriser stays fast while preserving shape. One panel per anatomical view.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = sum(len(f) for _v, f, _c in meshes.values() if len(f))
    frac = min(1.0, max_faces_total / max(total, 1))

    # merge meshes, remember per-vertex base colour
    V, F, C = [], [], []
    voff = 0
    for _label, (v, f, color) in meshes.items():
        if len(f) == 0:
            continue
        if frac < 1.0:
            v, f = _decimate_to(v, f, max(60, int(len(f) * frac)))
        V.append(np.asarray(v, float))
        F.append(np.asarray(f) + voff)
        C.append(np.tile(_hex(color), (len(v), 1)))
        voff += len(v)
    if not V:
        return
    V = np.vstack(V); F = np.vstack(F); C = np.vstack(C)
    N = _vertex_normals(V, F)
    L = np.array(light_dir, float); L /= np.linalg.norm(L)

    panels = [_render_view(V, F, C, N, VIEWS[vw], size, margin, bg, L)
              for vw in views]
    canvas = np.concatenate(panels, axis=1)

    fig = plt.figure(figsize=(canvas.shape[1] / dpi, canvas.shape[0] / dpi),
                     dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(canvas); ax.set_axis_off()
    if title:
        ax.text(0.5, 0.985, title, transform=ax.transAxes, ha="center",
                va="top", fontsize=11, color="#222")
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _render_view(V, F, C, N, view, size, margin, bg, L):
    d, up = view
    d = d / np.linalg.norm(d)
    right = np.cross(up, d); right /= np.linalg.norm(right)
    up = np.cross(d, right)

    sx = V @ right; sy = V @ up; depth = V @ d         # ortho projection
    # screen mapping preserving aspect
    lo = np.array([sx.min(), sy.min()]); hi = np.array([sx.max(), sy.max()])
    span = (hi - lo).max() * (1 + 2 * margin)
    cx, cy = (sx.min() + sx.max()) / 2, (sy.min() + sy.max()) / 2
    px = (sx - cx) / span * size + size / 2
    py = (cy - sy) / span * size + size / 2            # flip y for image coords

    img = np.ones((size, size, 3)) * np.array(bg)
    zbuf = np.full((size, size), -np.inf)

    P = np.stack([px, py], axis=1)
    tri = F
    p0, p1, p2 = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    z = depth[tri]; nrm = N[tri]; col = C[tri]
    # back-face cull (normal facing camera): use mean face normal . d
    fn = nrm.mean(1); facing = fn @ d
    keep = facing > -0.2                               # keep front-ish faces
    idx = np.where(keep)[0]

    area = ((p1[:, 0]-p0[:, 0])*(p2[:, 1]-p0[:, 1])
            - (p2[:, 0]-p0[:, 0])*(p1[:, 1]-p0[:, 1]))

    amb, dif, spec = 0.30, 0.75, 0.35
    for t in idx:
        a = area[t]
        if abs(a) < 1e-6:
            continue
        x0, y0 = p0[t]; x1, y1 = p1[t]; x2, y2 = p2[t]
        minx = max(int(np.floor(min(x0, x1, x2))), 0)
        maxx = min(int(np.ceil(max(x0, x1, x2))), size - 1)
        miny = max(int(np.floor(min(y0, y1, y2))), 0)
        maxy = min(int(np.ceil(max(y0, y1, y2))), size - 1)
        if minx > maxx or miny > maxy:
            continue
        xs = np.arange(minx, maxx + 1); ys = np.arange(miny, maxy + 1)
        gx, gy = np.meshgrid(xs, ys)
        w0 = ((y1-y2)*(gx-x2) + (x2-x1)*(gy-y2)) / a
        w1 = ((y2-y0)*(gx-x2) + (x0-x2)*(gy-y2)) / a
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        zt = w0*z[t, 0] + w1*z[t, 1] + w2*z[t, 2]
        sub = zbuf[gy[inside], gx[inside]]
        zin = zt[inside]
        closer = zin > sub
        if not closer.any():
            continue
        ii = gy[inside][closer]; jj = gx[inside][closer]
        w0c, w1c, w2c = w0[inside][closer], w1[inside][closer], w2[inside][closer]
        nn = (w0c[:, None]*nrm[t, 0] + w1c[:, None]*nrm[t, 1]
              + w2c[:, None]*nrm[t, 2])
        nn /= (np.linalg.norm(nn, axis=1, keepdims=True) + 1e-9)
        ndl = np.abs(nn @ L)
        # specular (Blinn-ish, halfway ~ light+view, view ~ d)
        h = L + d; h /= np.linalg.norm(h)
        sp = np.clip(np.abs(nn @ h), 0, 1) ** 24
        base = w0c[:, None]*col[t, 0] + w1c[:, None]*col[t, 1] + w2c[:, None]*col[t, 2]
        shade = amb + dif * ndl[:, None]
        rgb = np.clip(base * shade + spec * sp[:, None], 0, 1)
        img[ii, jj] = rgb
        zbuf[ii, jj] = zin[closer]
    return img
