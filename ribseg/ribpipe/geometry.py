"""Core geometry engine -- pure, dependency-light, fully unit-testable.

Nothing in this module touches the deep-learning segmentation step.  Every
function operates on binary masks + affines (or on point arrays), which is why
the same code that runs on TotalSegmentator output can be validated against a
synthetic phantom with analytically known dimensions (see validation/).

Coordinate conventions
----------------------
* ``affine`` is a 4x4 voxel->world matrix (NIfTI ``img.affine``); world units mm.
* The *anatomical frame* is right-handed with
    Z = spinal longitudinal axis (inferior -> superior),
    X = left-right,
    Y = posterior -> anterior (toward sternum).
  Transform:  ``anat = (world - origin) @ R`` where R = [X | Y | Z] (cols).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

try:                       # optional, high-quality TEASAR skeletons
    import kimimaro
    _HAS_KIMIMARO = True
except Exception:          # pragma: no cover - fallback path used in sandbox/tests
    _HAS_KIMIMARO = False


# --------------------------------------------------------------------------- #
# Voxel <-> world helpers
# --------------------------------------------------------------------------- #
def voxel_to_world(ijk: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Map (...,3) voxel indices to world (mm) coordinates via the NIfTI affine."""
    ijk = np.asarray(ijk, dtype=float)
    flat = ijk.reshape(-1, 3)
    homog = np.c_[flat, np.ones(len(flat))]
    world = (affine @ homog.T).T[:, :3]
    return world.reshape(ijk.shape)


def mask_to_world_points(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Return (M,3) world coordinates of every True voxel in ``mask``."""
    ijk = np.argwhere(mask)
    if ijk.size == 0:
        return np.empty((0, 3))
    return voxel_to_world(ijk, affine)


def center_of_mass_world(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """World-coordinate centroid of a binary mask."""
    com = np.array(ndimage.center_of_mass(mask.astype(bool)))
    return voxel_to_world(com[None, :], affine)[0]


def voxel_sizes(affine: np.ndarray) -> np.ndarray:
    """Per-axis voxel spacing (mm) recovered from the affine."""
    return np.linalg.norm(affine[:3, :3], axis=0)


# --------------------------------------------------------------------------- #
# Stage 03: spinal coordinate frame
# --------------------------------------------------------------------------- #
def build_anatomical_frame(vert_centroids: np.ndarray,
                           ap_hint: np.ndarray | None = None) -> dict:
    """Construct the anatomical frame from ordered T1..T12 centroids.

    Parameters
    ----------
    vert_centroids : (K,3) world coords, ordered superior(T1) -> inferior(T12).
    ap_hint : optional world vector approximating posterior->anterior; used to
        orient X/Y. Defaults to world +Y (typical NIfTI A axis).

    Returns
    -------
    dict with keys: origin, x_axis, y_axis, z_axis, R (3x3, columns X,Y,Z).
    """
    c = np.asarray(vert_centroids, dtype=float)
    if c.shape[0] < 2:
        raise ValueError("need >=2 vertebra centroids to fit a spinal axis")
    origin = c.mean(axis=0)

    # PC1 of the centroid cloud = best-fit line through the vertebral bodies.
    _, _, Vt = np.linalg.svd(c - origin)
    z = Vt[0]
    # Orient Z from inferior (T12, last) toward superior (T1, first).
    if np.dot(z, c[0] - c[-1]) < 0:
        z = -z
    z /= np.linalg.norm(z)

    ap = np.array([0.0, 1.0, 0.0]) if ap_hint is None else np.asarray(ap_hint, float)
    # X (left-right) = ap x z ; then Y = z x X  -> right-handed, all orthonormal.
    x = np.cross(ap, z)
    if np.linalg.norm(x) < 1e-6:                      # ap nearly parallel to z
        x = np.cross(np.array([1.0, 0.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)

    R = np.stack([x, y, z], axis=1)                   # columns are the axes
    return {"origin": origin, "x_axis": x, "y_axis": y, "z_axis": z, "R": R}


def world_to_anat(points: np.ndarray, frame: dict) -> np.ndarray:
    """Rotate/translate world points into the anatomical frame."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    return (pts - frame["origin"]) @ frame["R"]


# --------------------------------------------------------------------------- #
# Stage 04: rib centerline extraction
# --------------------------------------------------------------------------- #
def _skeleton_path_voxels(mask: np.ndarray) -> np.ndarray:
    """Ordered (P,3) voxel path along a thin tubular structure (scipy fallback).

    Strategy: 3D thinning -> build a graph over skeleton voxels (26-conn,
    spacing-agnostic here, refined later) -> the longest geodesic path
    (graph diameter, found by double shortest-path) is the centerline.
    """
    from skimage.morphology import skeletonize

    skel = skeletonize(mask.astype(bool))             # works for 2D & 3D
    nodes = np.argwhere(skel)
    if len(nodes) < 2:
        # Degenerate: fall back to the mask voxels themselves.
        nodes = np.argwhere(mask)
        if len(nodes) < 2:
            return nodes
    index = {tuple(p): i for i, p in enumerate(nodes)}
    tree = cKDTree(nodes)
    pairs = tree.query_pairs(r=np.sqrt(3) + 1e-6)     # 26-connectivity
    if not pairs:
        return nodes[:1]
    rows, cols, data = [], [], []
    for a, b in pairs:
        d = np.linalg.norm(nodes[a] - nodes[b])
        rows += [a, b]; cols += [b, a]; data += [d, d]
    n = len(nodes)
    g = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    # Double sweep to find the two extremal endpoints of the largest component.
    d0, _ = shortest_path(g, indices=0, return_predecessors=True)
    finite = np.isfinite(d0)
    src = int(np.argmax(np.where(finite, d0, -1)))
    dist, pred = shortest_path(g, indices=src, return_predecessors=True)
    dst = int(np.argmax(np.where(np.isfinite(dist), dist, -1)))

    # Reconstruct src -> dst path.
    path = []
    j = dst
    while j != src and j >= 0:
        path.append(j)
        j = pred[j]
    path.append(src)
    path = path[::-1]
    return nodes[path]


def extract_centerline(mask: np.ndarray, affine: np.ndarray,
                       n_points: int = 500, smooth_sigma: float = 1.5,
                       root_world: np.ndarray | None = None,
                       teasar_params: dict | None = None) -> np.ndarray:
    """Return an ordered (n_points, 3) world-coordinate centerline for a rib.

    Uses kimimaro/TEASAR when available (RibSeg v2 convention); otherwise a
    scipy-based thinning + graph-diameter fallback that produces an equivalent
    ordered medial polyline.  ``root_world`` (e.g. the vertebra centroid) orients
    the polyline so index 0 is the posterior/costo-vertebral end.
    """
    spacing = voxel_sizes(affine)
    if _HAS_KIMIMARO:
        skels = kimimaro.skeletonize(
            mask.astype("uint8"),
            teasar_params=teasar_params or {},
            anisotropy=tuple(float(s) for s in spacing),
            dust_threshold=0, progress=False, fix_branching=True,
        )
        if skels:
            sk = next(iter(skels.values()))
            # kimimaro vertices are already in physical (anisotropic) space.
            verts = sk.vertices
            path_world = _order_by_longest_path(verts, sk.edges)
        else:
            path_world = voxel_to_world(_skeleton_path_voxels(mask), affine)
    else:
        path_world = voxel_to_world(_skeleton_path_voxels(mask), affine)

    if len(path_world) < 2:
        return np.repeat(path_world, n_points, axis=0)[:n_points]

    # Orient: index 0 nearest to root (vertebra) if provided.
    if root_world is not None:
        if (np.linalg.norm(path_world[0] - root_world)
                > np.linalg.norm(path_world[-1] - root_world)):
            path_world = path_world[::-1]

    path_world = _smooth_polyline(path_world, smooth_sigma)
    return resample_polyline(path_world, n_points)


def _order_by_longest_path(verts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Order kimimaro skeleton vertices along the graph diameter."""
    n = len(verts)
    rows, cols, data = [], [], []
    for a, b in edges:
        d = np.linalg.norm(verts[a] - verts[b])
        rows += [a, b]; cols += [b, a]; data += [d, d]
    g = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    d0, _ = shortest_path(g, indices=0, return_predecessors=True)
    src = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    dist, pred = shortest_path(g, indices=src, return_predecessors=True)
    dst = int(np.argmax(np.where(np.isfinite(dist), dist, -1)))
    path, j = [], dst
    while j != src and j >= 0:
        path.append(j); j = pred[j]
    path.append(src)
    return verts[path[::-1]]


def _smooth_polyline(pts: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(pts) < 5:
        return pts
    return np.stack([ndimage.gaussian_filter1d(pts[:, k], sigma, mode="nearest")
                     for k in range(3)], axis=1)


def resample_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample an open polyline to ``n`` equidistant points by arc length."""
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] == 0:
        return np.repeat(pts[:1], n, axis=0)
    s_new = np.linspace(0, s[-1], n)
    return np.stack([np.interp(s_new, s, pts[:, k]) for k in range(3)], axis=1)


# --------------------------------------------------------------------------- #
# Arc length & curvature
# --------------------------------------------------------------------------- #
def arc_length(pts: np.ndarray) -> float:
    """Cumulative Euclidean arc length of an ordered polyline (mm)."""
    return float(np.linalg.norm(np.diff(np.asarray(pts, float), axis=0), axis=1).sum())


def curvature(pts: np.ndarray) -> np.ndarray:
    """Discrete curvature kappa(s) = ||d2C/ds2|| along an arc-length polyline."""
    p = np.asarray(pts, float)
    ds = np.gradient(np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    ds[ds == 0] = 1e-9
    d1 = np.gradient(p, axis=0) / ds[:, None]
    d2 = np.gradient(d1, axis=0) / ds[:, None]
    return np.linalg.norm(d2, axis=1)


# --------------------------------------------------------------------------- #
# Stage 05: morphometry  (length / width / height in the anatomical frame)
# --------------------------------------------------------------------------- #
def rib_measurements(mask: np.ndarray, affine: np.ndarray, frame: dict,
                     centerline_world: np.ndarray,
                     slab_voxels: float = 1.5) -> dict:
    """Compute the requested rib metrics, all in the anatomical (spinal) frame.

    Returns dict with:
      length_mm        : arc length of the centerline (true rib length)
      max_width_mm     : global left-right (X) extent of the rib
      max_height_mm    : global superior-inferior (Z) extent of the rib
      min_width_mm     : smallest per-cross-section X extent (bone thickness, LR)
      min_height_mm    : smallest per-cross-section Z extent (bone thickness, SI)
      max_depth_mm     : global posterior-anterior (Y) extent (bonus)
      curvature_max    : peak centerline curvature (1/mm) -- anomaly indicator
    """
    out = {
        "length_mm": float("nan"), "max_width_mm": float("nan"),
        "max_height_mm": float("nan"), "min_width_mm": float("nan"),
        "min_height_mm": float("nan"), "max_depth_mm": float("nan"),
        "curvature_max": float("nan"), "n_voxels": int(mask.sum()),
    }
    if mask.sum() == 0:
        return out

    # ---- length from the centerline ----
    out["length_mm"] = arc_length(centerline_world)
    out["curvature_max"] = float(np.nanmax(curvature(centerline_world)))

    # ---- global extents in anatomical frame ----
    pts_w = mask_to_world_points(mask, affine)
    pts_a = world_to_anat(pts_w, frame)               # columns: X(LR) Y(PA) Z(SI)
    out["max_width_mm"] = float(np.ptp(pts_a[:, 0]))
    out["max_depth_mm"] = float(np.ptp(pts_a[:, 1]))
    out["max_height_mm"] = float(np.ptp(pts_a[:, 2]))

    # ---- local cross-sections ----
    # Assign each rib voxel to its nearest centerline sample, then measure the
    # cross-section IN THE PLANE PERPENDICULAR TO THE LOCAL TANGENT.  Measuring
    # raw anatomical-X/Z spread would collapse wherever the rib runs parallel to
    # an anatomical axis, so we build an in-plane basis (u ~ left-right, v ~
    # superior-inferior) orthogonal to the tangent and measure extents there.
    cl_a = world_to_anat(centerline_world, frame)
    tang = np.gradient(cl_a, axis=0)
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12)
    tree = cKDTree(cl_a)
    _, idx = tree.query(pts_a, k=1)
    spacing = float(np.min(voxel_sizes(affine)))

    # Group the dense centerline samples into arc-length "stations" thick enough
    # (~2.5 mm) that each one collects a COMPLETE ring of rib voxels rather than
    # a sub-voxel sliver.  Each voxel inherits the station of its nearest sample.
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cl_a, axis=0), axis=1))]
    station_len = max(2.5, 2.0 * spacing)
    station_of_sample = np.floor(s / station_len).astype(int)
    n_stat = station_of_sample.max() + 1
    vox_station = station_of_sample[idx]

    Xa, Za = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    min_w, min_h = np.inf, np.inf
    lo, hi = int(0.05 * n_stat), int(np.ceil(0.95 * n_stat))  # trim tips
    for st in range(lo, hi):
        sel = vox_station == st
        sect = pts_a[sel]
        if len(sect) < 6:
            continue
        smask = station_of_sample == st
        c = cl_a[smask].mean(axis=0)
        t = tang[smask].mean(axis=0)
        t /= (np.linalg.norm(t) + 1e-12)
        u = Xa - (Xa @ t) * t                      # in-plane, ~ left-right
        if np.linalg.norm(u) < 1e-3:               # tangent ~parallel to X
            u = Za - (Za @ t) * t
        u /= np.linalg.norm(u)
        v = np.cross(t, u)
        v /= (np.linalg.norm(v) + 1e-12)
        rel = sect - c
        ext_u, ext_v = np.ptp(rel @ u), np.ptp(rel @ v)
        # assign the axis more aligned with anatomical X as "width"
        if abs(u @ Xa) >= abs(v @ Xa):
            w, h = ext_u, ext_v
        else:
            w, h = ext_v, ext_u
        if w >= spacing and h >= spacing:
            min_w = min(min_w, w); min_h = min(min_h, h)
    out["min_width_mm"] = float(min_w) if np.isfinite(min_w) else float("nan")
    out["min_height_mm"] = float(min_h) if np.isfinite(min_h) else float("nan")
    return out
