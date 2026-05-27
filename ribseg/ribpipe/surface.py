"""High-quality surface extraction from binary masks (VTK, headless/CPU).

Pipeline (the same one 3D Slicer uses for its "closed surface" export, and the
quality recommended in the medical-mesh literature -- Flying Edges + windowed-
sinc smoothing, optionally + quadric decimation):

    binary mask
      -> (optional) Gaussian anti-aliasing of the label field
      -> vtkFlyingEdges3D  (parallel marching cubes, iso = 0.5)
      -> map vertices to world (mm) via the NIfTI affine
      -> vtkWindowedSincPolyDataFilter   (low-pass, shrink-free smoothing)
      -> (optional) vtkQuadricDecimation  (lighten while keeping shape)

Windowed-sinc smoothing is a true low-pass filter on the surface (unlike naive
Laplacian it does not shrink the bone), which is why thin ribs keep their
calibre.  Everything here is CPU-only -- no OpenGL -- so it runs anywhere.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

import vtk
from vtk.util import numpy_support as ns

from .geometry import voxel_to_world


def _polydata_from_arrays(verts: np.ndarray, faces: np.ndarray) -> vtk.vtkPolyData:
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(verts, np.float64), deep=1))
    pd.SetPoints(pts)
    cells = np.empty((len(faces), 4), np.int64)
    cells[:, 0] = 3
    cells[:, 1:] = faces
    ca = vtk.vtkCellArray()
    ca.SetCells(len(faces),
                ns.numpy_to_vtkIdTypeArray(cells.ravel(), deep=1))
    pd.SetPolys(ca)
    return pd


def _arrays_from_polydata(pd: vtk.vtkPolyData):
    n = pd.GetNumberOfPoints()
    if n == 0:
        return np.empty((0, 3)), np.empty((0, 3), int)
    verts = ns.vtk_to_numpy(pd.GetPoints().GetData()).reshape(-1, 3)
    polys = ns.vtk_to_numpy(pd.GetPolys().GetData())
    # polys is [3,i,j,k, 3,i,j,k, ...]
    faces = polys.reshape(-1, 4)[:, 1:].astype(int)
    return verts, faces


def mask_to_surface(mask: np.ndarray, affine: np.ndarray,
                    smooth_iter: int = 20, pass_band: float = 0.10,
                    sigma_vox: float = 0.6, keep_largest: bool = True,
                    decimate: float = 0.0):
    """Smooth world-space surface (verts mm, faces) from a binary mask.

    Parameters
    ----------
    smooth_iter : windowed-sinc iterations (15-30 typical).
    pass_band   : windowed-sinc pass band in [0,2]; smaller = smoother
                  (0.1 is the Slicer-style default for medium smoothing).
    sigma_vox   : Gaussian anti-alias sigma in voxels applied to the label
                  field before iso-surfacing (0 disables).
    decimate    : fraction of triangles to REMOVE (0.0-0.95); 0 keeps all.
    """
    m = mask.astype(bool)
    if m.sum() < 27 or min(m.shape) < 3:
        return np.empty((0, 3)), np.empty((0, 3), int)

    if keep_largest:
        lab, n = ndimage.label(m)
        if n > 1:
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            m = lab == sizes.argmax()
    m = ndimage.binary_fill_holes(m)

    # Crop to the structure's bounding box (+pad) so iso-surfacing only touches
    # the relevant voxels -- a large speed-up on full-FOV volumes.  The crop
    # offset is added back to the vertex indices before mapping to world.
    ijk = np.argwhere(m)
    lo = np.maximum(ijk.min(0) - 2, 0)
    hi = np.minimum(ijk.max(0) + 3, np.array(m.shape))
    m = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    field = m.astype(np.float32)
    if sigma_vox > 0:
        field = ndimage.gaussian_filter(field, sigma_vox)

    # --- vtkImageData in index space (spacing 1) ---
    img = vtk.vtkImageData()
    img.SetDimensions(*field.shape)
    img.GetPointData().SetScalars(
        ns.numpy_to_vtk(field.ravel(order="F"), deep=1,
                        array_type=vtk.VTK_FLOAT))

    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(img); fe.SetValue(0, 0.5)
    fe.ComputeNormalsOff(); fe.ComputeGradientsOff(); fe.Update()
    verts_idx, faces = _arrays_from_polydata(fe.GetOutput())
    if len(verts_idx) == 0:
        return np.empty((0, 3)), np.empty((0, 3), int)

    # index -> world (mm), adding back the crop offset
    verts = voxel_to_world(verts_idx + lo, affine)
    pd = _polydata_from_arrays(verts, faces)

    if smooth_iter > 0:
        sinc = vtk.vtkWindowedSincPolyDataFilter()
        sinc.SetInputData(pd)
        sinc.SetNumberOfIterations(smooth_iter)
        sinc.SetPassBand(pass_band)
        sinc.BoundarySmoothingOff()
        sinc.FeatureEdgeSmoothingOff()
        sinc.NonManifoldSmoothingOn()
        sinc.NormalizeCoordinatesOn()
        sinc.Update()
        pd = sinc.GetOutput()

    if decimate and decimate > 0:
        dec = vtk.vtkQuadricDecimation()
        dec.SetInputData(pd); dec.SetTargetReduction(min(0.95, decimate))
        dec.Update(); pd = dec.GetOutput()

    return _arrays_from_polydata(pd)
