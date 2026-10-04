"""Offscreen mesh rendering with PyVista (VTK). Works on macOS / Apple Silicon
without OSMesa or EGL, unlike the pyrender backend of body_visualizer."""
import numpy as np
import pyvista as pv

BODY_COLOR = (0.75, 0.75, 0.75)


def render_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    width: int = 360,
    height: int = 400,
    azimuths_deg=(0,),
    color=BODY_COLOR,
    background="white",
) -> np.ndarray:
    """Render one mesh from several viewpoints.

    vertices: [V, 3], y-up, body facing +z (SMPL convention).
    faces: [F, 3]
    Returns uint8 images of shape [len(azimuths_deg), height, width, 3].
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    pv_faces = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    center = (vertices.max(0) + vertices.min(0)) / 2
    radius = np.linalg.norm(vertices.max(0) - vertices.min(0)) / 2

    images = []
    for az in azimuths_deg:
        a = np.deg2rad(az)
        rot_y = np.array(
            [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
        )
        mesh = pv.PolyData((vertices - center) @ rot_y.T, pv_faces)
        pl = pv.Plotter(off_screen=True, window_size=(width, height))
        pl.set_background(background)
        pl.add_mesh(mesh, color=color, smooth_shading=True)
        pl.camera_position = [(0, 0, 4.6 * radius), (0, 0, 0), (0, 1, 0)]
        images.append(pl.screenshot(return_img=True))
        pl.close()
    return np.stack(images)
