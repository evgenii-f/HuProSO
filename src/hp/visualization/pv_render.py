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


def render_overlay(
    vertices: np.ndarray,
    faces: np.ndarray,
    opacities=None,
    width: int = 360,
    height: int = 400,
    azimuths_deg=(0,),
    color=BODY_COLOR,
    background="white",
    center=None,
    radius=None,
    points=None,
    point_color="red",
    point_opacity=0.5,
    point_radius=0.05,
) -> np.ndarray:
    """Render several meshes of the same body in one scene, each with its own opacity (as Fig. 3 of the HuProSO3
    paper: the samples on top of each other, the less likely ones more transparent).

    vertices: [M, V, 3] (M meshes), y-up, body facing +z. faces: [F, 3]. opacities: [M] in (0, 1], default 1.
    center, radius: the framing; default: the bounding box of all meshes, so that every view of every call with the
    same center and radius has the same scale. points: [P, 3] positions (in the frame of the vertices, e.g. joints)
    drawn as dots (spheres of point_radius, meters) on top of the meshes with point_opacity, e.g. to mark joints: they
    are rendered alone with the same camera and blended over the picture, so a joint inside the body is seen too.
    Returns uint8 images [len(azimuths_deg), height, width, 3].
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim == 2:
        vertices = vertices[None]
    opacities = np.ones(len(vertices)) if opacities is None else np.asarray(opacities, dtype=np.float64)
    if len(opacities) != len(vertices):
        raise ValueError(f"{len(vertices)} meshes but {len(opacities)} opacities.")
    pv_faces = np.hstack([np.full((len(faces), 1), 3), np.asarray(faces)]).ravel()
    lo, hi = vertices.reshape(-1, 3).min(0), vertices.reshape(-1, 3).max(0)
    center = (hi + lo) / 2 if center is None else np.asarray(center, dtype=np.float64)
    radius = np.linalg.norm(hi - lo) / 2 if radius is None else float(radius)

    images = []
    for az in azimuths_deg:
        a = np.deg2rad(az)
        rot_y = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        pl = pv.Plotter(off_screen=True, window_size=(width, height))
        pl.set_background(background)
        pl.enable_depth_peeling()  # correct order of the translucent surfaces
        for v, opacity in zip(vertices, opacities):
            pl.add_mesh(pv.PolyData((v - center) @ rot_y.T, pv_faces), color=color, smooth_shading=True, opacity=float(opacity))
        camera = [(0, 0, 4.6 * radius), (0, 0, 0), (0, 1, 0)]
        pl.camera_position = camera
        image = pl.screenshot(return_img=True)
        pl.close()
        marks = np.zeros((0, 3)) if points is None else np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(marks):  # the spheres alone, with the same camera, blended over the picture
            pl = pv.Plotter(off_screen=True, window_size=(width, height))
            pl.set_background("white")
            for p in marks:
                pl.add_mesh(pv.Sphere(radius=point_radius, center=(p - center) @ rot_y.T), color="black")
            pl.camera_position = camera
            alpha = point_opacity * (pl.screenshot(return_img=True).mean(-1, keepdims=True) < 128)
            pl.close()
            rgb = np.array(pv.Color(point_color).int_rgb, dtype=np.float64)
            image = (image * (1 - alpha) + rgb * alpha).astype(np.uint8)
        images.append(image)
    return np.stack(images)
