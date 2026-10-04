import os
from os import path as osp

import numpy as np
import matplotlib.pyplot as plt
import torch

from human_body_prior.body_model.body_model import BodyModel

from hp.utils.paths import get_path
from hp.visualization.pv_render import render_mesh

RENDER_WIDTH = 360
RENDER_HEIGHT = 400
dpi = 100

# SMPL-X neutral model, obtain from https://smpl-x.is.tue.mpg.de/downloads
# The folder is set by "body_models" in config/paths.yaml.


def load_body_model(bm_fname: str = None, device: str = "cpu") -> BodyModel:
    if bm_fname is None:
        bm_fname = osp.join(get_path("body_models"), "smplx/SMPLX_NEUTRAL.npz")
    if not osp.exists(bm_fname):
        raise FileNotFoundError(
            f"SMPL-X model not found at {bm_fname}. Download SMPLX_NEUTRAL.npz from "
            "https://smpl-x.is.tue.mpg.de/downloads and set body_models in config/paths.yaml."
        )
    return BodyModel(bm_fname=bm_fname).to(device)


class HumanRender:
    def __init__(self, device: str = "cpu", bm: BodyModel = None) -> None:
        self.device = device
        self.bm = bm if bm is not None else load_body_model(device=device)

    def render_images(self, pose_body, w=None, h=None, azimuths_deg=(0,)) -> np.ndarray:
        """Render each pose of pose_body [N, 63] (or [21, 3]). Returns [N, n_views, h, w, 3]."""
        w = RENDER_WIDTH if w is None else w
        h = RENDER_HEIGHT if h is None else h
        pose_body = torch.as_tensor(pose_body).float().reshape(-1, 63)
        faces = self.bm.f.detach().cpu().numpy()
        images = []
        with torch.no_grad():
            for pose in pose_body:
                v = self.bm(pose_body=pose[None].to(self.device)).v[0].cpu().numpy()
                images.append(render_mesh(v, faces, w, h, azimuths_deg))
        return np.stack(images)

    def render_human(
        self,
        pose_body,
        N_plot: int = 1,
        dpi=150,
        show_img=True,
        w=None,
        h=None,
    ) -> None:
        w = RENDER_WIDTH if w is None else w
        h = RENDER_HEIGHT if h is None else h
        print(f"Rendering with {w}x{h}, DPI={dpi}")
        images = self.render_images(pose_body, w, h)  # [N, 1, h, w, 3]
        img = np.hstack(list(images[:, 0]))
        if show_img:
            fig = plt.figure(figsize=(4 * len(images), 4), dpi=dpi)
            plt.imshow(img)
            plt.axis("off")
        else:
            return img, images

    def get_vertices_from_pose(self, pose_body):
        root_orient = torch.zeros((1, 3)).to(self.device)
        vertices = self.bm(
            root_orient=root_orient, pose_body=pose_body.reshape(1, -1)
        ).v
        return vertices


def render_human(bm, pose_body: np.ndarray, N_plot: int = 1, show_img=True):
    return HumanRender(bm=bm).render_human(pose_body, N_plot=N_plot, show_img=show_img)


def render_human_from_pose_body(pose_body):
    pose_body = torch.tensor(pose_body).float()
    HumanRender().render_human(pose_body[1:])
