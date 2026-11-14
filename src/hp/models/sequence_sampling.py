"""Motion from a prior of single poses: joint sampling of a whole sequence with RSGM (zero-shot, no new training).

The sequence X = (x_1 ... x_T) is drawn from a product of experts: the pose prior of every frame (possibly conditioned
on the keypoints of that frame) times a Gaussian coupling of neighbouring frames,

    p(X) ∝ Π_t p(x_t | c_t) · exp(-E(X)),
    E(X) = Σ_t |v_t|² / (2 σ_1²)  +  Σ_t |v_{t+1} - v_t|² / (2 σ_2²),     v_t = log(x_t^T x_{t+1}) (per joint),

i.e. small velocities (a random walk) and, optionally, small accelerations. σ_1 and σ_2 may be given per joint, e.g.
the root mean square of the velocities and accelerations of real motion at the frame rate of the sequence.

The sampler is the one of RSGM (the reverse SDE by geodesic Euler-Maruyama, from uniform at t = 1 to t = eps), with
the score of every frame replaced by the score of the product at the noise level of the step,

    score_t(X) = score_θ(x_t, t, c_t) + coupling_t(X),

for all frames in one batch.

**Where to start.** From pure noise (t_start = 1, independent uniform frames) the coupled chain can get trapped: frames
far apart at high noise are pulled together into a smooth curve that winds around SO(3), and only the slowest modes
of the chain can unwind it (with a uniform prior and σ_1 = 0.05 rad, the steps between frames end up 30-40% too large;
with an acceleration term several times too large; with the pose prior of RSGM on a real sequence the visible
keypoints are missed by ~5 cm). The remedy is to start from a coherent sequence: sample_ode of RSGM with the same
start for every frame (a deterministic map of the start and the condition, so slowly changing conditions give slowly
changing poses), noised to t_start ≈ 0.1-0.2 (as SDEdit), and to refine it with the product below t_start.

**The coupling at a noise level.** At diffusion time t every frame carries noise of Brownian time s(t). Near the data
the coupling is a quadratic form: E ≈ ½ yᵀ Q y for small deviations y of the frames (per joint and Lie algebra
coordinate), with Q = D_1ᵀ D_1 / σ_1² + D_2ᵀ D_2 / σ_2² (D_1, D_2: first and second differences in time). For such a
Gaussian chain the prior of the noised frames is Gaussian again, with precision Q (I + s Q)⁻¹, so its score is

    coupling_t = -(I + s(t) Q)⁻¹ ∇E,

the gradient of the coupling filtered along time: modes of Q with eigenvalue λ are damped by 1 / (1 + s λ). Fast
modes (jitter) are free at high noise, slow modes (the course of the motion) stay coupled at every noise level. This
is exact for a Gaussian chain in a vector space; on SO(3) and together with the pose prior it is the usual
approximation of composing diffusion models (the noised product is not the product of the noised experts), so the
samples are close to, but not exactly from, p(X). The filter is applied with the eigenvectors of Q, computed once.

**Stiffness.** The coupling's gradient grows like 1/σ², far beyond the step size of the sampler near the data, so
every step is split: the step of the sampler with the score of the pose prior, then the flow of the coupling over the
same time β(t) dt, in as many explicit substeps as its stability needs (below 2 / L, with L the largest eigenvalue
λ / (1 + s λ) of the filtered coupling).

∇E is taken with autograd in the frame of every x_t (the Riemannian gradient in Lie algebra coordinates, as the
score). In the acceleration term v_{t+1} (in the frame of x_{t+1}) and v_t (in the frame of x_t) are compared
directly: turning v_{t+1} into the frame of x_t (R_t v_{t+1}, R_t = exp(v_t)) gives the same energy, because
(R_t v_{t+1}) · v_t = v_{t+1} · (R_tᵀ v_t) and R_t leaves its own axis v_t fixed. The filter acts on the gradients in
world coordinates (x_t g_t), where the tangent vectors of different frames are comparable.
"""
import math
from typing import Optional, Union

import roma
import torch

from hp.models import igso3
from hp.models.rsgm import RSGM, exp_frame, move

Sigma = Union[float, torch.Tensor]  # a number, or one value per joint [J]


def velocities(x: torch.Tensor) -> torch.Tensor:
    """v_t = log(x_t^T x_{t+1}) [T - 1, J, 3] of a sequence x [T, J, 3, 3]: the rotation from one frame to the next, in
    the frame of the first."""
    return roma.rotmat_to_rotvec(x[:-1].transpose(-1, -2) @ x[1:])


def coupling_energy(x: torch.Tensor, sigma1: Sigma, sigma2: Optional[Sigma] = None) -> torch.Tensor:
    """E(X) of the module doc, for x [T, J, 3, 3]; sigma1, sigma2 numbers or [J] (radians per frame)."""
    v = velocities(x)                                                    # [T - 1, J, 3]
    sigma1 = torch.as_tensor(sigma1, dtype=x.dtype, device=x.device)
    energy = (v.pow(2).sum(-1) / (2 * sigma1**2)).sum()
    if sigma2 is not None and len(v) > 1:
        sigma2 = torch.as_tensor(sigma2, dtype=x.dtype, device=x.device)
        energy = energy + ((v[1:] - v[:-1]).pow(2).sum(-1) / (2 * sigma2**2)).sum()
    return energy


def coupling_gradient(x: torch.Tensor, sigma1: Sigma, sigma2: Optional[Sigma] = None) -> torch.Tensor:
    """∇E [T, J, 3] in the frame of every x_t: d/dδ E(x_t exp(hat(δ_t))) at δ = 0."""
    with torch.enable_grad():
        delta = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device, requires_grad=True)
        energy = coupling_energy(exp_frame(x.detach(), delta), sigma1, sigma2)
        (gradient,) = torch.autograd.grad(energy, delta)
    return torch.nan_to_num(gradient)  # a frame equal to its neighbour has no defined direction


def coupling_spectrum(frames: int, joints: int, sigma1: Sigma, sigma2: Optional[Sigma] = None, dtype=torch.float64,
                      device="cpu"):
    """Eigenvalues [J, T] and eigenvectors [J, T, T] of the precision Q of the coupling along time, per joint."""
    eye = torch.eye(frames, dtype=dtype, device=device)
    d1 = eye[1:] - eye[:-1]                                              # [T - 1, T]
    d2 = d1[1:] - d1[:-1]                                                # [T - 2, T]
    s1 = torch.as_tensor(sigma1, dtype=dtype, device=device).expand(joints)
    q = (d1.T @ d1)[None] / s1[:, None, None] ** 2                       # [J, T, T]
    if sigma2 is not None and frames > 2:
        s2 = torch.as_tensor(sigma2, dtype=dtype, device=device).expand(joints)
        q = q + (d2.T @ d2)[None] / s2[:, None, None] ** 2
    eigenvalues, eigenvectors = torch.linalg.eigh(q)
    return eigenvalues.clamp_min(0), eigenvectors


def filtered(gradient: torch.Tensor, eigenvalues: torch.Tensor, eigenvectors: torch.Tensor, s: float) -> torch.Tensor:
    """(I + s Q)⁻¹ ∇E along time, per joint and coordinate: gradient [T, J, 3]."""
    coefficients = torch.einsum("jtk,tjc->jkc", eigenvectors, gradient) / (1 + s * eigenvalues)[..., None]
    return torch.einsum("jtk,jkc->tjc", eigenvectors, coefficients)


@torch.no_grad()
def sample_sequence(model: RSGM, frames: int, cond: Optional[torch.Tensor] = None, sigma1: Sigma = 0.05,
                    sigma2: Optional[Sigma] = None, steps: Optional[int] = None, start: Optional[torch.Tensor] = None,
                    t_start: float = 1.0) -> torch.Tensor:
    """A sequence [frames, J, 3, 3] from the product of the module doc. cond: [frames, cond_dim] for a conditional
    model (NaN keypoints: not observed in that frame). sigma1, sigma2: the coupling (radians per frame, numbers or per
    joint; sigma2 None: no acceleration term). steps: of the reverse SDE from t_start (default: the model's
    sample_steps). start: the uniform rotations at t = 1 [frames, J, 3, 3] (default: drawn).
    t_start < 1: start from a given sequence instead (e.g. a coherent one of sample_ode with a shared start), noised
    with IGSO(3) to time t_start, and run the sampler from there (as SDEdit): the coarse course of the motion is kept,
    the product refines it at the noise levels below t_start."""
    model.check_cond(cond, frames)
    dtype, device, joints = model.net[0].weight.dtype, model.device, model.num_joints
    f = model.cond_features(cond, frames)
    if t_start < 1.0:
        if start is None:
            raise ValueError("t_start < 1 needs a start sequence to noise.")
        s0 = model.s(torch.tensor(t_start, dtype=torch.float64)).expand(frames, joints)
        noise = model.angles.sample(s0)[..., None] * igso3.random_axes((frames, joints))
        start = exp_frame(move(start.reshape(frames, joints, 3, 3), "cpu", torch.float64), noise)
    elif start is None:
        start = roma.random_rotmat(frames * joints, device=device, dtype=dtype)
    x = move(start.reshape(frames, joints, 3, 3), device, dtype)
    times = torch.linspace(t_start, model.eps, (steps or model.sample_steps) + 1, device=device, dtype=dtype)
    # the spectrum in float64 on the cpu (mps has no float64 and eigh is small): then on the device in the model's dtype
    eigenvalues, eigenvectors = coupling_spectrum(frames, joints, sigma1, sigma2)
    largest = eigenvalues.max().item()
    eigenvalues, eigenvectors = move(eigenvalues, device, dtype), move(eigenvectors, device, dtype)

    def coupling_flow(x, t, duration):
        # x moved along the filtered coupling at time t for the time `duration`, in stable explicit substeps
        s = float(model.s(t))
        curvature = largest / (1 + s * largest)
        substeps = max(1, int(math.ceil(float(duration) * curvature)))
        for _ in range(substeps):
            # filtered in world coordinates (x_t g_t), where the tangent vectors of different frames are comparable
            g = (x @ coupling_gradient(x, sigma1, sigma2)[..., None])[..., 0]
            g = (x.transpose(-1, -2) @ filtered(g, eigenvalues, eigenvectors, s)[..., None])[..., 0]
            x = exp_frame(x, -(float(duration) / substeps) * g)
        return x

    for t, t_next in zip(times[:-1], times[1:]):
        dt, beta = t - t_next, model.beta(t)
        noise = torch.randn(frames, joints, 3, device=device, dtype=dtype)
        x = exp_frame(x, beta * model.score(x, t.expand(frames), f) * dt + torch.sqrt(beta * dt) * noise)
        x = coupling_flow(x, t, beta * dt)
    return x
