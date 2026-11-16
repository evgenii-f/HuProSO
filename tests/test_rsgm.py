"""RSGM on SO(3)^J (hp/models/rsgm.py).

The main checks put the exact score of a known distribution into the model (an "oracle"): data from IGSO(3) around
a mean mu with time s0, so p_t is IGSO(3) around mu with time s0 + s(t). Then log_prob must give the analytic
density and sample() must draw from it. That tests the ODE, the divergence, the base measure and the sampler
without any training.
"""
import math

import pytest
import roma
import torch

from hp.models import build_model, igso3
from hp.models.rsgm import RSGM, exp_frame
from test_igso3 import ks_distance


def log_map(r: torch.Tensor) -> torch.Tensor:
    """Rotation vectors of rotations [..., 3, 3] with angles in (0, pi), in plain ops (torch.func can vmap it)."""
    angle = torch.arccos(((r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
    axis = torch.stack([r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]], -1)
    return axis / axis.norm(dim=-1, keepdim=True) * angle[..., None]


class Oracle(RSGM):
    """RSGM with the exact score of data from IGSO(3) around mu with time s0."""

    s0 = 0.1

    def __init__(self, num_joints, **kwargs):
        super().__init__(num_joints, hidden=[4], **kwargs)
        self.double()
        self.mu = roma.random_rotmat(num_joints, generator=torch.Generator().manual_seed(1)).double()

    def score(self, x, t, f=None):
        v = log_map(self.mu.transpose(-1, -2) @ x)
        angle = v.norm(dim=-1)
        return (igso3.d_log_density(angle, self.s0 + self.s(t)[:, None]) / angle)[..., None] * v

    def data(self, n, seed=0):
        """n poses from the oracle's distribution and their angles to mu."""
        generator = torch.Generator().manual_seed(seed)
        angles = igso3.AngleSampler(1e-3, 1.0).sample(torch.full((n, self.num_joints), self.s0), generator)
        poses = exp_frame(self.mu.expand(n, -1, 3, 3), angles[..., None] * igso3.random_axes((n, self.num_joints), generator))
        return poses, angles


def test_log_prob_of_the_oracle_is_the_analytic_density():
    # beta_f = 20: p_1 is uniform to 1e-8, so the only error left is the ODE solver's
    model = Oracle(2, beta_f=20.0, likelihood_steps=100)
    poses, angles = model.data(300)
    exact = igso3.log_density(angles, model.s0 + model.s(torch.tensor(model.eps))).sum(-1)
    error = model.log_prob(poses) - exact
    assert error.mean().abs() < 0.01 and error.abs().max() < 0.1


def test_with_the_default_schedule_the_likelihood_is_off_only_by_the_base_mismatch():
    """beta_f = 6 (the JAX so3 experiment): p_1 is not quite uniform, which costs about 0.01 nats per joint."""
    model = Oracle(2, likelihood_steps=100)
    poses, angles = model.data(300)
    exact = igso3.log_density(angles, model.s0 + model.s(torch.tensor(model.eps))).sum(-1)
    assert (model.log_prob(poses) - exact).mean().abs() < 0.05


def test_samples_of_the_oracle_follow_its_distribution():
    model = Oracle(2, beta_f=20.0, sample_steps=200)
    torch.manual_seed(0)
    angles = log_map(model.mu.transpose(-1, -2) @ model.sample(10_000)).norm(dim=-1)
    for joint in range(2):
        assert ks_distance(angles[:, joint], model.s0) < 1.63 / math.sqrt(10_000)


# The ODE with Euler needs ~1000 steps here (Euler-Maruyama 200), and more with "edm": rho = 7 spends the steps on
# noise levels far below the scale of the oracle's data (s0 = 0.1), while Heun with "edm" passes with 20.
@pytest.mark.parametrize("sampler, schedule, steps", [("heun", "uniform", 50), ("heun", "edm", 20), ("euler", "uniform", 1000)])
def test_the_other_samplers_of_the_oracle_follow_its_distribution(sampler, schedule, steps):
    model = Oracle(2, beta_f=20.0, sample_steps=steps, sampler=sampler, schedule=schedule)
    torch.manual_seed(0)
    angles = log_map(model.mu.transpose(-1, -2) @ model.sample(10_000)).norm(dim=-1)
    for joint in range(2):
        assert ks_distance(angles[:, joint], model.s0) < 1.63 / math.sqrt(10_000)


def test_heun_is_second_order():
    """Without noise the error of the end point against a fine solution falls as h^2 for Heun and as h for Euler."""
    model = Oracle(2, beta_f=20.0)  # one oracle: a new one has another mu

    def end(sampler, steps):
        model.sampler, model.sample_steps = sampler, steps
        torch.manual_seed(0)
        return model.sample(200)

    reference = end("heun", 1000)
    for sampler, order in (("heun", 2), ("euler", 1)):
        errors = [(log_map(reference.transpose(-1, -2) @ end(sampler, steps)).norm(dim=-1)).mean() for steps in (20, 40)]
        assert abs(math.log2(errors[0] / errors[1]) - order) < 0.35


def test_the_edm_schedule_runs_from_1_to_eps_with_the_small_steps_at_the_end():
    model = RSGM(2, hidden=[4], schedule="edm")
    times = model.sample_times(10)
    steps = -times.diff()
    assert times[0] == 1.0 and times[-1] == model.eps and (steps > 0).all() and steps[-1] < steps[0] / 100
    assert torch.allclose(model.s(model.t_of_s(model.s(times))), model.s(times))


def test_the_sampler_settings_are_checked_and_counted():
    assert RSGM(2, hidden=[4], sampler="heun", sample_steps=10).sample_nfe == 20
    assert RSGM(2, hidden=[4], sample_steps=10).sample_nfe == 10
    with pytest.raises(ValueError):
        RSGM(2, hidden=[4], sampler="rk4")
    with pytest.raises(ValueError):
        RSGM(2, hidden=[4], schedule="cosine")


def test_ode_samples_of_the_oracle_follow_its_distribution_and_are_a_map_of_the_start():
    model = Oracle(2, beta_f=20.0, sample_steps=100)
    torch.manual_seed(0)
    start = roma.random_rotmat(10_000 * 2).double().reshape(10_000, 2, 3, 3)
    poses = model.sample_ode(10_000, start=start)
    angles = log_map(model.mu.transpose(-1, -2) @ poses).norm(dim=-1)
    for joint in range(2):
        assert ks_distance(angles[:, joint], model.s0) < 1.63 / math.sqrt(10_000)
    assert torch.equal(model.sample_ode(5, start=start[:5]), poses[:5])  # deterministic


def test_the_target_of_the_loss_is_the_riemannian_gradient_of_the_heat_kernel():
    """d log f / d omega * u, in the frame of x_t, against autograd of log f(angle(x0^T x_t exp(hat(delta))))."""
    torch.manual_seed(0)
    s = torch.tensor(0.3, dtype=torch.float64)
    x0 = roma.random_rotmat(5).double()
    omega = torch.rand(5, dtype=torch.float64) * 2 + 0.2
    axis = igso3.random_axes((5,))
    xt = exp_frame(x0, omega[:, None] * axis)
    delta = torch.zeros(5, 3, dtype=torch.float64, requires_grad=True)
    angle = log_map(x0.transpose(-1, -2) @ exp_frame(xt, delta)).norm(dim=-1)
    igso3.log_density(angle, s).sum().backward()
    assert torch.allclose(delta.grad, igso3.d_log_density(omega, s)[:, None] * axis, atol=1e-8)


def test_the_divergence_is_the_one_of_finite_differences():
    torch.manual_seed(0)
    model = build_model("rsgm", num_joints=3, hidden=[32, 32]).double()
    x, t = roma.random_rotmat(6 * 3).double().reshape(6, 3, 3, 3), torch.rand(6, dtype=torch.float64)
    with torch.no_grad():
        _, divergence = model.velocity_and_divergence(x, t)
        numeric = torch.zeros(6, dtype=torch.float64)
        for joint in range(3):
            for i in range(3):
                e = torch.zeros(6, 3, 3, dtype=torch.float64)
                e[:, joint, i] = 1e-5
                numeric += (model.velocity(exp_frame(x, e), t) - model.velocity(exp_frame(x, -e), t))[:, joint, i] / 2e-5
    assert torch.allclose(divergence, numeric, atol=1e-6)


def test_the_validation_loss_is_the_same_every_time_and_the_training_loss_is_not():
    model = build_model("rsgm", num_joints=2, hidden=[16])
    pose = roma.random_rotmat(16 * 2).reshape(16, 2, 3, 3)
    model.eval()
    assert model.loss(pose).item() == model.loss(pose).item()
    model.train()
    assert model.loss(pose).item() != model.loss(pose).item()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs mps")
def test_the_training_loss_stays_finite_on_mps():
    """On mps, a copy that changes device and dtype in one .to() sometimes returns garbage (torch 2.13)."""
    torch.manual_seed(0)
    model = build_model("rsgm", num_joints=19, hidden=[64]).to("mps")
    pose = roma.random_rotmat(256 * 19).reshape(256, 19, 3, 3).to("mps")
    assert all(torch.isfinite(model.loss(pose)) for _ in range(100))


def test_likelihood_weighting_draws_the_times_with_density_beta_over_std_squared():
    model = build_model("rsgm", num_joints=1, hidden=[8], beta_f=10.0, weighting="likelihood")
    t = model.loss_times(torch.rand(50_000, dtype=torch.float64, generator=torch.Generator().manual_seed(0))).sort().values
    assert t.min() >= model.loss_t_min - 1e-12 and t.max() <= 1 + 1e-12
    # the CDF of q(t) ~ beta / std^2 is log(exp(s(t)) - 1), normalized
    log_em1 = lambda tt: torch.log(torch.expm1(model.s(torch.as_tensor(tt, dtype=torch.float64))))
    cdf = (log_em1(t) - log_em1(model.loss_t_min)) / (log_em1(1.0) - log_em1(model.loss_t_min))
    empirical = torch.arange(1, len(t) + 1, dtype=torch.float64) / len(t)
    assert (cdf - empirical).abs().max() < 1.63 / math.sqrt(len(t))
    std = build_model("rsgm", num_joints=1, hidden=[8], beta_f=10.0)
    assert torch.equal(std.loss_times(torch.tensor([0.0, 1.0], dtype=torch.float64)), torch.tensor([std.loss_t_min, 1.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="weighting"):
        build_model("rsgm", num_joints=1, weighting="uniform")


# ---- conditioning on keypoints ----

from hp.data.amass_cfg import AmassJoints4Learning  # noqa: E402
from hp.data.conditions import keypoints  # noqa: E402
from hp.models.io import load_model, save_model  # noqa: E402

J19 = len(AmassJoints4Learning)


def poses19(n, seed=0, dtype=torch.float32):
    return roma.random_rotmat(n * J19, generator=torch.Generator().manual_seed(seed), dtype=dtype).reshape(n, J19, 3, 3)


def test_an_unconditional_model_has_no_condition_weights():
    model = build_model("rsgm", num_joints=2, hidden=[8])
    assert model.cond_dim == 0 and not any(k.startswith("cond_net") for k in model.state_dict())
    with pytest.raises(ValueError):
        model.loss(roma.random_rotmat(2 * 2).reshape(2, 2, 3, 3), torch.zeros(2, 66))


def test_the_condition_sets_cond_dim_and_bad_settings_are_refused():
    assert build_model("rsgm", num_joints=J19, hidden=[8], condition="3D").cond_dim == 66
    assert build_model("rsgm", num_joints=J19, hidden=[8], condition="2D").cond_dim == 44
    with pytest.raises(ValueError):
        build_model("rsgm", num_joints=J19, hidden=[8], cond_dim=66, cond_mask="random")  # a mask needs keypoints
    with pytest.raises(ValueError):
        build_model("rsgm", num_joints=J19, hidden=[8], condition="3D", cond_dim=44)
    with pytest.raises(ValueError):
        build_model("rsgm", num_joints=J19, hidden=[8], condition="3D", cond_mask="joints")


def test_without_a_condition_the_loss_takes_the_keypoints_of_the_poses():
    torch.manual_seed(0)
    model = build_model("rsgm", num_joints=J19, hidden=[16], condition="3D").eval()
    pose = poses19(32)
    assert model.loss(pose).item() == model.loss(pose, keypoints(pose).reshape(32, -1)).item()
    assert model.loss(pose).item() != model.loss(pose, keypoints(poses19(32, seed=1)).reshape(32, -1)).item()


def test_a_hidden_joint_is_its_mask_bit_whatever_its_keypoint():
    torch.manual_seed(0)
    model = build_model("rsgm", num_joints=J19, hidden=[16], condition="3D")
    c = keypoints(poses19(4)).reshape(4, 22, 3)
    a, b = c.clone(), c.clone()
    a[:, 5], b[:, 5] = float("nan"), 0.0
    b[:, 5, 0] = float("nan")  # one NaN coordinate hides the whole joint
    fa, fb = model.cond_features(a.reshape(4, -1), 4), model.cond_features(b.reshape(4, -1), 4)
    assert torch.isfinite(fa).all() and torch.equal(fa, fb)
    assert not torch.equal(fa, model.cond_features(c.reshape(4, -1), 4))


def test_random_masks_hide_half_of_the_joints_on_average(monkeypatch):
    model = build_model("rsgm", num_joints=J19, hidden=[16], condition="3D", cond_mask="random").eval()
    seen = []
    features = model.cond_features
    monkeypatch.setattr(model, "cond_features", lambda cond, batch: seen.append(cond.clone()) or features(cond, batch))
    model.loss(poses19(2000))
    hidden = torch.isnan(seen[0].reshape(2000, 22, 3)).all(-1)
    assert abs(hidden.float().mean().item() - 0.5) < 0.02
    # the share of hidden joints varies from pose to pose (p ~ U(0, 1)), not around a fixed p
    assert hidden.float().mean(1).std() > 0.25


def test_sample_and_log_prob_need_the_condition_and_depend_on_it():
    torch.manual_seed(0)
    model = build_model("rsgm", num_joints=J19, hidden=[16], condition="3D", sample_steps=5, likelihood_steps=3).eval()
    pose = poses19(3)
    c, other = keypoints(pose).reshape(3, -1), keypoints(poses19(3, seed=1)).reshape(3, -1)
    assert model.sample(3, c).shape == (3, J19, 3, 3)
    with pytest.raises(ValueError):
        model.sample(3)
    lp = model.log_prob(pose, c)
    assert lp.shape == (3,) and torch.isfinite(lp).all() and not torch.allclose(lp, model.log_prob(pose, other))


def test_the_conditional_divergence_is_the_one_of_finite_differences():
    torch.manual_seed(0)
    model = build_model("rsgm", num_joints=J19, hidden=[32], condition="2D").double()
    x, t = poses19(4, dtype=torch.float64), torch.rand(4, dtype=torch.float64)
    f = model.cond_features(keypoints(poses19(4, seed=2, dtype=torch.float64), "2D").reshape(4, -1), 4)
    with torch.no_grad():
        _, divergence = model.velocity_and_divergence(x, t, f)
        numeric = torch.zeros(4, dtype=torch.float64)
        for joint in range(J19):
            for i in range(3):
                e = torch.zeros(4, J19, 3, dtype=torch.float64)
                e[:, joint, i] = 1e-5
                numeric += (model.velocity(exp_frame(x, e), t, f) - model.velocity(exp_frame(x, -e), t, f))[:, joint, i] / 2e-5
    assert torch.allclose(divergence, numeric, atol=1e-6)


def test_the_parents_are_those_of_the_skeleton_file():
    from hp.data.conditions import _skeleton
    from hp.models.rsgm import SMPL_PARENTS
    try:
        parents, _ = _skeleton()
    except FileNotFoundError:
        pytest.skip("needs body_models/smpl_skeleton.npz")
    assert parents[1:] == SMPL_PARENTS[1:]


def test_a_global_rotation_is_the_product_along_the_chain_to_the_root():
    from hp.models.rsgm import global_rotations
    pose = poses19(5, dtype=torch.float64)
    names = [joint.name for joint in AmassJoints4Learning]
    g = global_rotations(pose)
    # LeftHand <- LeftElbow <- LeftArm <- LeftShoulder <- Spine2 <- Spine1 <- Spine0 <- root
    chain = ["Spine0", "Spine1", "Spine2", "LeftShoulder", "LeftArm", "LeftElbow", "LeftHand"]
    product = pose[:, names.index(chain[0])]
    for name in chain[1:]:
        product = product @ pose[:, names.index(name)]
    assert torch.allclose(g[:, names.index("LeftHand")], product)
    assert torch.equal(g[:, names.index("LeftUpLeg")], pose[:, names.index("LeftUpLeg")])  # a child of the root


def test_global_inputs_and_time_features_keep_the_divergence_exact():
    torch.manual_seed(0)
    model = build_model("rsgm", num_joints=J19, hidden=[32], inputs="local+global", time_features=4).double()
    assert model.net[0].in_features == 2 * 9 * J19 + 1 + 8
    x, t = poses19(3, dtype=torch.float64), torch.rand(3, dtype=torch.float64)
    with torch.no_grad():
        _, divergence = model.velocity_and_divergence(x, t)
        numeric = torch.zeros(3, dtype=torch.float64)
        for joint in range(J19):
            for i in range(3):
                e = torch.zeros(3, J19, 3, dtype=torch.float64)
                e[:, joint, i] = 1e-5
                numeric += (model.velocity(exp_frame(x, e), t) - model.velocity(exp_frame(x, -e), t))[:, joint, i] / 2e-5
    assert torch.allclose(divergence, numeric, atol=1e-6)
    with pytest.raises(ValueError, match="inputs"):
        build_model("rsgm", num_joints=3, inputs="local+global")
    with pytest.raises(ValueError, match="inputs"):
        build_model("rsgm", num_joints=J19, inputs="global")


def test_a_conditional_model_is_saved_and_loaded(tmp_path):
    torch.manual_seed(0)
    model = build_model("rsgm", num_joints=J19, hidden=[16], condition="3D", cond_mask="random")
    save_model(model, tmp_path)
    loaded = load_model(tmp_path)
    assert loaded.condition == "3D" and loaded.cond_mask == "random"
    c = keypoints(poses19(2)).reshape(2, -1)
    assert torch.equal(model.cond_features(c, 2), loaded.cond_features(c, 2))


def test_the_timeres_net_starts_at_a_zero_score_and_keeps_the_shapes():
    import roma
    model = RSGM(19, net="timeres", net_width=64, net_blocks=2, net_embed=32).eval()
    x = roma.random_rotmat(3 * 19).view(3, 19, 3, 3)
    t = torch.tensor([0.003, 0.1, 0.9])
    assert model.score(x, t).shape == (3, 19, 3)
    assert torch.equal(model.score(x, t), torch.zeros(3, 19, 3))  # the last layer is zero at the start
    assert model.dtype == torch.float32


def test_the_timeres_net_learns_and_samples_and_has_a_likelihood():
    import roma
    torch.manual_seed(0)
    model = RSGM(19, net="timeres", net_width=64, net_blocks=1, net_embed=32, dropout=0.1, sample_steps=5,
                 likelihood_steps=5, inputs="local+global")
    poses = roma.random_rotmat(64 * 19).view(64, 19, 3, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    first = None
    for _ in range(30):
        loss = model.loss(poses)
        first = first if first is not None else float(loss)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert float(loss) < first
    model.eval()
    samples = model.sample(4)
    assert samples.shape == (4, 19, 3, 3) and torch.isfinite(samples).all()
    assert torch.isfinite(model.log_prob(poses[:3])).all()


def test_the_timeres_net_is_saved_and_loaded(tmp_path):
    from hp.models import build_model
    from hp.models.io import load_model, save_model
    import roma
    model = build_model("rsgm", num_joints=19, net="timeres", net_width=32, net_blocks=1, net_embed=16).eval()
    torch.nn.init.normal_(model.net.last.weight)  # a non-trivial score
    save_model(model, tmp_path)
    loaded = load_model(tmp_path)
    x, t = roma.random_rotmat(2 * 19).view(2, 19, 3, 3), torch.tensor([0.01, 0.5])
    assert loaded.net_kind == "timeres" and torch.allclose(loaded.score(x, t), model.score(x, t))


def test_the_timeres_net_refuses_time_features():
    import pytest
    with pytest.raises(ValueError):
        RSGM(19, net="timeres", time_features=4)
    with pytest.raises(ValueError):
        RSGM(19, net="transformer")


def test_timeres_with_every_component_off_is_a_plain_mlp_with_the_level_at_the_input():
    from hp.models.score_nets import TimeResNet
    torch.manual_seed(0)
    net = TimeResNet(20, 6, width=16, blocks=2, time_in="input", time_embed="scalar", residual=False, norm=False,
                     zero_init=False, dropout=0.0)
    layers = [net.first] + [layer for block in net.blocks for layer in block]
    assert all(layer.dense_t is None and isinstance(layer.norm, torch.nn.Identity) for layer in layers)
    mlp = []
    for layer in layers:
        mlp += [layer.dense, torch.nn.SiLU()]
    mlp = torch.nn.Sequential(*mlp, net.last)
    x, level = torch.randn(5, 20), torch.rand(5)
    assert torch.allclose(net(x, level), mlp(torch.cat([x, level[:, None]], 1)), atol=1e-6)


def test_timeres_switches_change_only_their_component():
    from hp.models.score_nets import TimeResNet
    full = TimeResNet(20, 6, width=16, blocks=1)
    assert full.first.dense_t is not None and isinstance(full.first.norm, torch.nn.LayerNorm) and full.residual
    assert torch.equal(full.last.weight, torch.zeros_like(full.last.weight))
    assert TimeResNet(20, 6, width=16, blocks=1, time_in="input").first.dense.in_features == 20 + 256
    assert TimeResNet(20, 6, width=16, blocks=1, time_embed="scalar").first.dense_t.in_features == 1
    assert TimeResNet(20, 6, width=16, blocks=1, zero_init=False).last.weight.abs().sum() > 0
    model = RSGM(19, net="timeres", net_width=16, net_blocks=1, net_level="t")
    x = __import__("roma").random_rotmat(2 * 19).view(2, 19, 3, 3)
    assert model.score(x, torch.tensor([0.01, 0.5])).shape == (2, 19, 3)
    import pytest
    with pytest.raises(ValueError):
        RSGM(19, net="timeres", net_level="sigma")
    with pytest.raises(ValueError):
        TimeResNet(20, 6, time_in="everywhere")


# ---- the z-scored tangent input (inputs="local+z") ----

from hp.models.rsgm import z_statistics  # noqa: E402


def concentrated_poses(n, seed=0):
    """Poses around a random mean M_j with a different spread per joint and per axis (in the frame of M_j)."""
    generator = torch.Generator().manual_seed(seed)
    means = roma.random_rotmat(J19, generator=generator, dtype=torch.float64)
    scale = torch.linspace(0.05, 0.6, J19, dtype=torch.float64)[:, None] * torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64)
    v = torch.randn(n, J19, 3, generator=generator, dtype=torch.float64) * scale
    return exp_frame(means.expand(n, -1, 3, 3), v), means, scale


def test_the_z_statistics_are_the_karcher_mean_and_the_spread_around_it():
    poses, means, scale = concentrated_poses(20_000)
    z_mean, z_std = z_statistics(poses, chunk=7_000)
    m = roma.rotvec_to_rotmat(torch.tensor(z_mean, dtype=torch.float64).reshape(J19, 3))
    logs = roma.rotmat_to_rotvec(m.transpose(-1, -2) @ poses)
    assert logs.mean(0).abs().max() < 1e-9  # the Karcher mean: the logs average to zero
    assert roma.rotmat_geodesic_distance(m, means).max() < 0.02  # near the mean of the generating distribution
    assert torch.allclose(torch.tensor(z_std, dtype=torch.float64).reshape(J19, 3), scale, rtol=0.05)


def z_model(seed=0, **kwargs):
    torch.manual_seed(seed)
    poses, _, _ = concentrated_poses(2_000, seed)
    z_mean, z_std = z_statistics(poses)
    return build_model("rsgm", num_joints=J19, inputs="local+z", z_mean=z_mean, z_std=z_std, **kwargs), poses


def test_the_z_input_is_z_scored_and_kept_as_buffers(tmp_path):
    model, poses = z_model(hidden=[16])
    assert model.net[0].in_features == 9 * J19 + 3 * J19 + 1
    assert {"z_rot", "z_std"} <= set(dict(model.named_buffers()))
    assert not any(name.startswith("z_") for name, _ in model.named_parameters())
    z = roma.rotmat_to_rotvec(model.z_rot.double().transpose(-1, -2) @ poses) / model.z_std.double()
    assert z.mean(0).abs().max() < 1e-4 and (z.std(0) - 1).abs().max() < 1e-4
    save_model(model, tmp_path)
    state = torch.load(tmp_path / "weights.pt", weights_only=True)
    assert torch.equal(state["z_rot"], model.z_rot) and torch.equal(state["z_std"], model.z_std)
    loaded = load_model(tmp_path)
    x, t = poses19(3), torch.tensor([0.01, 0.3, 0.9])
    with torch.no_grad():
        assert torch.equal(model.eval().score(x, t), loaded.score(x, t))


def test_the_timeres_net_takes_the_z_input():
    model, _ = z_model(net="timeres", net_width=64, net_blocks=2, net_embed=32, net_zero_init=False)
    x, t = poses19(3), torch.tensor([0.003, 0.1, 0.9])
    assert model.eval().score(x, t).shape == (3, J19, 3)
    assert torch.isfinite(model.loss(poses19(8)))


def test_the_z_input_keeps_the_divergence_exact():
    model, _ = z_model(hidden=[32])
    model = model.double()
    x, t = poses19(3, dtype=torch.float64), torch.rand(3, dtype=torch.float64)
    with torch.no_grad():
        _, divergence = model.velocity_and_divergence(x, t)
        numeric = torch.zeros(3, dtype=torch.float64)
        for joint in range(J19):
            for i in range(3):
                e = torch.zeros(3, J19, 3, dtype=torch.float64)
                e[:, joint, i] = 1e-5
                numeric += (model.velocity(exp_frame(x, e), t) - model.velocity(exp_frame(x, -e), t))[:, joint, i] / 2e-5
    assert torch.allclose(divergence, numeric, atol=1e-6)


def test_the_z_input_needs_its_statistics_and_only_it_takes_them():
    with pytest.raises(ValueError, match="z_mean"):
        build_model("rsgm", num_joints=J19, inputs="local+z")
    with pytest.raises(ValueError, match="z_mean"):
        build_model("rsgm", num_joints=J19, z_mean=[0.0] * 57, z_std=[1.0] * 57)
    with pytest.raises(ValueError, match="positive"):
        build_model("rsgm", num_joints=J19, inputs="local+z", z_mean=[0.0] * 57, z_std=[0.0] * 57)


def test_the_plain_log_is_roma_s_away_from_pi():
    from hp.models.rsgm import log_rotations
    r = roma.random_rotmat(10_000, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    v = roma.rotmat_to_rotvec(r)
    far = v.norm(dim=-1) < math.pi - 1e-3
    assert torch.allclose(log_rotations(r)[far], v[far], atol=1e-9)
    assert torch.equal(log_rotations(torch.eye(3)), torch.zeros(3))
