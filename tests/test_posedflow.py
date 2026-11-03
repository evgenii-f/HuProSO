"""PoseD-Flow trained here (hp/models/posedflow.py, PoseDFlow): their loss, sampler and likelihood, and the keypoint
condition (hp/models/conditioning.py, shared with RSGM)."""
import pytest
import roma
import torch

from hp.data.amass_cfg import AmassJoints4Learning
from hp.data.conditions import keypoints
from hp.models import build_model
from hp.models.io import load_model, save_model
from hp.models.posedflow import PoseDFlow, wrapped_normal_log_density
from hp.models.rsgm import exp_frame

J19 = len(AmassJoints4Learning)


def poses(n, joints=J19, seed=0, dtype=torch.float32):
    return roma.random_rotmat(n * joints, generator=torch.Generator().manual_seed(seed), dtype=dtype).reshape(n, joints, 3, 3)


def zero_field(model):
    """The model with a field that is 0 everywhere (the last layer of the network set to 0)."""
    last = model.net.layers[-1].module
    with torch.no_grad():
        for p in (last._layer.weight, last._layer.bias, last._hyper_bias.weight):
            p.zero_()
    return model


def test_the_target_of_the_loss_is_the_velocity_of_the_geodesic():
    x0, x1 = poses(8, 3, seed=1, dtype=torch.float64), poses(8, 3, seed=2, dtype=torch.float64)
    t, h = torch.rand(8, dtype=torch.float64), 1e-6
    xt, ut = PoseDFlow.geodesic(x0, x1, t)
    numeric = (PoseDFlow.geodesic(x0, x1, t + h)[0] - PoseDFlow.geodesic(x0, x1, t - h)[0]) / (2 * h)
    assert torch.allclose(ut, numeric, atol=1e-6)
    ends = PoseDFlow.geodesic(x0, x1, torch.ones(8, dtype=torch.float64))[0]
    assert torch.allclose(ends, x1, atol=1e-9)


def test_with_a_zero_field_the_loss_is_the_mean_squared_velocity_and_log_prob_the_noise_density():
    model = zero_field(build_model("posedflow", num_joints=3, d_model=16, num_layers=3, likelihood_steps=4).double())
    pose = poses(5, 3, dtype=torch.float64)
    assert torch.allclose(model.log_prob(pose), wrapped_normal_log_density(pose).sum(-1))
    # u_t = x_t hat(l): its squared Frobenius norm is 2 |l|^2, with l the rotation vector of x0^T x1
    generator = torch.Generator().manual_seed(0)
    torch.rand(5, dtype=torch.float64, generator=generator)
    x0 = roma.rotvec_to_rotmat(torch.randn(5, 3, 3, dtype=torch.float64, generator=generator))
    log = roma.rotmat_to_rotvec(x0.transpose(-1, -2) @ pose)
    expected = (2 * (log ** 2).sum(dim=(1, 2))).mean() / 27
    assert torch.allclose(model.eval().loss(pose), expected)


def test_the_divergence_is_the_one_of_finite_differences():
    torch.manual_seed(0)
    model = build_model("posedflow", num_joints=J19, d_model=32, num_layers=3, condition="3D").double()
    x, t = poses(3, dtype=torch.float64), torch.rand(3, dtype=torch.float64)
    f = model.cond_features(keypoints(poses(3, seed=2, dtype=torch.float64)).reshape(3, -1), 3)
    with torch.no_grad():
        _, divergence = model.velocity_and_divergence(x, t, f)
        numeric = torch.zeros(3, dtype=torch.float64)
        for joint in range(J19):
            for i in range(3):
                e = torch.zeros(3, J19, 3, dtype=torch.float64)
                e[:, joint, i] = 1e-5
                numeric += (model.field_frame(exp_frame(x, e), t, f) - model.field_frame(exp_frame(x, -e), t, f))[:, joint, i] / 2e-5
    assert torch.allclose(divergence, numeric, atol=1e-6)


def test_the_condition_sets_cond_dim_and_bad_settings_are_refused():
    assert build_model("posedflow", num_joints=J19, d_model=8, condition="3D").cond_dim == 66
    assert not any(k.startswith("cond_net") for k in build_model("posedflow", num_joints=2, d_model=8).state_dict())
    with pytest.raises(ValueError):
        build_model("posedflow", num_joints=J19, d_model=8, cond_dim=66, cond_mask="random")
    with pytest.raises(ValueError):
        build_model("posedflow", num_joints=J19, d_model=8, method="rk4")


def test_without_a_condition_the_loss_takes_the_keypoints_of_the_poses():
    torch.manual_seed(0)
    model = build_model("posedflow", num_joints=J19, d_model=16, num_layers=3, condition="3D").eval()
    pose = poses(32)
    assert model.loss(pose).item() == model.loss(pose, keypoints(pose).reshape(32, -1)).item()
    assert model.loss(pose).item() != model.loss(pose, keypoints(poses(32, seed=1)).reshape(32, -1)).item()


def test_sample_and_log_prob_need_the_condition_and_depend_on_it():
    torch.manual_seed(0)
    model = build_model("posedflow", num_joints=J19, d_model=16, num_layers=3, condition="3D", cond_mask="random",
                        sample_steps=5, likelihood_steps=3).eval()
    pose = poses(3)
    c, other = keypoints(pose).reshape(3, -1), keypoints(poses(3, seed=1)).reshape(3, -1)
    hidden = c.clone().reshape(3, 22, 3)
    hidden[:, 7:12] = float("nan")
    assert model.sample(3, hidden.reshape(3, -1)).shape == (3, J19, 3, 3)
    with pytest.raises(ValueError):
        model.sample(3)
    lp = model.log_prob(pose, c)
    assert lp.shape == (3,) and torch.isfinite(lp).all() and not torch.allclose(lp, model.log_prob(pose, other))


def test_a_conditional_model_is_saved_and_loaded(tmp_path):
    torch.manual_seed(0)
    model = build_model("posedflow", num_joints=J19, d_model=16, num_layers=3, condition="3D", cond_mask="random").eval()
    save_model(model, tmp_path)
    loaded = load_model(tmp_path)
    assert type(loaded) is PoseDFlow and loaded.condition == "3D" and loaded.cond_mask == "random"
    pose = poses(4)
    assert torch.equal(model.loss(pose), loaded.loss(pose))
