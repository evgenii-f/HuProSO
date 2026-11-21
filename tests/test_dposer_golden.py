"""Golden values of DPoser on fixed inputs: pin the published model against later changes of hp/models/dposer.py.

PLAIN was computed with the code of commit 7847844, before any change of DPoser for this line of work (noise_scale,
the ELBO); SCALED with noise_scale = D at commit 4e0e612 (after the reverse test, before the ELBO). A small model
(seed 0, 2 joints, hidden 64) on fixed rotations and vectors: the loss in eval mode, the score and the probability
flow velocity, the likelihood with 5 ODE steps, and 4 samples (seed 1) in the normalized space. If one of these tests
fails, the numerics of DPoser as published (or of its noise_scale) changed.
"""
import pytest
import torch
from conftest import fixed_rotations

from hp.models import build_model

D = [0.5, 2.0, 1.0, 3.0, 0.25, 1.5]
STATE_KEYS = ['data_mean', 'data_std', 'net.b1_dense1.bias', 'net.b1_dense1.weight', 'net.b1_dense1_t.bias',
 'net.b1_dense1_t.weight', 'net.b1_dense2.bias', 'net.b1_dense2.weight', 'net.b1_dense2_t.bias',
 'net.b1_dense2_t.weight', 'net.b1_gnorm1.bias', 'net.b1_gnorm1.weight', 'net.b1_gnorm2.bias', 'net.b1_gnorm2.weight',
 'net.post_dense.bias', 'net.post_dense.weight', 'net.pre_dense.bias', 'net.pre_dense.weight', 'net.pre_dense_t.bias',
 'net.pre_dense_t.weight', 'net.pre_gnorm.bias', 'net.pre_gnorm.weight', 'net.shared_time_embed.0.bias',
 'net.shared_time_embed.0.weight', 'sigmas']
PLAIN = {'log_prob': [-907.0454, -2097.15, -1442.37],
 'loss': 24.65368,
 'sample': [0.2907666, 0.1099813, -0.1431534, -0.3658156, 0.8354837, -0.9886936, -0.8619279, 0.1154273, -0.0132136,
            0.5675594, 0.5467601, -0.175747, -0.5728869, -0.1185595, -0.189687, 0.2899792, 1.798711, -0.8095173,
            -1.702313, 2.220619, -1.049057, 1.460709, 0.6488227, 1.270588],
 'score': [0.4210394, -0.2716099, 0.06494649, -0.01259579, -0.5606025, -0.1583243, -0.2977988, -0.1394908,
           -0.09708638, -0.1198827, -0.5244325, -0.3454558, 14.35804, -14.55598, -5.818551, 2.40213, -24.40115,
           11.76362],
 'velocity': [0.8218937, 0.4193769, -0.3850619, -0.1085972, -0.4561647, -0.1287836, 3.825776, -2.712798, -1.393442,
              -4.255284, 11.08225, 7.168182, -129.8448, 136.6394, 59.58266, -20.44591, 226.9223, -108.9547]}
SCALED = {'log_prob': [-1748.188, -1365.335, -1061.802],
 'loss': 9.571525,
 'sample': [0.3749664, 2.107066, -1.239609, -2.453946, 0.4763557, -1.832569, 0.1492939, -1.004175, -0.3138908,
            -0.2527037, -0.1026404, 0.5282674, 0.2536652, 1.288298, -0.08036522, -0.1732519, 0.1207855, -2.11207,
            1.00504, -2.019333, 2.01567, -0.1496432, -0.03866439, -0.1962484],
 'score': [0.8420787, -0.1358049, 0.06494649, -0.004198595, -2.24241, -0.1055495, -0.5955975, -0.06974542,
           -0.09708638, -0.03996091, -2.09773, -0.2303039, 28.71607, -7.277991, -5.818551, 0.8007098, -97.6046,
           7.842414],
 'velocity': [0.8285787, 0.4280019, -0.3850619, -0.1077973, -0.4695161, -0.1262698, 3.248655, -2.172144, -1.393442,
              -3.325974, 9.557762, 7.837661, -65.19776, 267.716, 59.58266, -63.70826, 62.12303, -161.9203]}


def outputs(**kwargs):
    torch.manual_seed(0)
    m = build_model("dposer", num_joints=2, hidden=64, embed=16, blocks=1, sample_steps=5, **kwargs).eval()
    pose = fixed_rotations(batch=16, joints=2)
    x = torch.randn(3, 6, generator=torch.Generator().manual_seed(1))
    t = torch.tensor([0.05, 0.4, 0.9])
    with torch.no_grad():
        out = {"loss": m.loss(pose).item(), "score": m.score(x, t).flatten().tolist(),
               "velocity": m.ode_velocity(x, t).flatten().tolist(),
               "log_prob": m.log_prob_normalized(x, steps=5).tolist()}
        torch.manual_seed(1)
        out["sample"] = m.normalize(m.sample(4)).flatten().tolist()
    return m, out


@pytest.mark.parametrize("kwargs, golden", [({}, PLAIN), ({"noise_scale": D}, SCALED)],
                         ids=["published", "noise_scale"])
def test_dposer_reproduces_its_golden_values(kwargs, golden):
    m, out = outputs(**kwargs)
    assert sorted(m.state_dict()) == STATE_KEYS  # checkpoints of the published model load unchanged
    for name, value in golden.items():
        assert out[name] == pytest.approx(value, rel=1e-5, abs=1e-6), name
