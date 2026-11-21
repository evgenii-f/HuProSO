"""Golden values of DPoser on fixed inputs: pin the published model against later changes of hp/models/dposer.py.

PLAIN was computed with the code of commit 7847844, before any change of DPoser for this line of work (noise_scale,
the ELBO); SCALED with noise_scale = D at commit 4e0e612 (after the reverse test, before the ELBO). A small model
(seed 0, 2 joints, hidden 64), in float64, on fixed rotations and vectors: the loss in eval mode, the score and the probability
flow velocity, the likelihood with 5 ODE steps, and 4 samples (seed 1) in the normalized space. If one of these tests
fails, the numerics of DPoser as published (or of its noise_scale) changed.

In float64 everything is pinned to rel 1e-8. In float32 (the dtype of training) the loss, the score and the velocity
are pinned to rel 1e-5, the samples and the likelihood only to 2%: the sampler and the ODE of the untrained network
(its output divided by sigma down to 0.01) amplify rounding, and the BLAS of macOS and of Linux gave samples 0.5% and
log p 0.3% apart.
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
PLAIN = {'log_prob': [-4254.03563339, -1502.91660097, -2468.20127197],
 'loss': 107.506153245,
 'sample': [0.101783878645, -0.0655700113742, -0.0345444403345, -1.9292267737, -2.06895561793, -0.809040168783,
            -0.968238671193, 0.0832919824543, -0.880028641912, 0.0205262156335, -0.0307084801445, 0.0292943328194,
            0.123102247136, -0.117380029216, 0.368874249473, -0.461881632174, -1.19677449429, -0.236653276824,
            0.830161001524, -0.867562622323, 1.54298163907, -0.0193732658248, -0.122230695817, -0.0323171437412],
 'score': [0.0341808165398, 0.287821162438, -0.486061841966, -0.563440382222, 0.0410276107813, 0.151431979975,
           0.924222347048, -0.271802614577, -0.270113563681, 0.718253824682, -0.0668007517524, -0.0500744783718,
           -4.93467004783, 11.9364258165, -8.71450715387, 32.0792939852, 3.33319033049, -12.471786967],
 'velocity': [0.169345619167, 0.381244316211, -0.298775817327, -0.841528811495, -0.479514289443, 2.02349470843,
              3.72883578134, 8.47904536407, 0.667601195321, -4.8470197782, -3.98188661188, 0.824232807306,
              63.3562537553, -121.713074286, 68.7184978776, -296.885537245, -12.6484833184, 114.748858885]}
SCALED = {'log_prob': [-1952.24148632, -4963.48191078, 1359.8490569],
 'loss': 158.291667061,
 'sample': [-0.0218016206214, 0.0254808362636, 0.0256584826104, 0.0253770170202, -0.0785820168999, 0.504093657979,
            -0.69758663963, -1.28905919879, -0.364043807968, -0.369768500679, 0.393797210286, 1.87587033952,
            0.514479993244, -2.11945625696, -1.11243777067, 1.87995268465, -0.059930840356, 0.473377203612,
            1.33298204951, -1.88535861, 1.00819216367, 1.75682902077, -0.0422389519728, 0.122613116852],
 'score': [0.0683616330796, 0.143910581219, -0.486061841966, -0.187813460741, 0.164110443125, 0.100954653317,
           1.8484446941, -0.135901307288, -0.270113563681, 0.239417941561, -0.26720300701, -0.0333829855812,
           -9.86934009566, 5.96821290827, -8.71450715387, 10.6930979951, 13.3327613219, -8.31452464468],
 'velocity': [0.169888325043, 0.372104555852, -0.298775817327, -0.805744715381, -0.478537165678, 2.02109034734,
              5.5199381817, 9.53252847221, 0.667601195321, -10.4147975043, -4.17607200546, 0.921274951651,
              41.1379037191, -229.200579794, 68.7184978776, -874.633573705, 9.86304899754, 170.903075018]}
PLAIN_32 = {'log_prob': [-907.0454, -2097.15, -1442.37],
 'loss': 24.65368,
 'sample': [0.2907666, 0.1099813, -0.1431534, -0.3658156, 0.8354837, -0.9886936, -0.8619279, 0.1154273, -0.0132136,
            0.5675594, 0.5467601, -0.175747, -0.5728869, -0.1185595, -0.189687, 0.2899792, 1.798711, -0.8095173,
            -1.702313, 2.220619, -1.049057, 1.460709, 0.6488227, 1.270588],
 'score': [0.4210394, -0.2716099, 0.06494649, -0.01259579, -0.5606025, -0.1583243, -0.2977988, -0.1394908,
           -0.09708638, -0.1198827, -0.5244325, -0.3454558, 14.35804, -14.55598, -5.818551, 2.40213, -24.40115,
           11.76362],
 'velocity': [0.8218937, 0.4193769, -0.3850619, -0.1085972, -0.4561647, -0.1287836, 3.825776, -2.712798, -1.393442,
              -4.255284, 11.08225, 7.168182, -129.8448, 136.6394, 59.58266, -20.44591, 226.9223, -108.9547]}
SCALED_32 = {'log_prob': [-1748.188, -1365.335, -1061.802],
 'loss': 9.571525,
 'sample': [0.3749664, 2.107066, -1.239609, -2.453946, 0.4763557, -1.832569, 0.1492939, -1.004175, -0.3138908,
            -0.2527037, -0.1026404, 0.5282674, 0.2536652, 1.288298, -0.08036522, -0.1732519, 0.1207855, -2.11207,
            1.00504, -2.019333, 2.01567, -0.1496432, -0.03866439, -0.1962484],
 'score': [0.8420787, -0.1358049, 0.06494649, -0.004198595, -2.24241, -0.1055495, -0.5955975, -0.06974542,
           -0.09708638, -0.03996091, -2.09773, -0.2303039, 28.71607, -7.277991, -5.818551, 0.8007098, -97.6046,
           7.842414],
 'velocity': [0.8285787, 0.4280019, -0.3850619, -0.1077973, -0.4695161, -0.1262698, 3.248655, -2.172144, -1.393442,
              -3.325974, 9.557762, 7.837661, -65.19776, 267.716, 59.58266, -63.70826, 62.12303, -161.9203]}


def outputs(dtype=torch.float64, **kwargs):
    torch.manual_seed(0)
    m = build_model("dposer", num_joints=2, hidden=64, embed=16, blocks=1, sample_steps=5, **kwargs).to(dtype).eval()
    pose = fixed_rotations(batch=16, joints=2)
    x = torch.randn(3, 6, generator=torch.Generator().manual_seed(1), dtype=dtype)
    t = torch.tensor([0.05, 0.4, 0.9], dtype=dtype)
    with torch.no_grad():
        out = {"loss": m.loss(pose).item(), "score": m.score(x, t).flatten().tolist(),
               "velocity": m.ode_velocity(x, t).flatten().tolist(),
               "log_prob": m.log_prob_normalized(x, steps=5).tolist()}
        torch.manual_seed(1)
        out["sample"] = m.normalize(m.sample(4)).flatten().tolist()
    return m, out


LONG_CHAINS = ("sample", "log_prob")


@pytest.mark.parametrize("dtype, kwargs, golden", [
    (torch.float64, {}, PLAIN), (torch.float64, {"noise_scale": D}, SCALED),
    (torch.float32, {}, PLAIN_32), (torch.float32, {"noise_scale": D}, SCALED_32),
], ids=["published-f64", "noise_scale-f64", "published-f32", "noise_scale-f32"])
def test_dposer_reproduces_its_golden_values(dtype, kwargs, golden):
    m, out = outputs(dtype, **kwargs)
    assert sorted(m.state_dict()) == STATE_KEYS  # checkpoints of the published model load unchanged
    for name, value in golden.items():
        if dtype == torch.float64:
            tolerance = {"rel": 1e-8, "abs": 1e-10}
        else:
            tolerance = {"rel": 2e-2, "abs": 1e-3} if name in LONG_CHAINS else {"rel": 1e-5, "abs": 1e-6}
        assert out[name] == pytest.approx(value, **tolerance), name
