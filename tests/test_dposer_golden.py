"""Golden values of DPoser on fixed inputs: pin the published model against later changes of hp/models/dposer.py.

The values of the published model were computed with the code of commit 7847844, before any change of DPoser for this
line of work (noise_scale, the ELBO); those with noise_scale = D with commit 4e0e612 (after the reverse test, before the
ELBO). A small model (seed 0, 2 joints, hidden 64) on fixed rotations and vectors: the loss in eval mode, the score and
the probability flow velocity, the likelihood with 5 ODE steps, and 4 samples (seed 1) in the normalized space. If one
of these tests fails, the numerics of DPoser as published (or of its noise_scale) changed.

The last layer of the network is scaled by 1e-3, so that the problem is well conditioned: the untrained network, its
output divided by sigma down to 0.01, makes the sampler and the ODE chaotic (a relative change of the weights of 1e-9
moved the likelihood and the samples by 1e-5 to 1e-3), and the golden values differed between macOS and Linux even in
float64. Scaled, no output amplifies a change of the weights more than about 400 times. Float64 is pinned to rel 1e-7,
float32 (the dtype of training) to rel 1e-3.
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
PLAIN_64 = {'log_prob': [-35.6393589306, -37.5713184053, -35.9899609671],
 'loss': 1.04455084663,
 'sample': [-1.25256908364, -1.94382749106, -1.48684644174, -1.18497705309, -2.04359855628, -0.763234835744,
            0.448353293682, 0.206230531484, 0.13610973582, 0.0219095809597, -0.0653498031326, 2.28038609039,
            0.369912644415, -0.14977630253, -0.299841762774, 0.394732339488, 1.43164841715, 0.234719877692,
            0.0106306904442, 0.104387704842, -0.0445873722823, 1.96678083406, 0.471963817265, 0.630823517816],
 'score': [3.41808076263e-05, 0.000287821154299, -0.000486061859316, -0.000563440401099, 4.10276092484e-05,
           0.000151432001893, 0.000924222389992, -0.000271802621049, -0.000270113570548, 0.000718253852912,
           -6.68007634301e-05, -5.00744728307e-05, -0.00493467053911, 0.0119364267483, -0.0087145068684,
           0.0320792950831, 0.00333318921874, -0.0124717871391],
 'velocity': [0.170429945508, 0.39037493681, -0.314195275121, -0.859402967504, -0.478212760588, 2.02829862189,
              7.3074583771, 7.42661573905, -0.378288363094, -2.06591480414, -4.24054155608, 0.630342602885,
              18.9639903874, -14.3330562918, -9.67715836648, -8.30039304315, 17.3368777364, 2.55273505297]}
SCALED_64 = {'log_prob': [-35.5698200827, -37.8792960221, -40.4051288184],
 'loss': 1.04564419089,
 'sample': [0.377527618229, 2.26419213624, 0.93580916953, -1.4936495771, -0.23512423726, -0.436837420967,
            -0.65142284413, -1.06426256086, -0.471314141336, -0.106952943215, -0.0187717197811, 1.69257012013,
            0.0267056316947, -0.046798984299, -0.0407125418154, 1.0648126109, 0.419379931574, 0.338036178242,
            -0.0641463365616, -2.46408916528, 0.493020778299, -2.35589493427, -0.0518622295594, -0.382945519513],
 'score': [6.83616152525e-05, 0.00014391057715, -0.000486061859316, -0.000187813467033, 0.000164110436994,
           0.000100954667929, 0.00184844477998, -0.000135901310524, -0.000270113570548, 0.000239417950971,
           -0.000267203053721, -3.33829818871e-05, -0.00986934107821, 0.00596821337414, -0.0087145068684,
           0.010693098361, 0.013332756875, -0.00831452475938],
 'velocity': [0.170430488213, 0.39036579705, -0.314195275121, -0.859367183406, -0.478211783464, 2.02829621753,
              7.30924947958, 7.42766922219, -0.378288363094, -2.07148258209, -4.24073574151, 0.630439645018,
              18.9417720352, -14.4405438057, -9.67715836648, -8.87814109939, 17.3593892612, 2.60888926988]}
PLAIN_32 = {'log_prob': [-35.639774, -37.571384, -35.98999],
 'loss': 1.1339107,
 'sample': [-0.57711285, -0.23008671, 0.12314385, 0.32810917, 0.058697987, 0.31973514, -0.80900091, -2.3594375,
            1.3025912, -0.22945915, -0.64637721, 0.3523027, -0.0074740788, 0.028472871, -0.022015922, 0.10532122,
            1.0478483, -0.25678074, -1.3476728, 2.3223617, -1.2180821, 0.1878856, -0.5453468, 1.4694476],
 'score': [3.4180801e-05, 0.00028782152, -0.00048606229, -0.00056344055, 4.1028037e-05, 0.00015143164, 0.00092422235,
           -0.00027180265, -0.00027011349, 0.00071825396, -6.6800792e-05, -5.0074639e-05, -0.0049346876, 0.011936419,
           -0.0087145064, 0.032079298, 0.0033331877, -0.012471791],
 'velocity': [0.17042994, 0.39037493, -0.31419528, -0.85940301, -0.47821277, 2.0282986, 7.3074594, 7.4266157,
              -0.37828839, -2.0659149, -4.2405419, 0.6303426, 18.963991, -14.333056, -9.6771593, -8.3003931,
              17.336878, 2.5527351]}
SCALED_32 = {'log_prob': [-35.567627, -37.879307, -40.405163],
 'loss': 1.1320342,
 'sample': [1.0515575, 1.6116492, -0.42913628, -1.6308891, -0.017997632, -0.80765927, -0.29466501, -2.8353641,
            0.625377, 1.7267045, 0.3654764, -1.1006224, 0.1341106, -2.1240594, 0.82198614, -1.5864311, -1.4066302,
            2.2916243, 0.37286624, -2.4847121, 0.69602674, -0.20884402, 0.046843827, -0.72138995],
 'score': [6.8361602e-05, 0.00014391076, -0.00048606229, -0.00018781351, 0.00016411215, 0.00010095443, 0.0018484447,
           -0.00013590133, -0.00027011349, 0.00023941799, -0.00026720317, -3.3383094e-05, -0.0098693753, 0.0059682094,
           -0.0087145064, 0.010693099, 0.013332751, -0.0083145276],
 'velocity': [0.1704305, 0.39036581, -0.31419528, -0.85936719, -0.47821179, 2.0282962, 7.3092504, 7.4276695,
              -0.37828839, -2.0714827, -4.240736, 0.63043964, 18.941772, -14.440544, -9.6771593, -8.8781414, 17.35939,
              2.6088893]}
TOLERANCE = {torch.float64: {"rel": 1e-7, "abs": 1e-10}, torch.float32: {"rel": 1e-3, "abs": 1e-6}}


def outputs(dtype=torch.float64, **kwargs):
    torch.manual_seed(0)
    m = build_model("dposer", num_joints=2, hidden=64, embed=16, blocks=1, sample_steps=5, **kwargs).eval()
    with torch.no_grad():
        m.net.post_dense.weight.mul_(1e-3)
        m.net.post_dense.bias.mul_(1e-3)
    m = m.to(dtype)
    pose = fixed_rotations(batch=16, joints=2)
    x = torch.randn(3, 6, generator=torch.Generator().manual_seed(1), dtype=torch.float64).to(dtype)
    t = torch.tensor([0.05, 0.4, 0.9], dtype=torch.float64).to(dtype)
    with torch.no_grad():
        out = {"loss": m.loss(pose).item(), "score": m.score(x, t).flatten().tolist(),
               "velocity": m.ode_velocity(x, t).flatten().tolist(),
               "log_prob": m.log_prob_normalized(x, steps=5).tolist()}
        torch.manual_seed(1)
        out["sample"] = m.normalize(m.sample(4)).double().flatten().tolist()
    return m, out


@pytest.mark.parametrize("dtype, kwargs, golden", [
    (torch.float64, {}, PLAIN_64), (torch.float64, {"noise_scale": D}, SCALED_64),
    (torch.float32, {}, PLAIN_32), (torch.float32, {"noise_scale": D}, SCALED_32),
], ids=["published-f64", "noise_scale-f64", "published-f32", "noise_scale-f32"])
def test_dposer_reproduces_its_golden_values(dtype, kwargs, golden):
    m, out = outputs(dtype, **kwargs)
    assert sorted(m.state_dict()) == STATE_KEYS  # checkpoints of the published model load unchanged
    for name, value in golden.items():
        assert out[name] == pytest.approx(value, **TOLERANCE[dtype]), name
