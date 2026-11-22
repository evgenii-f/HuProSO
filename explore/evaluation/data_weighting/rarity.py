"""Rarity of the training poses and the sampling weights of the data-weighting runs.

The hypothesis: rare poses are not absent from the training data but poorly represented, so the score is learned worst
where the data is sparse; sampling the sparse regions more often may improve the completion of hard poses (the
coverage check, section 17 of the pose completion report, found the hard test poses 1.7-2.3x farther from the data).

Rarity r_i of a c10 training pose: the distance to its k-th nearest neighbour (k = 5) among the c100 poses (every 10th
c10 pose, 186k), itself excluded, in the chordal metric over the 19 joints the model learns, as an RMS angle
(deg) per joint. Static stretches (many near-identical poses) get a small r, fast or unusual motion a large one.

Sampling weight: w_i = clip((r_i / median r)^beta, w_min, w_max), w_min = 0.1, w_max = 10 (the cap keeps a few broken
mocap poses from dominating). beta is set by the effective sample size ESS = (sum w)^2 / sum w^2 relative to N, before
any training: "soft" ESS 0.7 N, "strong" ESS 0.4 N. beta = 0 is the uniform sampling of the baseline (fact_RDW_s10).
The weights are in the order of the poses of PoseDFlow_train.pt of the c10 folder, the order of the training set.

    python explore/evaluation/data_weighting/rarity.py --device mps    # about 5 minutes on the Mac
    python explore/evaluation/data_weighting/rarity.py --reuse         # the weights again from rarity.pt
"""
import argparse
import json
import time
from pathlib import Path

import roma
import torch

from hp.data.amass_cfg import AmassJoints4Learning

C10 = Path("~/data/AMASS_rotations_posedflow_c10/PoseDFlow_train.pt").expanduser()
OUT = Path("~/data/experiments/data_weighting").expanduser()
JOINTS = [j.value for j in AmassJoints4Learning]  # the 19 joints among the 22 of the file
K = 5
REF_STRIDE = 10
W_MIN, W_MAX = 0.1, 10.0
ESS_TARGETS = {"soft": 0.7, "strong": 0.4}


def matrices(quat: torch.Tensor) -> torch.Tensor:
    """[N, 22, 4] quaternions [x, y, z, w] -> [N, 19 * 9] flattened matrices of the learned joints."""
    return roma.unitquat_to_rotmat(quat[:, JOINTS].double()).float().reshape(len(quat), -1)


def rarity(x: torch.Tensor, device: str, chunk: int = 4096) -> torch.Tensor:
    """r [N] (deg): RMS angle per joint to the K-th nearest c100 pose, the pose itself excluded."""
    ref_index = torch.arange(0, len(x), REF_STRIDE)
    ref = x[ref_index].to(device)
    j = len(JOINTS)
    out = torch.empty(len(x))
    t0 = time.time()
    for start in range(0, len(x), chunk):
        q = x[start:start + chunk].to(device)
        dots = q @ ref.T  # [chunk, R]; chordal distance^2 = sum_j (6 - 2 tr) = 6 j - 2 dot
        own = torch.arange(start, start + len(q), device=device)
        is_ref = own % REF_STRIDE == 0
        rows = torch.nonzero(is_ref).squeeze(1)
        dots[rows, own[rows] // REF_STRIDE] = -1e9  # not its own neighbour
        kth = dots.topk(K, dim=1).values[:, -1]
        chord2 = (6 * j - 2 * kth).clamp_min(0) / j  # per joint: 4 (1 - cos theta) ~ theta^2 at small angles
        theta = torch.acos((1 - chord2 / 4).clamp(-1, 1))  # the angle of a joint at that chordal distance
        out[start:start + len(q)] = theta.rad2deg().cpu()
        if start // chunk % 50 == 0:
            print(f"  {start + len(q):,} / {len(x):,}, {time.time() - t0:.0f} s", flush=True)
    return out


def weights(r: torch.Tensor, beta: float) -> torch.Tensor:
    return (r / r.median()).pow(beta).clamp(W_MIN, W_MAX)


def ess(w: torch.Tensor) -> float:
    w = w.double()
    return float(w.sum() ** 2 / (w ** 2).sum() / len(w))


def beta_for(r: torch.Tensor, target: float) -> float:
    """The beta at which ESS / N = target, by bisection on [0, 5]: ESS falls with beta there; at larger beta the
    clipped weights turn into two levels (w_min and w_max) and ESS rises again."""
    lo, hi = 0.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ess(weights(r, mid)) > target else (lo, mid)
    return (lo + hi) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse", action="store_true", help="the weights from the saved rarity.pt")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.reuse:
        r = torch.load(OUT / "rarity.pt", weights_only=True)
    else:
        x = matrices(torch.as_tensor(torch.load(C10, weights_only=True)))
        print(f"{len(x):,} c10 poses, {len(x[::REF_STRIDE]):,} reference poses, k = {K}", flush=True)
        r = rarity(x, args.device)
        torch.save(r, OUT / "rarity.pt")
    q = torch.quantile(r, torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99, 0.999]))
    info = {"k": K, "ref_stride": REF_STRIDE, "w_min": W_MIN, "w_max": W_MAX,
            "rarity_quantiles_deg": {p: round(float(v), 3) for p, v in zip(["1%", "10%", "50%", "90%", "99%", "99.9%"], q)}}
    for name, target in ESS_TARGETS.items():
        beta = beta_for(r, target)
        w = weights(r, beta)
        torch.save(w, OUT / f"weights_{name}.pt")
        top = w.topk(int(0.1 * len(w))).values.sum() / w.sum()
        info[name] = {"beta": round(beta, 4), "ess": round(ess(w), 4), "share_of_top10pct": round(float(top), 4),
                      "at_cap": round(float((w >= W_MAX).float().mean()), 5),
                      "at_floor": round(float((w <= W_MIN).float().mean()), 5)}
    (OUT / "weights.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
