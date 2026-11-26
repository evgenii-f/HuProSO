"""Do the noise levels of an energy agree with each other? The noised densities p_s of RSGM obey the heat equation of
IGSO(3) (its generator is Laplace / 2, hp.models.igso3), so the energy E_s = -log p_s obeys

    dE/d log s = s/2 (Laplace E - |grad E|^2) + s c'(s),

with c(s) a constant per level that a score does not fix. The residual R = dE/d log s - s/2 (Laplace E - |grad E|^2)
of an energy (inspect_energy.load_energy; E_s = E~_s / std(s)) is measured at points of several kinds; R minus its mean
on the points near the data ("near": training poses noised at s, where the energy was trained) estimates the violation.
Kinds: "near"; "far 0.2", "far 1": training poses noised at s' = 0.2 and 1 (rad^2), evaluated at the level s;
"hyp MLP", "hyp net": the MLP's and the network's completions of the legs on the 40 validation poses where the MLP was
worst (diag5/inpaint_mlp_pl_k8.pt, diag2/inpaint_network_rank.pt; a diagnostic only, never for training).

Derivatives by central differences in float64: grad and Laplace along x_j exp(h e_a) (the 57 tangent directions of
the metric of IGSO(3), where D^l has the eigenvalue -l(l+1)), d/d log s by +-delta. "--toy" first checks the method on
E_s = -sum_j log f(angle(x_j); s), the IGSO(3) density around the identity, whose residual is 0. On the CPU, from the
root of the repository:

    python explore/ace_energy/fp_residual.py --toy
    python explore/ace_energy/fp_residual.py --save ~/data/experiments/ace/fp_residual.pt
"""
import argparse
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit import ROTATIONS, SPLIT, draw  # noqa: E402
from hp.evaluation.completion import valid_poses  # noqa: E402
from hp.evaluation.data import load_reference  # noqa: E402
from hp.models import igso3  # noqa: E402
from hp.models.rsgm import exp_frame  # noqa: E402
from inspect_energy import load_energy  # noqa: E402

ACE = Path("~/data/experiments/ace").expanduser()
ENERGIES = {"P+L": "energy_b_rich_products_jd_r0.001.pt", "MLP": "energy_mlp_pl_k8.pt"}


def angle(x: torch.Tensor) -> torch.Tensor:
    tr = x.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.arccos(((tr - 1) / 2).clamp(-1 + 1e-15, 1 - 1e-15))


def derivatives(energy_of, x: torch.Tensor, s: float, h: float, delta: float, batch: int):
    """(E, grad [N, 57], Laplace E, dE/d log s) of poses x [N, 19, 3, 3] at the level s, by central differences."""
    n, j = len(x), x.shape[1]
    d = torch.zeros(2 * 3 * j + 1, j, 3, dtype=torch.float64)
    for k in range(3 * j):
        d[1 + 2 * k, k // 3, k % 3], d[2 + 2 * k, k // 3, k % 3] = h, -h
    xs = exp_frame(x[:, None].expand(n, len(d), j, 3, 3), d[None].expand(n, -1, -1, -1)).reshape(-1, j, 3, 3)
    e = torch.cat([energy_of(xs[i:i + batch], s) for i in range(0, len(xs), batch)]).reshape(n, len(d))
    e0, ep, em = e[:, 0], e[:, 1::2], e[:, 2::2]
    grad = (ep - em) / (2 * h)
    lap = ((ep + em - 2 * e0[:, None]) / h ** 2).sum(1)
    up = torch.cat([energy_of(x[i:i + batch], s * math.exp(delta)) for i in range(0, n, batch)])
    down = torch.cat([energy_of(x[i:i + batch], s * math.exp(-delta)) for i in range(0, n, batch)])
    return e0, grad, lap, (up - down) / (2 * delta)


def residual(energy_of, x, s, args):
    """R [N] and the terms dE/d log s, s/2 Laplace E, s/2 |grad E|^2 [N] (nats per unit of log s)."""
    _, grad, lap, dlog = derivatives(energy_of, x, s, args.h, args.delta, args.batch)
    a, b = s / 2 * lap, s / 2 * (grad ** 2).sum(1)
    return dlog - a + b, dlog, a, b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", type=float, nargs="+", default=[0.0005, 0.002, 0.005, 0.02, 0.05, 0.2, 1.0])
    parser.add_argument("--n", type=int, default=200, help="points per kind and level")
    parser.add_argument("--h", type=float, default=1e-3)
    parser.add_argument("--delta", type=float, default=0.02, help="step in log s")
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--energies", nargs="+", default=list(ENERGIES))
    parser.add_argument("--energy", nargs="+", default=None, metavar="LABEL=FILE",
                        help="other energies (files in ~/data/experiments/ace or paths), instead of the default two")
    parser.add_argument("--save", default=None)
    args = parser.parse_args()
    if args.energy:
        ENERGIES.clear()
        ENERGIES.update(dict(e.split("=", 1) for e in args.energy))
    g = torch.Generator().manual_seed(0)
    train = load_reference("train", ROTATIONS, SPLIT)
    pick = torch.randperm(len(train), generator=g)[:args.n]
    x0 = train[pick].double()

    # the hypotheses of the legs on the 40 poses where the MLP was worst, and the network's on the same poses
    legs = {r: torch.load(ACE / f, weights_only=True)["occlusions"]["legs"] for r, f in
            (("MLP", "diag5/inpaint_mlp_pl_k8.pt"), ("net", "diag2/inpaint_network_rank.pt"))}
    worst = legs["MLP"]["mpvpe"].double().mean(1).argsort(descending=True)[:40]
    hyps = {f"hyp {r}": o["poses"][worst].reshape(-1, 19, 3, 3).double()[:args.n] for r, o in legs.items()}
    assert valid_poses(n=1000)[:200].shape[0] == legs["MLP"]["mpvpe"].shape[0]

    if args.toy:
        def toy(x, s):
            return -igso3.log_density(angle(x), torch.tensor(s, dtype=torch.float64)).sum(-1)
        print("toy: E_s = -sum_j log f(angle(x_j); s); the residual should be 0 (c' = 0 for a normalized density)")
        for s in args.levels:
            x = exp_frame(torch.eye(3, dtype=torch.float64).expand(args.n, 19, 3, 3),
                          torch.randn(args.n, 19, 3, generator=g, dtype=torch.float64) * math.sqrt(s))
            r, dlog, a, b = residual(toy, x, s, args)
            print(f"  s {s:g}: R mean {float(r.mean()):+.4f}, rms {float(r.pow(2).mean().sqrt()):.4f}; "
                  f"terms |dE/dlog s| {float(dlog.abs().mean()):.2f}, s/2 Laplace {float(a.mean()):.2f}, "
                  f"s/2 |grad|^2 {float(b.mean()):.2f}", flush=True)
        return

    results = {}
    for ename in (list(ENERGIES) if args.energy else args.energies):
        energy, ref, bins, info = load_energy(str(ACE / ENERGIES[ename]), "cpu")
        energy = energy.double()

        @torch.no_grad()
        def energy_of(x, s):
            st = torch.tensor(s, dtype=torch.float64)
            t = ref.t_of_s(st).expand(len(x))
            return energy(x, t) / torch.sqrt(-torch.expm1(-st))

        print(f"\n== {ename}: {info}", flush=True)
        print("| s | terms near: dE/dlog s, s/2 Laplace, s/2 grad^2 | c' s (near mean) | near rms | "
              "far 0.2 | far 1 | hyp MLP | hyp net |")
        print("|---|---|---|---|---|---|---|---|")
        for s in args.levels:
            t0 = time.perf_counter()
            t = ref.t_of_s(torch.tensor(s, dtype=torch.float64)).expand(args.n)
            points = {"near": draw(ref, x0, t, g)[0]}
            for far in (0.2, 1.0):
                points[f"far {far:g}"] = draw(ref, x0, ref.t_of_s(torch.tensor(far, dtype=torch.float64)).expand(
                    args.n), g)[0]
            points.update(hyps)
            out = {k: residual(energy_of, p.double(), s, args) for k, p in points.items()}
            c = out["near"][0].mean()
            results[(ename, s)] = {k: torch.stack(v) for k, v in out.items()}
            near = out["near"]
            cells = [f"{float(near[1].mean()):+.1f}, {float(near[2].mean()):+.1f}, {float(near[3].mean()):+.1f}",
                     f"{float(c):+.2f}", f"{float((near[0] - c).pow(2).mean().sqrt()):.2f}"]
            for k in ("far 0.2", "far 1", "hyp MLP", "hyp net"):
                r = out[k][0] - c
                cells.append(f"{float(r.mean()):+.1f} ({float(r.pow(2).mean().sqrt()):.1f})")
            print(f"| {s:g} | " + " | ".join(cells) + f" |  ({time.perf_counter() - t0:.0f} s)", flush=True)
    if args.save:
        torch.save(dict(levels=args.levels, n=args.n, results={f"{e}/{s:g}": v for (e, s), v in results.items()}),
                   Path(args.save).expanduser())
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
