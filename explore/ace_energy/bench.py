"""The speed of an ACE energy's score, and a check that a faster code gives the same numbers. From the root:

    python explore/ace_energy/bench.py --save-reference      # once, with the code before an optimisation
    python explore/ace_energy/bench.py                       # time, and compare with the reference
    python explore/ace_energy/bench.py --device cuda:0

Models: the linear energy P+L (energy_b_rich_products_jd_r0.001.pt: products and the joint density) and the MLP
embedding K = 8 on it (energy_mlp_pl_k8.pt), as inspect_energy.load_energy loads them. 500 validation poses, noised.
Cases, in ms per call (median of --repeats after one warm-up):

- score, one t: scaled_score of a batch at one noise level (the samplers, inpainting and the likelihood ODE);
- score, many t: at noise levels drawn as in training (one per pose);
- DPS: the score with its graph and the gradient of a fixed linear function of it with respect to the poses, as the
  guidance of hp.models.inpainting does;
- train step: the MLP's loss and its gradient with respect to all parameters, as embedding.py (create_graph).

The check: the scores, the DPS gradients and the training gradients against those saved by --save-reference
(~/data/experiments/ace/bench_reference.pt), as the largest error relative to the largest entry. A faster code is
accepted when every error is below 1e-4 (float32).
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit import ROTATIONS, SPLIT, draw  # noqa: E402
from hp.evaluation.data import load_reference  # noqa: E402
from hp.models.rsgm import exp_frame  # noqa: E402
from inspect_energy import FOLDER, load_energy  # noqa: E402

MODELS = {"linear P+L": "energy_b_rich_products_jd_r0.001.pt", "MLP K8": "energy_mlp_pl_k8.pt"}
REFERENCE_FILE = FOLDER / "bench_reference.pt"


def poses(ref, n: int):
    """(x one t, t one, x many t, t many): n validation poses noised at s = 0.05, and at times drawn as in training."""
    valid = load_reference("valid", ROTATIONS, SPLIT)
    x0 = valid[torch.randperm(len(valid), generator=torch.Generator().manual_seed(11))[:n]]
    g = torch.Generator().manual_seed(12)
    t1 = torch.full((n,), float(ref.t_of_s(torch.tensor(0.05, dtype=torch.float64))), dtype=torch.float64)
    x1 = draw(ref, x0, t1, g)[0]
    tm = ref.loss_t_min + (1 - ref.loss_t_min) * torch.rand(n, dtype=torch.float64, generator=g)
    xm = draw(ref, x0, tm, g)[0]
    return x1.float(), t1.float(), xm.float(), tm.float()


def timed(f, repeats: int, device: torch.device):
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    out = f()
    sync()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        f()
        sync()
        times.append(1000 * (time.perf_counter() - t0))
    return statistics.median(times), out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--poses", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--save-reference", action="store_true")
    parser.add_argument("--label", default="", help="a name for this code version in the table")
    parser.add_argument("--compile", action="store_true", help="Embedded.compile(): the scores without a graph")
    parser.add_argument("--no-quadratic", action="store_true",
                        help="the products as columns of the features, not as a quadratic form (the code before it)")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    _, ref, _, _ = load_energy(FOLDER / MODELS["linear P+L"])
    x1, t1, xm, tm = (v.to(dev) for v in poses(ref, args.poses))
    w = torch.randn(x1.shape[0], 19, 3, generator=torch.Generator().manual_seed(13)).to(dev)
    out, rows = {}, []
    for name, f in MODELS.items():
        energy = load_energy(FOLDER / f, device=args.device)[0]

        def dps():
            delta = torch.zeros(x1.shape[0], 19, 3, device=dev, requires_grad=True)
            s = energy.scaled_score(exp_frame(x1, delta), t1, create_graph=True)
            return torch.autograd.grad((s * w).sum(), delta)[0]

        cases = {"score, one t": lambda: energy.scaled_score(x1, t1).detach(),
                 "score, many t": lambda: energy.scaled_score(xm, tm).detach(),
                 "DPS": dps}
        if args.no_quadratic:
            energy.quadratic = False
        if args.compile:
            energy.compile()
        if name.startswith("MLP"):
            params = [p for p in energy.parameters()]
            target = torch.randn(xm.shape[0], 19, 3, generator=torch.Generator().manual_seed(14)).to(dev)

            def train():  # the parameters need gradients only here (as in training; inference has none)
                for p in params:
                    p.requires_grad_(True)
                loss = ((energy.scaled_score(xm, tm, create_graph=True) - target) ** 2).sum((1, 2)).mean()
                grads = torch.autograd.grad(loss, params, allow_unused=True)  # the last bias does not reach the score
                for p in params:
                    p.requires_grad_(False)
                return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p, g in zip(params, grads)])

            cases["train step"] = train
        for case, f_case in cases.items():
            ms, value = timed(f_case, args.repeats, dev)
            out[name, case] = value.detach().float().cpu()
            rows.append((name, case, ms))
    if args.save_reference:
        torch.save({f"{n}|{c}": v for (n, c), v in out.items()}, REFERENCE_FILE)
        print("saved", REFERENCE_FILE)
        saved = None
    else:
        saved = torch.load(REFERENCE_FILE, weights_only=True)
    print(f"{args.label or 'this code'}: {args.poses} poses, {args.device}, threads {torch.get_num_threads()}")
    print("| model | case | ms | error to the reference |\n|---|---|---|---|")
    worst = 0.0
    for name, case, ms in rows:
        err = ""
        if saved is not None:
            r = saved[f"{name}|{case}"]
            e = float((out[name, case] - r).abs().max() / r.abs().max())
            worst = max(worst, e)
            err = f"{e:.1e}"
        print(f"| {name} | {case} | {ms:.1f} | {err} |")
    if saved is not None:
        print("same numbers (all errors < 1e-4):", worst < 1e-4)


if __name__ == "__main__":
    main()
