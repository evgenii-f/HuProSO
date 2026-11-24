"""Is the ACE energy a -log p? Its differences against the log-likelihood of its own probability flow ODE, and against
the network's, on real noised poses (docs/experiments/2026-11-23_ace_descriptors.md). From the root of the repository:

    python explore/ace_energy/ode_check.py --fit ~/data/experiments/ace/energy_b_smooth_w123_p123.pt --poses 200

At every noise level s0 of --levels, validation poses x_0 are noised to x ~ p_{t0}(x | x_0) (the noising of the reference
run). Three log p_{t0}(x) are compared, up to a constant:

    energy     -E~_{t0}(x) / std(t0)                  the energy as it is (score matching leaves its constant open)
    ODE        the ODE of the energy's score from t0 to 1, with the exact divergence (normalized, uniform at t = 1)
    network    the same ODE with the reference network (RSGM.log_prob)

If the scores of all noise levels come from one diffusion, energy and ODE agree: correlation 1, slope 1. The bins of
log s are fitted independently, so nothing enforces it. Printed: Pearson and Spearman correlation, the slope of the
energy on the ODE, and the standard deviations in nats. Validation poses only.
"""
import argparse
import sys
from pathlib import Path

import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from complete import energy_model  # noqa: E402
from fit import REFERENCE, ROTATIONS, SPLIT, draw  # noqa: E402
from hp.evaluation.completion import build  # noqa: E402
from hp.evaluation.data import load_reference  # noqa: E402
from hp.models.io import load_model  # noqa: E402
from hp.models.rsgm import exp_frame  # noqa: E402


def exact_divergence(model):
    """velocity_and_divergence for a model whose score is a gradient taken by autograd (RSGM's uses torch.func, which
    does not go through it): the trace of the Jacobian of the velocity along the tangent directions, by one batched
    backward pass."""
    def velocity_and_divergence(x, t, f=None):
        n, dim = x.shape[0], 3 * model.num_joints
        delta = torch.zeros(n, model.num_joints, 3, device=x.device, dtype=x.dtype, requires_grad=True)
        with torch.enable_grad():
            v = model.velocity(exp_frame(x, delta), t, f)
            one_hot = torch.eye(dim, device=x.device, dtype=x.dtype).reshape(dim, 1, model.num_joints, 3)
            (jac,) = torch.autograd.grad(v, delta, one_hot.expand(dim, n, -1, -1), is_grads_batched=True)
        div = jac.reshape(dim, n, dim).diagonal(dim1=0, dim2=2).sum(-1)
        return v.detach(), div
    return velocity_and_divergence


def log_prob_from(model, x: torch.Tensor, t0: float, steps: int, batch: int, label: str) -> torch.Tensor:
    """log p_{t0}(x) by RSGM.log_prob started at t0 instead of eps."""
    eps, n_steps = model.eps, model.likelihood_steps
    model.eps, model.likelihood_steps = t0, steps
    try:
        out = []
        for i in range(0, len(x), batch):
            out.append(model.log_prob(x[i:i + batch]).cpu().double())
            print(f"\r  {label}: {min(i + batch, len(x))} / {len(x)}", end="", flush=True)
        print(flush=True)
        return torch.cat(out)
    finally:
        model.eps, model.likelihood_steps = eps, n_steps


def compare(a: torch.Tensor, b: torch.Tensor) -> str:
    """a against b: Pearson, Spearman, the slope of a on b, the standard deviations."""
    da, db = a - a.mean(), b - b.mean()
    r = float((da * db).sum() / (da.norm() * db.norm()))
    return (f"{r:.3f} | {spearmanr(a, b)[0]:.3f} | {float((da * db).sum() / (db ** 2).sum()):.3f} | "
            f"{float(a.std()):.1f} | {float(b.std()):.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", default="~/data/experiments/ace/energy_b_smooth_w123_p123.pt")
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--levels", type=float, nargs="+", default=[2e-3, 2e-2, 2e-1], help="noise levels s0")
    parser.add_argument("--poses", type=int, default=200)
    parser.add_argument("--steps", type=int, default=200, help="midpoint steps of the ODE from t0 to 1")
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--check", action="store_true", help="only: the divergence against RSGM's, for the network")
    args = parser.parse_args()

    network = build(dict(run=args.reference), args.device)
    valid = load_reference("valid", ROTATIONS, SPLIT)
    valid = valid[torch.randperm(len(valid), generator=torch.Generator().manual_seed(0))[:args.poses]]
    if args.check:
        x = valid[:4].float().to(args.device)
        for level in args.levels:
            t = network.t_of_s(torch.tensor(level, dtype=torch.float64)).float().to(args.device).expand(4)
            v1, d1 = network.velocity_and_divergence(x, t)
            v2, d2 = exact_divergence(network)(x, t)
            print(f"s {level:g}: divergence {d1.tolist()}\n    by autograd     {d2.tolist()}, "
                  f"velocity error {float((v1 - v2).abs().max()):.1e}")
            assert torch.allclose(d1, d2, rtol=1e-3, atol=1e-3)
        return
    model, energy, info = energy_model(args.fit, args.reference, args.device)
    model.velocity_and_divergence = exact_divergence(model)
    print(f"{info}; {len(valid)} validation poses, {args.steps} ODE steps")
    ref = load_model(Path(args.reference).expanduser(), device="cpu", weights="weights_ema").eval()  # the noising
    rows = []
    for level in args.levels:
        t0 = float(model.t_of_s(torch.tensor(level, dtype=torch.float64)))
        g = torch.Generator().manual_seed(0)
        xt, _, std = draw(ref, valid, torch.full((len(valid),), t0, dtype=torch.float64), g)
        x = xt.float()
        with torch.no_grad():
            e = torch.cat([energy(x[i:i + 500].to(args.device), torch.full((len(x[i:i + 500]),), t0, device=args.device))
                           for i in range(0, len(x), 500)]).cpu().double()
        minus_energy = -e / std[:, 0]
        ode = log_prob_from(model, x, t0, args.steps, args.batch, f"s {level:g}, ODE of the energy")
        net = log_prob_from(network, x, t0, args.steps, args.batch, f"s {level:g}, ODE of the network")
        rows += [(level, "-E against its ODE", compare(minus_energy, ode)),
                 (level, "-E against the network's ODE", compare(minus_energy, net)),
                 (level, "ODE of the energy against the network's", compare(ode, net))]
        print(f"s {level:g}: mean log p by the ODE of the energy {float(ode.mean()):.2f}, of the network "
              f"{float(net.mean()):.2f} nats", flush=True)
    print("| s0 | a against b | Pearson | Spearman | slope | std a | std b |\n|---|---|---|---|---|---|---|")
    for level, name, text in rows:
        print(f"| {level:g} | {name} | {text} |")


if __name__ == "__main__":
    main()
