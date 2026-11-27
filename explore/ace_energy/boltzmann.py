"""Maximum likelihood for the pairwise matrix-Fisher energy of pseudo_lik.py: a Boltzmann machine on SO(3)^19 trained
by persistent contrastive divergence. The gradient of the mean negative log-likelihood is

    E_data[grad E] - E_model[grad E],

the model's expectation taken over persistent Gibbs chains (gibbs.py: every joint drawn from its exact conditional),
one sweep of all chains per step. Unlike the pseudo-likelihood, which sees only configurations where all but one
joint are real, the chains wander off the data and the gradient raises the energy where they settle wrongly (the
holes). No normalizer is needed, so one-body terms of higher degree can be added:

    E(x) = E_pairwise(x) - sum_j sum_{l in degrees} <C_jl, D^l(x_j)>,

the real Wigner D-matrices of hp.models.so3_harmonics (--degrees 2 3: 74 coefficients per joint, the one-body part of
fit.py --wigner 1 2 3). Their conditionals are exp(<F_j, x_j> + g_j(x_j)), drawn by Metropolis steps with the matrix
Fisher as proposal (gibbs.gibbs). Start: a fit of pseudo_lik.py (C = 0); the chains start from uniform rotations and
take --burn sweeps under the starting energy. A ridge on the pair blocks B and on C. Adam with a cosine schedule.
Checkpoints every --every steps (the same file). Validation numbers come from gibbs.py --generate / --inpaint --fit.

From the root of the repository:

    python explore/ace_energy/boltzmann.py --out ~/data/experiments/ace/pl/energy_bm.pt                  # D^1, CPU
    python explore/ace_energy/boltzmann.py --degrees 2 3 --device cuda:0 --out ~/data/experiments/ace/pl/energy_bm_w123.pt
"""
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pseudo_lik import ACE, PairwiseFisher  # noqa: E402
from hp.models.so3_harmonics import wigner_d  # noqa: E402

J = 19


class Boltzmann(PairwiseFisher):
    """The pairwise energy with one-body terms of the Wigner degrees `degrees` (none: PairwiseFisher itself)."""

    def __init__(self, degrees=()):
        super().__init__()
        self.degrees = tuple(int(d) for d in degrees)
        self.c = torch.nn.Parameter(torch.zeros(J, sum((2 * l + 1) ** 2 for l in self.degrees), dtype=torch.float64))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """The entries of D^l(x) for the degrees, [..., P], of rotations x [..., 3, 3]."""
        return torch.cat([wigner_d(x, l).flatten(-2) for l in self.degrees], -1)

    def one_body(self, xj: torch.Tensor, j: int) -> torch.Tensor:
        """g_j [B] of joint j at rotations xj [B, 3, 3]."""
        return self.features(xj.to(self.c)) @ self.c[j]

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        e = super().energy(x)
        if not self.degrees:
            return e
        flat = x.reshape(-1, J, 3, 3).to(self.c)
        return e - (self.features(flat) * self.c).sum((-1, -2)).reshape(x.shape[:-3])

    def ridge(self) -> torch.Tensor:
        return super().ridge() + (self.c ** 2).sum()


def main() -> None:
    from fit import ROTATIONS, SPLIT  # noqa: E402
    from gibbs import gibbs, uniform  # noqa: E402
    from hp.evaluation.data import load_reference  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", default=str(ACE / "pl" / "energy_pl.pt"), help="a fit of pseudo_lik.py")
    parser.add_argument("--degrees", type=int, nargs="*", default=[], help="one-body Wigner degrees (e.g. 2 3)")
    parser.add_argument("--n", type=int, default=200_000, help="training poses the batches are drawn from")
    parser.add_argument("--chains", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=2000, help="training poses per step")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--burn", type=int, default=20, help="sweeps of the chains before the first step")
    parser.add_argument("--mh", type=int, default=3, help="Metropolis steps per update (one-body terms only)")
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--ridge", type=float, default=None, help="default: that of the start")
    parser.add_argument("--every", type=int, default=250)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    dev = torch.device(args.device)
    start = torch.load(Path(args.init).expanduser(), weights_only=True)
    ridge = start["ridge"] if args.ridge is None else args.ridge
    model = Boltzmann(args.degrees)
    model.load_state_dict(start["state"], strict=False)
    model.to(dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    cpu = torch.Generator().manual_seed(args.seed)
    train = load_reference("train", ROTATIONS, SPLIT)
    train = train[torch.randperm(len(train), generator=cpu)[:args.n]].double().to(dev)
    valid = load_reference("valid", ROTATIONS, SPLIT).double().to(dev)
    print(f"start {args.init} (ridge {ridge}); one-body degrees {args.degrees}; {len(train):,} training poses, "
          f"{args.chains} chains, {args.steps} steps of {args.batch} poses, lr {args.lr}", flush=True)
    chains = gibbs(model, uniform(args.chains * J, gen).reshape(args.chains, J, 3, 3), range(J), args.burn, gen,
                   mh=args.mh)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, accept = time.time(), []
    for step in range(1, args.steps + 1):
        x = train[torch.randint(len(train), (args.batch,), generator=gen, device=dev)]
        chains = gibbs(model, chains, range(J), 1, gen, mh=args.mh, stats=accept)
        e_data, e_model = model.energy(x).mean(), model.energy(chains).mean()
        loss = e_data - e_model + 0.5 * ridge * model.ridge()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % args.every == 0 or step == args.steps:
            with torch.no_grad():
                e_valid = float(model.energy(valid).mean())
            acc = f", Metropolis accepted {sum(accept) / len(accept):.2f}" if accept else ""
            accept.clear()
            print(f"step {step}: E data {float(e_data):.1f}, valid {e_valid:.1f}, chains {float(e_model):.1f} "
                  f"(gap {float(e_model - e_data):+.1f}){acc}, |B| {float(PairwiseFisher.ridge(model)) ** 0.5:.0f}, "
                  f"{(time.time() - t0) / step:.2f} s/step", flush=True)
            torch.save(dict(state={k: v.cpu() for k, v in model.state_dict().items()}, ridge=ridge,
                            degrees=list(args.degrees), step=step, init=args.init, settings=vars(args)), out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
