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
Fisher as proposal (gibbs.gibbs). --pairs adds pair terms of degree 2 on the edges of the kinematic tree (EDGES: every
joint with its parent, and the three children of the root with each other), u = (x, D^2(x)) of the two joints:

    - sum_{(p, k) in EDGES} [ x_p^T Q12 D^2(x_k) + D^2(x_p)^T Q21 x_k + D^2(x_p)^T Q22 D^2(x_k) ]

(1075 coefficients per edge; needs 2 among --degrees). The conditional of joint j stays exp(<h_j, (x_j, D^{2,3}(x_j))>):
the pair terms add to F_j and to the coefficients of D^2(x_j), with the neighbours' x and D^2(x). Start: a fit of pseudo_lik.py (C = 0); the chains start from uniform rotations and
take --burn sweeps under the starting energy. A ridge on the pair blocks B and on C. Adam with a cosine schedule.
Checkpoints every --every steps (the same file). Validation numbers come from gibbs.py --generate / --inpaint --fit.

From the root of the repository:

    python explore/ace_energy/boltzmann.py --out ~/data/experiments/ace/pl/energy_bm.pt                  # D^1, CPU
    python explore/ace_energy/boltzmann.py --degrees 2 3 --device cuda:0 --out ~/data/experiments/ace/pl/energy_bm_w123.pt
    python explore/ace_energy/boltzmann.py --check                 # the conditionals against energy differences
    python explore/ace_energy/boltzmann.py --init ~/data/experiments/ace/pl/energy_bm_w123.pt --degrees 2 3 --pairs \
        --chains 2000 --batch 2000 --steps 2000 --mh 1 --out ~/data/experiments/ace/pl/energy_bm_pairs.pt
    python explore/ace_energy/boltzmann.py --init ~/data/experiments/ace/pl/energy_bm_w123.pt --degrees 2 3 --pairs \
        --chains 2000 --batch 2000 --steps 2000 --mh 1 --burn 1000 --proposal linear \
        --out ~/data/experiments/ace/pl/energy_bm_pairs_lin.pt

The last checkpoint keeps the persistent chains (gibbs.py --generate --from-chains starts from them; a diagnostic,
not a measure of the model: with poor mixing they stay near the data while the model's equilibrium does not).
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
from hp.models.rsgm import PARENTS19  # noqa: E402
from hp.models.so3_harmonics import wigner_d  # noqa: E402

J = 19
ROOTS = [j for j, p in enumerate(PARENTS19) if p < 0]
EDGES = [(p, j) for j, p in enumerate(PARENTS19) if p >= 0] + [(a, b) for i, a in enumerate(ROOTS) for b in ROOTS[i + 1:]]


class Boltzmann(PairwiseFisher):
    """The pairwise energy with one-body terms of the Wigner degrees `degrees` (none: PairwiseFisher itself)."""

    def __init__(self, degrees=(), pairs=False):
        super().__init__()
        self.degrees = tuple(int(d) for d in degrees)
        self.c = torch.nn.Parameter(torch.zeros(J, sum((2 * l + 1) ** 2 for l in self.degrees), dtype=torch.float64))
        self.pairs = bool(pairs)
        assert not self.pairs or 2 in self.degrees, "--pairs needs degree 2 among the one-body degrees"
        start = sum((2 * l + 1) ** 2 for l in self.degrees[:self.degrees.index(2)]) if self.pairs else 0
        self.d2 = slice(start, start + 25)  # D^2 among the features
        e = len(EDGES) if self.pairs else 0
        z = lambda *shape: torch.nn.Parameter(torch.zeros(e, *shape, dtype=torch.float64))  # noqa: E731
        self.q12, self.q21, self.q22 = z(9, 25), z(25, 9), z(25, 25)
        self.touch = {j: [(i, p, k) for i, (p, k) in enumerate(EDGES) if j in (p, k)] for j in range(J)}

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
        feats = self.features(flat)
        e = e - (feats * self.c).sum((-1, -2)).reshape(x.shape[:-3])
        if not self.pairs:
            return e
        d1, d2 = flat.flatten(-2), feats[..., self.d2]
        p, k = [list(t) for t in zip(*EDGES)]
        t = (torch.einsum("nea,eab,neb->n", d1[:, p], self.q12, d2[:, k])
             + torch.einsum("nea,eab,neb->n", d2[:, p], self.q21, d1[:, k])
             + torch.einsum("nea,eab,neb->n", d2[:, p], self.q22, d2[:, k]))
        return e - t.reshape(x.shape[:-3])

    def pair_fields(self, d1: torch.Tensor, d2: torch.Tensor, j: int):
        """What the pair terms add to F_j [B, 9] and to the coefficients of D^2(x_j) [B, 25], of the entries d1
        [B, 19, 9] and D^2 features d2 [B, 19, 25] of all joints."""
        df = torch.zeros(len(d1), 9, dtype=d1.dtype, device=d1.device)
        dc = torch.zeros(len(d1), 25, dtype=d1.dtype, device=d1.device)
        for i, p, k in self.touch[j]:
            if p == j:
                df = df + d2[:, k] @ self.q12[i].T
                dc = dc + d1[:, k] @ self.q21[i].T + d2[:, k] @ self.q22[i].T
            else:
                df = df + d2[:, p] @ self.q21[i]
                dc = dc + d1[:, p] @ self.q12[i] + d2[:, p] @ self.q22[i]
        return df, dc

    def ridge(self) -> torch.Tensor:
        return super().ridge() + (self.c ** 2).sum() + sum((q ** 2).sum() for q in (self.q12, self.q21, self.q22))


def _forms(l: int) -> torch.Tensor:
    """The entries of D^l(x) as homogeneous forms of degree l in the 9 entries of x: [(2l+1)^2, 9^l] with
    D^l(x)_k = <A_k, x (x) ... (x) x> on SO(3) (fitted by least squares on random rotations; the residual is printed by
    --check). A form that vanishes on SO(3) has its gradient normal to it, so the tangent slope does not depend on
    the choice."""
    from gibbs import uniform  # noqa: E402

    x = uniform(4000, torch.Generator().manual_seed(1)).reshape(-1, 9)
    mono = x
    for _ in range(l - 1):
        mono = (mono[:, :, None] * x[:, None, :]).reshape(len(x), -1)
    target = wigner_d(x.reshape(-1, 3, 3), l).flatten(-2)
    return torch.linalg.lstsq(mono, target, driver="gelsd").solution.T.contiguous()


_FORMS = {}


def forms(l: int) -> torch.Tensor:
    if l not in _FORMS:
        _FORMS[l] = _forms(l)
    return _FORMS[l]


class Poly:
    """g(x) = <coef, (D^2, D^3)(x)> of one joint as a quadratic form Q [B, 9, 9] (per chain) and a cubic form T
    [9, 9, 9] (the same for all chains: the D^3 coefficients have no pair terms), symmetrized: g = v^T Q v + T(v, v, v),
    its gradient 2 Q v + 3 T(., v, v), and the D^2 features v^T A_k v, without autograd (v the 9 entries of x)."""

    def __init__(self, model, coef: torch.Tensor, j: int):
        assert model.degrees == (2, 3), "the forms are written for one-body degrees 2 3"
        a2, a3 = forms(2).to(coef), forms(3).to(coef)
        self.a2 = a2.reshape(25, 9, 9)
        q = (coef[:, :25] @ a2).reshape(-1, 9, 9)
        self.q = 0.5 * (q + q.transpose(-1, -2))
        t = (model.c[j, 25:].to(coef) @ a3).reshape(9, 9, 9)
        self.t = (t + t.permute(0, 2, 1) + t.permute(1, 0, 2) + t.permute(1, 2, 0) + t.permute(2, 0, 1)
                  + t.permute(2, 1, 0)) / 6

    def __call__(self, x: torch.Tensor):
        """g [B], the tangent slope [B, 3, 3] and the D^2 features [B, 25] of rotations x [B, 3, 3]."""
        v = x.reshape(len(x), 9)
        qv = (self.q @ v[:, :, None])[..., 0]
        tvv = torch.einsum("ijk,bj,bk->bi", self.t, v, v)
        g = (qv * v).sum(-1) + (tvv * v).sum(-1)
        grad = (2 * qv + 3 * tvv).reshape(-1, 3, 3)
        s = x.transpose(-1, -2) @ grad
        d2 = torch.einsum("kij,bi,bj->bk", self.a2, v, v)
        return g, x @ (0.5 * (s - s.transpose(-1, -2))), d2


def check() -> None:
    """The conditional of gibbs.local against differences of the energy: for random pair and one-body
    coefficients, E(x with x_j = r1) - E(x with x_j = r2) = -(<h_j, phi(r1)> - <h_j, phi(r2)>)."""
    from gibbs import local, uniform  # noqa: E402

    gen = torch.Generator().manual_seed(0)
    model = Boltzmann((2, 3), pairs=True)
    with torch.no_grad():
        for prm in model.parameters():
            prm.copy_(torch.randn(prm.shape, dtype=prm.dtype, generator=gen))
    model.requires_grad_(False)
    x = uniform(8 * J, gen).reshape(8, J, 3, 3)
    w, a = model.coupling(), model.a
    d2 = model.features(x)[..., model.d2]
    worst = 0.0
    for j in range(J):
        f, coef = local(model, x, d2, w, a, j)
        r1, r2 = uniform(8, gen), uniform(8, gen)
        x1, x2 = x.clone(), x.clone()
        x1[:, j], x2[:, j] = r1, r2
        lin = lambda r: (f * r).sum((-1, -2)) + (model.features(r) * coef).sum(-1)  # noqa: E731
        diff = (model.energy(x1) - model.energy(x2)) + (lin(r1) - lin(r2))
        worst = max(worst, float(diff.abs().max()))
    old = Boltzmann((2, 3))
    old.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("q")}, strict=False)
    model.q12.zero_(), model.q21.zero_(), model.q22.zero_()
    from gibbs import slope  # noqa: E402

    xs = uniform(200, gen)
    coef = torch.randn(200, 74, dtype=torch.float64, generator=gen)
    coef[:, 25:] = model.c[3, 25:]
    g, s, d2 = Poly(model, coef, 3)(xs)
    g_ref, feat_ref, s_ref = slope(model, xs, coef)
    print(f"forms: max |g - autograd| {float((g - g_ref).abs().max()):.1e}, slope {float((s - s_ref).abs().max()):.1e}, "
          f"D^2 {float((d2 - feat_ref[:, :25]).abs().max()):.1e}")
    print(f"{len(EDGES)} edges; max |conditional - energy difference| = {worst:.2e} (nats, random coefficients); "
          f"zero pair terms: max |E - E without pairs| = {float((model.energy(x) - old.energy(x)).abs().max()):.2e}")


def main() -> None:
    from fit import ROTATIONS, SPLIT  # noqa: E402
    from gibbs import gibbs, uniform  # noqa: E402
    from hp.evaluation.data import load_reference  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", default=str(ACE / "pl" / "energy_pl.pt"), help="a fit of pseudo_lik.py")
    parser.add_argument("--degrees", type=int, nargs="*", default=[], help="one-body Wigner degrees (e.g. 2 3)")
    parser.add_argument("--pairs", action="store_true", help="pair terms of degree 2 on the edges of the tree")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--n", type=int, default=200_000, help="training poses the batches are drawn from")
    parser.add_argument("--chains", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=2000, help="training poses per step")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--burn", type=int, default=20, help="sweeps of the chains before the first step")
    parser.add_argument("--mh", type=int, default=3, help="Metropolis steps per update (one-body terms only)")
    parser.add_argument("--proposal", choices=["fisher", "linear"], default="fisher",
                        help="Metropolis proposal: the matrix Fisher of the D^1 part, or linearized at the state")
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--ridge", type=float, default=None, help="default: that of the start")
    parser.add_argument("--every", type=int, default=250)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.check:
        check()
        return
    assert args.out, "--out is needed"
    dev = torch.device(args.device)
    start = torch.load(Path(args.init).expanduser(), weights_only=True)
    ridge = start["ridge"] if args.ridge is None else args.ridge
    model = Boltzmann(args.degrees, args.pairs)
    missing, unexpected = model.load_state_dict(start["state"], strict=False)
    assert not unexpected and set(missing) <= {"c", "q12", "q21", "q22"}, (missing, unexpected)
    model.to(dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    cpu = torch.Generator().manual_seed(args.seed)
    train = load_reference("train", ROTATIONS, SPLIT)
    train = train[torch.randperm(len(train), generator=cpu)[:args.n]].double().to(dev)
    valid = load_reference("valid", ROTATIONS, SPLIT).double().to(dev)
    print(f"start {args.init} (ridge {ridge}); one-body degrees {args.degrees}; pair terms on {len(EDGES)} edges: "
          f"{args.pairs}; {len(train):,} training poses, "
          f"{args.chains} chains, {args.steps} steps of {args.batch} poses, lr {args.lr}", flush=True)
    chains = gibbs(model, uniform(args.chains * J, gen).reshape(args.chains, J, 3, 3), range(J), args.burn, gen,
                   mh=args.mh, proposal=args.proposal)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, accept = time.time(), []
    for step in range(1, args.steps + 1):
        x = train[torch.randint(len(train), (args.batch,), generator=gen, device=dev)]
        chains = gibbs(model, chains, range(J), 1, gen, mh=args.mh, stats=accept, proposal=args.proposal)
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
                  f"|Q| {float(sum((q ** 2).sum() for q in (model.q12, model.q21, model.q22))) ** 0.5:.1f}, "
                  f"{(time.time() - t0) / step:.2f} s/step", flush=True)
            torch.save(dict(state={k: v.cpu() for k, v in model.state_dict().items()}, ridge=ridge,
                            degrees=list(args.degrees), pairs=args.pairs, step=step, init=args.init,
                            settings=vars(args), chains=chains.float().cpu()), out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
