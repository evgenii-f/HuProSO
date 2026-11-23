"""A per-joint ACE energy with a Finnis-Sinclair embedding, trained by denoising score matching (the step after the
linear energy of fit.py; docs/experiments/2026-11-23_ace_descriptors.md). From the root of the repository:

    python explore/ace_energy/embedding.py --check                     # the energy at init is the linear fit
    python explore/ace_energy/embedding.py --init ~/data/experiments/ace/energy_b_smooth_w123_p123.pt \\
        --device cuda:0 --out ~/data/experiments/ace/energy_fs.pt

Model. Every feature B_f of fit.py's basis (smooth positions, D^l(x_j), pair harmonics) belongs to the joints whose
rotations it depends on most directly (`owners`): a pair of points to the joints that place the two points, a triple
to the joint that places its centre, one-body terms to their joint, a pair harmonic to its two joints. Every joint j
has K = 2 densities, linear in its features, and an energy through the embedding F:

    phi_j^(k)(x, t) = sum_{f in N(j)} c_jf^(k)(t) B_f(x),     E~_t(x) = sum_j [ phi_j^(1) + sqrt(1 + phi_j^(2)^2) - 1 ]

(a smooth Finnis-Sinclair-like embedding: quadratic near 0, |phi| far from it). E~_t = std(t) E_t, so the scaled score
of RSGM's loss is -grad E~_t, taken by autograd along the tangent directions x_j exp(eps e_a). The coefficients are
per bin of log s and interpolated linearly between bin centres, as in fit.py. Training: Adam on RSGM's loss (std
weighting, t uniform on [t_min, 1]), the noise drawn by the reference run's noising, as fit.py.

Init: phi^(1) from a linear fit of fit.py (--init, with its basis), each feature's coefficient on its first owner, and
phi^(2) small; then the energy starts as that linear fit (--check verifies it on the validation draws).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit import COMPARE, REFERENCE, ROTATIONS, SPLIT, Bins, WithHarmonics, draw  # noqa: E402
from hp.evaluation.data import load_reference  # noqa: E402
from hp.models.ace_features import LEARNED, AceFeatures  # noqa: E402
from hp.models.io import load_model  # noqa: E402
from hp.models.rsgm import exp_frame  # noqa: E402
from hp.models.so3_harmonics import PAIR_INDEX  # noqa: E402


def owners(basis: WithHarmonics) -> list:
    """For every feature of the basis, in its layout, the joints (indices into the 19 of x) it belongs to."""
    ace = basis.ace
    parents = ace.parents

    def owner(point):  # the joint whose rotation places the point; None for the root and the points fixed to it
        p = parents[point]
        return LEARNED.index(p) if p in LEARNED else None

    def both(*points):
        return sorted({o for o in (owner(p) for p in points) if o is not None})

    out = []
    pair_owners = [both(j, k) for j, k in ace.pairs]
    out += pair_owners * ace.radial  # T_1..T_radial, each a block over the pairs
    tri = [[owner(j)] if owner(j) is not None else both(k, q) for k, j, q in ace.triples]
    out += tri * (ace.angular * ace.triple_radial ** 2)
    if ace.twist_joints:
        out += [[LEARNED.index(j)] for j in ace.twist_joints] * (2 * ace.twist)
    for l in basis.wigner:  # D^l(x_j): [19, 2l+1, 2l+1]
        out += [[j] for j in range(19) for _ in range((2 * l + 1) ** 2)]
    for l in basis.pairs:  # D^l(G_j) of every joint
        out += [[j] for j in range(19) for _ in range((2 * l + 1) ** 2)]
    for l in basis.pairs:  # D^l(G_i^T G_k) of the pairs
        out += [[i, k] for i, k in PAIR_INDEX for _ in range((2 * l + 1) ** 2)]
    assert len(out) == basis.dim, (len(out), basis.dim)
    return out


class Embedded(torch.nn.Module):
    """E~_t(x) [B] (module doc): per-joint densities of the features and a Finnis-Sinclair-like embedding."""

    def __init__(self, basis, bins: Bins, own: list, k: int = 2):
        super().__init__()
        self.basis, self.bins = basis, bins
        feat = [f for f, js in enumerate(own) for _ in js]
        joint = [j for js in own for j in js]
        self.register_buffer("feat", torch.tensor(feat))
        self.register_buffer("joint", torch.tensor(joint))
        self.register_buffer("first", torch.tensor([i == 0 for js in own for i in range(len(js))]))
        n = len(bins.centres)
        self.c = torch.nn.Parameter(torch.zeros(k, n, len(feat)))  # [K, bins, entries]

    def init_linear(self, w: torch.Tensor, scale2: float = 1e-3) -> None:
        """phi^(1) = the linear fit w [bins, P] on the first owner of every feature; phi^(2) small."""
        with torch.no_grad():
            self.c[0] = w[:, self.feat.cpu()].to(self.c) * self.first.to(self.c)
            self.c[1:] = scale2 * torch.randn_like(self.c[1:]) * w[:, self.feat.cpu()].abs().mean().to(self.c)

    def coefficients(self, t: torch.Tensor) -> torch.Tensor:
        """c(t) [K, B, entries], linear in log s between bin centres."""
        b = self.bins
        u = b.ref.s(t.double()).log()
        pos = ((u - b.centres[0]) / (b.centres[1] - b.centres[0])).clamp(0, len(b.centres) - 1)
        lo = pos.floor().long().clamp(max=len(b.centres) - 2)
        a = (pos - lo).to(self.c)[None, :, None]
        lo = lo.to(self.c.device)
        return (1 - a) * self.c[:, lo] + a * self.c[:, lo + 1]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        f = self.basis(x)[:, self.feat]  # [B, entries]
        c = self.coefficients(t)  # [K, B, entries]
        phi = torch.zeros(c.shape[0], x.shape[0], 19, device=x.device, dtype=x.dtype)
        phi = phi.index_add(2, self.joint, c * f[None])  # [K, B, 19]
        energy = phi[0] + (torch.sqrt(1 + phi[1:] ** 2) - 1).sum(0)
        return energy.sum(-1)

    def scaled_score(self, x: torch.Tensor, t: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        """-grad E~_t [B, J, 3] along the tangent directions (the scaled score of RSGM's loss)."""
        with torch.enable_grad():
            delta = torch.zeros(x.shape[0], x.shape[1], 3, device=x.device, dtype=x.dtype, requires_grad=True)
            e = self(exp_frame(x, delta), t).sum()
            (g,) = torch.autograd.grad(e, delta, create_graph=create_graph)
        return -g


def losses(model, xt, target, t, batch) -> torch.Tensor:
    out = []
    for i in range(0, len(xt), batch):
        sl = slice(i, i + batch)
        pred = model.scaled_score(xt[sl], t[sl]).detach().double().cpu()
        out.append(((pred - target[sl]) ** 2).sum((1, 2)))
    return torch.cat(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", default="~/data/experiments/ace/energy_b_smooth_w123_p123.pt",
                        help="a linear fit of fit.py; its basis (--wigner, --pairs) is taken from it")
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--compare", nargs="*", default=COMPARE[:1])
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--valid", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--check", action="store_true", help="only: the energy at init against the linear fit")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    saved = torch.load(Path(args.init).expanduser(), weights_only=False)
    s = saved["settings"]
    ref = load_model(Path(args.reference).expanduser(), device="cpu", weights="weights_ema").eval()
    basis = WithHarmonics(AceFeatures(smooth=not s.get("plain", False)), s.get("wigner", []), s.get("pairs", []))
    bins = Bins(ref, ref.loss_t_min, s["bins"])
    assert torch.equal(saved["edges"], bins.edges)
    model = Embedded(basis, bins, owners(basis)).to(device)
    model.init_linear(saved["w"])
    print(f"{basis.dim} features, {model.c.numel():,} coefficients ({model.c.shape[2]} joint-feature entries x "
          f"{model.c.shape[1]} bins x K = {model.c.shape[0]})", flush=True)

    valid = load_reference("valid", ROTATIONS, SPLIT)
    valid = valid[torch.randperm(len(valid), generator=torch.Generator().manual_seed(0))[:args.valid]]
    g = torch.Generator().manual_seed(0)  # the validation draws of fit.py
    tv = ref.loss_t_min + (1 - ref.loss_t_min) * torch.rand(len(valid), dtype=torch.float64, generator=g)
    xv, yv, stdv = draw(ref, valid, tv, g)
    xv_d, tv_d = xv.float().to(device), tv.float().to(device)

    def evaluate():
        loss = losses(model, xv_d, yv, tv_d, 500)
        k = bins.index(tv)
        return float(loss.mean()), [float(loss[k == i].mean()) for i in range(len(bins.centres))], float(loss.median())

    if args.check:
        mean, per, med = evaluate()
        print(f"init: valid loss {mean:.3f} (median {med:.3f}); the linear fit: {saved['table']['ACE energy']['mean']:.3f}")
        return

    compare = {}
    for r in args.compare:
        net = load_model(Path(r).expanduser(), device=args.device, weights="weights_ema").eval()
        with torch.no_grad():
            pred = torch.cat([net.score(xv_d[i:i + 500], tv_d[i:i + 500]).double().cpu() for i in range(0, len(xv), 500)])
        compare[Path(r).expanduser().parent.name] = float((((pred * stdv[..., None]) - yv) ** 2).sum((1, 2)).mean())
    print("networks on the same draws:", compare, flush=True)

    train = load_reference("train", ROTATIONS, SPLIT)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    gen = torch.Generator().manual_seed(args.seed + 1)
    history = []
    mean, per, med = evaluate()
    print(f"step 0: valid loss {mean:.3f} (median {med:.3f})", flush=True)
    history.append(dict(step=0, valid=mean, median=med, per_bin=per))
    t0, running = time.perf_counter(), []
    for step in range(1, args.steps + 1):
        x0 = train[torch.randint(len(train), (args.batch,), generator=gen)]
        t = ref.loss_t_min + (1 - ref.loss_t_min) * torch.rand(args.batch, dtype=torch.float64, generator=gen)
        xt, target, _ = draw(ref, x0, t, gen)
        pred = model.scaled_score(xt.float().to(device), t.float().to(device), create_graph=True)
        loss = ((pred - target.float().to(device)) ** 2).sum((1, 2)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running.append(float(loss))
        if step % args.eval_every == 0 or step == args.steps:
            mean, per, med = evaluate()
            history.append(dict(step=step, valid=mean, median=med, per_bin=per, train=sum(running) / len(running)))
            print(f"step {step}: train {sum(running) / len(running):.3f}, valid loss {mean:.3f} (median {med:.3f}), "
                  f"{(time.perf_counter() - t0) / step:.2f} s/step", flush=True)
            running = []
    edges = [f"{bins.edges[i].exp():.0e}" for i in range(len(bins.centres))]
    print("| model | mean | " + " | ".join(f"s {e}" for e in edges) + " |")
    print(f"| FS energy | {mean:.3f} | " + " | ".join(f"{v:.2f}" for v in per) + " |")
    lin = saved["table"]["ACE energy"]
    print(f"| linear energy (--init) | {lin['mean']:.3f} | " + " | ".join(f"{v:.2f}" for v in lin["per_bin"]) + " |")
    for name in saved["table"]:
        if name.startswith("rsgm"):
            r = saved["table"][name]
            print(f"| {name} | {r['mean']:.3f} | " + " | ".join(f"{v:.2f}" for v in r["per_bin"]) + " |")
    if args.out:
        out = Path(args.out).expanduser()
        torch.save(dict(c=model.c.detach().cpu(), edges=bins.edges, history=history, settings=vars(args),
                        basis=dict(wigner=basis.wigner, pairs=basis.pairs)), out)
        out.with_suffix(".json").write_text(json.dumps(dict(history=history, settings=vars(args), compare=compare),
                                                       indent=1))
        print(f"saved to {out}")


if __name__ == "__main__":
    main()
