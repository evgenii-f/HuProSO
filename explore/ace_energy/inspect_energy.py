"""Helpers of bins.ipynb: load an ACE energy (a linear fit of fit.py or a trained embedding of embedding.py), draw the
validation noise as fit.py does, compute the loss of every pose for the energy and a network, and split the energy's
predicted score into its feature groups. As a script, the losses of several energies in the bins of low noise, where
the draws of fit.py have only about a hundred poses per bin:

    python explore/ace_energy/inspect_energy.py ~/data/experiments/ace/energy_b_ridge_r0.001.pt ...
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "src", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from embedding import Embedded, owners  # noqa: E402
from joint_density import JointDensity  # noqa: E402
from fit import (JOINT_PAIRS, OTHER_PAIRS, REFERENCE, ROTATIONS, SPLIT, SQUARE_DIM, TRIPLE_DIM, Bins,  # noqa: E402
                 draw, harmonic_basis, jacobian)
from hp.evaluation.data import load_reference  # noqa: E402
from hp.models.ace_features import AceFeatures  # noqa: E402
from hp.models.io import load_model  # noqa: E402
from hp.models.so3_harmonics import WIGNER_PAIRS  # noqa: E402

FOLDER = Path("~/data/experiments/ace").expanduser()


def _local(path) -> Path:
    """A path written on a server, found here: as it is, else by its name in FOLDER."""
    p = Path(path).expanduser()
    return p if p.exists() else FOLDER / p.name


def load_energy(path, device: str = "cpu"):
    """(energy, ref, bins, info): the energy as an Embedded model (a linear fit has its second density zero), the
    reference RSGM run of the noising, the bins of log s, and a short description."""
    saved = torch.load(_local(path), map_location="cpu", weights_only=True)
    ref = load_model(Path(REFERENCE).expanduser(), device="cpu", weights="weights_ema").eval()
    if "c" in saved:  # embedding.py (its first runs saved the coefficients c only, the FS embedding)
        args = saved["settings"]
        init = torch.load(_local(args["init"]), map_location="cpu", weights_only=True)
        s = init["settings"]
        basis = harmonic_basis(s, init.get("triples"))
        bins = Bins(ref, ref.loss_t_min, s["bins"])
        energy = Embedded(basis, bins, owners(basis), k=args.get("k", 2), embedding=args.get("embedding", "fs"),
                          hidden=args.get("hidden", 64))
        if "state" in saved:
            energy.load_state_dict(saved["state"])
        else:
            with torch.no_grad():
                energy.c.copy_(saved["c"])
        info = f"{args.get('embedding', 'fs')} embedding, K = {args.get('k', 2)}, basis {basis.dim}"
    else:  # fit.py
        s = saved["settings"]
        if s.get("one_body"):
            raise ValueError("fits with --one-body are not supported; use the --wigner fits")
        basis = harmonic_basis(s, saved.get("triples"))
        bins = Bins(ref, ref.loss_t_min, s["bins"])
        energy = Embedded(basis, bins, owners(basis))
        energy.init_linear(saved["w"], scale2=0.0)
        info = f"linear fit, basis {basis.dim}"
        if "prior" in saved:
            energy.prior = JointDensity(saved["prior"]["centres"], saved["prior"]["s0"])
            info += f", joint density of {saved['prior']['centres'].shape[-1]} poses"
    energy.requires_grad_(False)
    return energy.to(device).eval(), ref, bins, info


def validation_draws(ref, n: int = 5000) -> dict:
    """The validation poses and the one draw per pose of fit.py (seed 0): x0, t, xt, target, std."""
    valid = load_reference("valid", ROTATIONS, SPLIT)
    valid = valid[torch.randperm(len(valid), generator=torch.Generator().manual_seed(0))[:n]]
    g = torch.Generator().manual_seed(0)
    t = ref.loss_t_min + (1 - ref.loss_t_min) * torch.rand(len(valid), dtype=torch.float64, generator=g)
    xt, target, std = draw(ref, valid, t, g)
    return dict(x0=valid, t=t, xt=xt, target=target, std=std, s=ref.s(t))


def energy_predictions(energy, d: dict, batch: int = 500) -> torch.Tensor:
    """The energy's scaled scores [N, J, 3] at the draws."""
    dev = next(energy.parameters()).device
    out = []
    for i in range(0, len(d["t"]), batch):
        out.append(energy.scaled_score(d["xt"][i:i + batch].float().to(dev),
                                       d["t"][i:i + batch].float().to(dev)).cpu().double())
    return torch.cat(out)


@torch.no_grad()
def network_predictions(net, d: dict, batch: int = 500) -> torch.Tensor:
    """A network's scaled scores [N, J, 3] at the draws."""
    out = []
    for i in range(0, len(d["t"]), batch):
        sl = slice(i, i + batch)
        out.append(net.score(d["xt"][sl].float().to(net.device), d["t"][sl].float().to(net.device)).cpu().double()
                   * d["std"][sl][..., None])
    return torch.cat(out)


def stratified_draws(ref, bins, k: int, pool: torch.Tensor, n: int, seed: int) -> dict:
    """One draw for each of n random poses of the pool at noise levels of bin k (t uniform within it, as fit.py)."""
    g = torch.Generator().manual_seed(1000 * seed + k)
    x0 = pool[torch.randperm(len(pool), generator=g)[:n]]
    lo, hi = bins.t_range(k)
    t = lo + (hi - lo) * torch.rand(len(x0), dtype=torch.float64, generator=g)
    xt, target, std = draw(ref, x0, t, g)
    return dict(x0=x0, t=t, xt=xt, target=target, std=std, s=ref.s(t))


def low_noise_table(paths, device: str, s_max: float, n: int, batch: int) -> None:
    """Losses in the bins below s_max, of fresh train poses and of validation poses, the same draws for every model;
    the last column is the mean over the bins, with the standard error of its paired difference to the network (the
    poses of the bins as independent). "in-sample" is the fit's own loss on the poses it was fitted to."""
    _, ref, bins, _ = load_energy(paths[0])
    ks = [k for k in range(len(bins.centres)) if float(bins.edges[k + 1].exp()) <= s_max]
    net = load_model(Path(REFERENCE).expanduser(), device=device, weights="weights_ema").eval()
    draws = {}
    for seed, split in enumerate(("train", "valid")):
        pool = load_reference(split, ROTATIONS, SPLIT)
        draws[split] = [stratified_draws(ref, bins, k, pool, n, seed) for k in ks]
    loss = lambda pred, d: ((pred - d["target"]) ** 2).sum((1, 2))  # noqa: E731
    base = {split: [loss(network_predictions(net, d, batch), d) for d in ds] for split, ds in draws.items()}
    print("| model | poses | " + " | ".join(f"s {bins.centres[k].exp():.1e}" for k in ks)
          + " | mean | - network | s.e. |")
    print("|---|---|" + "---|" * (len(ks) + 3), flush=True)

    def row(name, split, per):
        means = [float(v.mean()) for v in per]
        diff = [v - b for v, b in zip(per, base[split])]
        gap = sum(float(v.mean()) for v in diff) / len(ks)
        se = sum(float(v.var()) / len(v) for v in diff) ** 0.5 / len(ks)
        print(f"| {name} | {split} | " + " | ".join(f"{m:.2f}" for m in means)
              + f" | {sum(means) / len(ks):.3f} | {gap:.3f} | {se:.3f} |", flush=True)

    for split in draws:
        row("network", split, base[split])
    for path in paths:
        energy, _, b, info = load_energy(path, device)
        name = f"{Path(path).stem} ({info})"
        saved = torch.load(_local(path), map_location="cpu", weights_only=True)
        if "fit_loss" in saved and torch.equal(b.edges, bins.edges):  # its own bins are those of the table
            own = [float(saved["fit_loss"][k]) for k in ks]
            print(f"| {name} | in-sample | " + " | ".join(f"{v:.2f}" for v in own)
                  + f" | {sum(own) / len(ks):.3f} | – | – |")
        for split, ds in draws.items():
            row(name, split, [loss(energy_predictions(energy, d, batch), d) for d in ds])


def feature_groups(basis) -> list:
    """The group of every feature of the basis, in its layout (as owners)."""
    ace = basis.ace
    out = ["pairs of points"] * (len(ace.pairs) * ace.radial)
    out += ["triples of points"] * (len(ace.triples) * ace.angular * ace.triple_radial ** 2)
    out += ["twist"] * (len(ace.twist_joints) * 2 * ace.twist)
    for l in basis.wigner:
        out += [f"D^{l}(x_j)"] * (19 * (2 * l + 1) ** 2)
    for l in basis.pairs:
        out += [f"D^{l}(G_j)"] * (19 * (2 * l + 1) ** 2)
    for l in basis.pairs:
        out += [f"D^{l}(G_i^T G_k)"] * (len(WIGNER_PAIRS) * (2 * l + 1) ** 2)
    for l in basis.all_pairs:
        out += [f"D^{l}(G_i^T G_k), other pairs"] * (len(OTHER_PAIRS) * (2 * l + 1) ** 2)
    if basis.products:
        out += ["x_i x_k"] * (81 * len(JOINT_PAIRS))
    out += ["triples"] * (TRIPLE_DIM * len(basis.triple_joints))
    out += ["u u x u of a pair"] * (SQUARE_DIM * len(basis.square_pairs))
    assert len(out) == basis.dim
    return out


def decompose(energy, xt: torch.Tensor, t: torch.Tensor) -> dict:
    """The energy's scaled score at one pose xt [J, 3, 3], t [] split by feature group: {group: [J, 3]}, which sum to
    the whole score. Through the features: score = -G(x) dE/dB, G the tangent Jacobian of the features."""
    dev = next(energy.parameters()).device
    x = xt[None].float().to(dev)
    tt = t.reshape(1).float().to(dev)
    gm = jacobian(energy.basis, x)[0]  # [3J, P]
    with torch.enable_grad():
        f = energy.basis(x).detach().requires_grad_(True)
        (de,) = torch.autograd.grad(energy.energy_of_features(f, tt).sum(), f)
    groups = feature_groups(energy.basis)
    out = {}
    for name in dict.fromkeys(groups):
        idx = torch.tensor([i for i, g in enumerate(groups) if g == name], device=dev)
        out[name] = (-(gm[:, idx] @ de[0, idx])).reshape(19, 3).cpu().double()
    return out


def energy_per_joint(energy, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """E~_t of every joint [J] at one pose."""
    dev = next(energy.parameters()).device
    with torch.no_grad():
        f = energy.basis(xt[None].float().to(dev))
        return energy.energy_of_features(f, t.reshape(1).float().to(dev), per_joint=True)[0].cpu().double()


def points(x: torch.Tensor) -> torch.Tensor:
    """The 25 points [25, 3] of a pose [J, 3, 3] (forward kinematics of hp.models.ace_features)."""
    return AceFeatures().positions(x.double()[None])[0]


PARENTS = AceFeatures().parents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=low_noise_table.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fits", nargs="+", help="energy_*.pt of fit.py or embedding.py")
    parser.add_argument("--s-max", type=float, default=0.04)
    parser.add_argument("--poses", type=int, default=5000, help="per bin and split")
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    low_noise_table(args.fits, args.device, args.s_max, args.poses, args.batch)
