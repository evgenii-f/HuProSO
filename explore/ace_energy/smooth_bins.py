"""Smooth the weights of a linear fit over its noise bins (a check, not a better model): w of every bin below --s-max is
replaced by the mean of it and its neighbours, with weights 1/4, 1/2, 1/4, --times times. From the root of the
repository:

    python explore/ace_energy/smooth_bins.py ~/data/experiments/ace/energy_b_rich_products_jd_r0.001.pt --times 3 \\
        --out ~/data/experiments/ace/energy_b_rich_products_jd_smooth3_r0.001.pt
"""
import argparse
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("fit")
parser.add_argument("--times", type=int, default=1)
parser.add_argument("--s-max", type=float, default=0.18, help="bins whose upper edge is below this s are smoothed")
parser.add_argument("--out", required=True)
args = parser.parse_args()
saved = torch.load(Path(args.fit).expanduser(), map_location="cpu", weights_only=True)
k = int((saved["edges"][1:].exp() <= args.s_max).sum())
w = saved["w"]
for _ in range(args.times):
    padded = torch.cat([w[:1], w, w[-1:]])
    smooth = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    w = torch.cat([smooth[:k], w[k:]])
out = {key: value for key, value in saved.items() if key != "fit_loss"}  # the in-sample loss is that of the fit
out["w"] = w
torch.save(out, Path(args.out).expanduser())
print(f"{k} of {len(w)} bins smoothed, saved to {args.out}")
