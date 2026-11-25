# Weighted sampling of the training poses

`train.sample_weights` draws the training poses in proportion to given weights instead of uniformly. Off by default.
Results of the study of weights by rarity: [2026-11-22_data_weighting.md](experiments/2026-11-22_data_weighting.md).

## When to use it

- **Not by default.** Weights by rarity are a trade: the stronger they are, the better best of 10 of inpainting and SDS
  and the worse generation (Wasserstein, precision). At an unchanged generation quality there is no gain. The uniform
  model (β = 0) is the one we use.
- **D-Flow may profit** (test: band mean 91.53 → 87.76 mm at β 3.3, one model pair), at the price of generation and of a
  separate model. This is left open.
- Weak weights (β 0.6-0.9) gave a better FID to the validation poses and log-likelihood, but worse Wasserstein and
  precision.
- For a new question, choose the weights on validation, and compare with the uniform model trained with the same seed.

## How to use it

The weights are a file with a tensor `[N]` of non-negative numbers, one per training pose in the order of the dataset
(`PoseDFlow_train.pt` of the rotation folder), read with `torch.load(..., weights_only=True)`. It needs
`train.data_on_device=true`. Every pass draws N poses with replacement, pose i with probability w_i / sum w. The loss is
not reweighted, so the model learns the weighted distribution.

Rarity weights for c10 (the rarity takes about 5 minutes on the Mac; `rarity.pt` is kept in
`~/data/experiments/data_weighting`):

```bash
PYTHONPATH=src python explore/evaluation/data_weighting/rarity.py --device mps
```

Any other weights, for example β 0.7 with floor 0.1 and cap 10:

```bash
python - <<'EOF'
import torch
from pathlib import Path
d = Path("~/data/experiments/data_weighting").expanduser()
r = torch.load(d / "rarity.pt", weights_only=True)
w = (r / r.median()).pow(0.7).clamp(0.1, 10)
torch.save(w, d / "weights_b0.7.pt")
print("ESS / N:", float(w.double().sum() ** 2 / (w.double() ** 2).sum() / len(w)))
EOF
```

Training (`rdw` is the network of the study without weights, `rdw_b3.3` etc. are its weighted versions):

```bash
python scripts/train_rsgm_posedflow.py --config rdw --run-root ~/data/experiments/data_weighting --device cuda:0 train.sample_weights=~/data/experiments/data_weighting/weights_b0.7.pt
```

Use a `--run-root` of its own for every set of weights: the run is named by the config and the seed only, and a finished
run of that name is skipped.

`scripts/tune_weighting.py` searches β, w_min and w_max with Optuna against the generation harness and completion
(`config/weighting_tuning/*.yaml`); a study is announced before it starts (CLAUDE.md).

## Precision

The cumulative weights and the random numbers are float64 (from 2026-11-25). The runs of the study used float32: for c10
the mass per decile of rarity is right within 0.1%, but at β 3.3 14% of the poses (0.7% of the mass, the lightest ones)
were never drawn; for the full split (18.6M poses) 10% of the poses would never be drawn even with equal weights. The
numbers are computed from the weights and the spacing of float32, see `docs/experiments/2026-11-22_data_weighting.md`.
