"""Does D-Flow depend on the scale of its loss? A larger batch of --together halves the gradient of the mean loss of
every hypothesis, and Adam is invariant to that scale only while its eps is negligible. The same validation poses and
seeds, PoseD-Flow, one occlusion, the loss (and the running weight) multiplied by each --scales value; the control
1.0001 shows the spread of float noise alone over the optimizer steps. Run from the root:

    PYTHONPATH=src python explore/evaluation/completion_diagnostics/dflow_loss_scale.py --device mps

Prints, per scale, the mean angle (rad) of the hidden joints to the ground truth and to the run at scale 1.
"""
import argparse

import torch

import hp.models.dflow as dflow
from hp.evaluation.completion import OCCLUSIONS, build, run, valid_poses

CKPT = "~/data/posedflow_ckpt/checkpoints/epoch-009_step-0_loss-0.0000.ckpt"


def angle(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    trace = (x.transpose(-1, -2) @ y).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((trace - 1) / 2).clamp(-1, 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--poses", type=int, default=100)
    parser.add_argument("--occlusion", default="left_leg", choices=list(OCCLUSIONS))
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.5, 1.0001])
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    torch.manual_seed(0)
    gt = valid_poses("~/data/AMASS_rotations_posedflow_c10", 1000)[:args.poses]
    model = build({"posedflow": CKPT}, args.device)
    hidden, original = list(OCCLUSIONS[args.occlusion]), dflow.geodesic_loss
    out = {}
    try:
        for scale in [1.0] + [s for s in args.scales if s != 1.0]:
            dflow.geodesic_loss = lambda p, t, s=scale: s * original(p, t)
            out[scale] = run(model, gt, args.occlusion, 10, steps=args.steps, lr=args.lr, running_weight=1e-3 * scale,
                             batch=args.poses, together=True, progress=False)[:, :, hidden]
    finally:
        dflow.geodesic_loss = original
    for scale, pred in out.items():
        print(f"scale {scale}: to gt {angle(pred, gt[:, None, hidden]).mean():.5f} rad, "
              f"to scale 1 {angle(pred, out[1.0]).mean():.5f} rad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
