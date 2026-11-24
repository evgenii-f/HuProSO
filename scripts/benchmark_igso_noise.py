"""Speed of drawing the IGSO(3) noise of RSGM on the device against the CPU (experimental/igso_gpu), e.g.

    python scripts/benchmark_igso_noise.py --device cuda:0

For both ways (RSGM.noise_on_cpu False and True) it times a training step (loss, backward, Adam), a network call of
inpainting (hp.models.inpainting.rsgm_inpaint, with and without guidance) and a step of SDS, DPoser's optimization
(hp.models.score_completion.rsgm_complete), with the same untrained network and
uniform random poses: the timings do not depend on the data, and no dataset is needed. On mps the loss always draws
on the CPU (no float64 there), so only inpainting differs.
"""
import argparse
import time

import roma
import torch

from hp.models import build_model
from hp.models.inpainting import rsgm_inpaint
from hp.models.score_completion import rsgm_complete


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn, device: str, repeats: int, warmup: int = 2) -> float:
    """Mean seconds of fn() over `repeats` runs, after `warmup` runs."""
    for _ in range(warmup):
        fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync(device)
    return (time.perf_counter() - start) / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--joints", type=int, default=19)
    parser.add_argument("--hidden", nargs="+", type=int, default=[512] * 4)
    parser.add_argument("--train-batches", nargs="+", type=int, default=[4096, 16384])
    parser.add_argument("--train-repeats", type=int, default=100)
    parser.add_argument("--inpaint-poses", nargs="+", type=int, default=[5000, 20000], help="K x poses of a call")
    parser.add_argument("--inpaint-steps", type=int, default=20)
    parser.add_argument("--sds-steps", type=int, default=20, help="steps of a timed SDS call")
    args = parser.parse_args()
    device, joints = args.device, args.joints
    torch.manual_seed(0)
    model = build_model("rsgm", num_joints=joints, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    name = torch.cuda.get_device_name(device) if device.startswith("cuda") else device
    print(f"torch {torch.__version__}, {name}, {torch.get_num_threads()} CPU threads, MLP {args.hidden}, {joints} joints")
    print("| measurement | noise on CPU | noise on device | speed-up |")
    print("|---|---|---|---|")

    def row(what, unit, fn, repeats, warmup=2):
        sec = {}
        for on_cpu in (True, False):
            model.noise_on_cpu = on_cpu
            sec[on_cpu] = timed(fn, device, repeats, warmup)
        print(f"| {what} | {unit(sec[True])} | {unit(sec[False])} | {sec[True] / sec[False]:.2f}x |", flush=True)

    model.train()
    for batch in args.train_batches:
        pose = roma.random_rotmat(batch * joints).reshape(batch, joints, 3, 3).to(device)

        def step():
            optimizer.zero_grad(set_to_none=True)
            model.loss(pose).backward()
            optimizer.step()

        row(f"train step, batch {batch}", lambda s: f"{1 / s:.1f} it/s", step, args.train_repeats, warmup=10)

    model.eval()
    visible = torch.ones(joints, dtype=torch.bool)
    visible[: joints // 4] = False
    for n in args.inpaint_poses:
        observed = roma.random_rotmat(n * joints).reshape(n, joints, 3, 3)
        for guidance, resample in ((0.0, 3), (0.1, 1)):
            calls = args.inpaint_steps * resample
            row(f"inpainting, {n} poses, guidance {guidance}, resample {resample}",
                lambda s, calls=calls: f"{1000 * s / calls:.2f} ms/call",
                lambda: rsgm_inpaint(model, observed, visible, steps=args.inpaint_steps, resample=resample,
                                     guidance=guidance, replace=True, schedule="edm"), 3, warmup=1)
    for n in args.inpaint_poses:
        observed = roma.random_rotmat(n * joints).reshape(n, joints, 3, 3)
        init = roma.random_rotmat(n * joints).reshape(n, joints, 3, 3)
        row(f"SDS (DPoser's method), {n} poses", lambda s: f"{1000 * s / args.sds_steps:.2f} ms/step",
            lambda: rsgm_complete(model, observed, visible, init, iterations=1, steps_per_iter=args.sds_steps), 3,
            warmup=1)


if __name__ == "__main__":
    main()
