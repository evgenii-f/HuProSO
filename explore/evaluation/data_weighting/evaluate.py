"""Evaluation of the data-weighting runs on validation: inpainting and log-likelihood, by how far a pose is from the data.

Runs: the uniform baseline fact_RDW_s10 of the factorial and rdw_soft / rdw_strong (sampling weights of rarity.py).
On the 1000 validation poses of the completion studies (valid_poses, seed 0):
- inpainting at the fixed point of the factorial (the rule point of the abl_norm study: 200 EDM steps, guidance
  0.0952, resample 5, no replacement), K = 10, four occlusions: MPVPE per pose and hypothesis, APD;
- log-likelihood per pose (EMA weights).
Poses are split into quintiles by the distance of their hidden part (inpainting) or of the whole pose (likelihood) to
the nearest c10 training pose, from the coverage check (explore/evaluation/data_coverage/coverage.py on dev, its
result ~/data/experiments/data_coverage/coverage.pt); the quintiles depend on no model.

Rule, fixed before training: a weighted run is better than the baseline if its inpainting mean on the farthest quintile
(averaged over the four occlusions) is lower by 2 mm or more and its inpainting mean on all poses is not higher by
more than 0.5 mm.

Every (run, part) is saved to ~/data/experiments/data_weighting/eval/ when done and skipped on a restart; Ctrl+C
stops after the current part, the same command goes on.

    python explore/evaluation/data_weighting/evaluate.py --device mps
    python explore/evaluation/data_weighting/evaluate.py --report-only
"""
import argparse
import signal
import time
from pathlib import Path

import numpy as np
import torch

from hp.evaluation.completion import KEEP, OCCLUSIONS, Body, build, metrics, run, valid_poses

E = Path("~/data/experiments").expanduser()
RUNS = {
    "uniform": E / "factorial/rsgm_posedflow_fact_RDW_s10/run_0001",
    "soft": E / "data_weighting/rsgm_posedflow_rdw_soft_s10/run_0001",
    "strong": E / "data_weighting/rsgm_posedflow_rdw_strong_s10/run_0001",
}
OUT = E / "data_weighting/eval"
COVERAGE = E / "data_coverage/coverage.pt"
INPAINT = dict(steps=200, schedule="edm", guidance=0.09515197497012039, resample=5, replace=False)
PARTS = (*OCCLUSIONS, "loglik")
QUINTILES = 5

stop = False


def on_interrupt(signum, frame):
    global stop
    if stop:
        raise KeyboardInterrupt
    stop = True
    print("\nCtrl+C: stopping after the current part (again: at once)", flush=True)


def evaluate(name: str, part: str, gt: torch.Tensor, device: str, body: Body) -> dict:
    model = build(dict(run=str(RUNS[name])), device)
    if part == "loglik":
        torch.manual_seed(10)
        x = gt[:, KEEP]
        with torch.no_grad():
            ll = torch.cat([model.log_prob(x[i:i + 100].to(model.device)).cpu() for i in range(0, len(x), 100)])
        return {"loglik": ll.numpy()}
    torch.manual_seed(10)
    pred = run(model, gt, part, 10, batch=500, together=True, progress=False, method="inpaint", prior=INPAINT)
    m = metrics(pred, gt, part, body)
    return {"mpvpe": m["mpvpe"], "apd": m["apd"]}


def compute(device: str):
    signal.signal(signal.SIGINT, on_interrupt)
    OUT.mkdir(parents=True, exist_ok=True)
    gt, body = valid_poses(n=1000, seed=0), Body()
    todo = [(n, p) for n in RUNS for p in PARTS if not (OUT / f"{n}_{p}.pt").exists()]
    missing = [n for n in RUNS if not (RUNS[n] / "weights_ema.pt").exists()]
    todo = [(n, p) for n, p in todo if n not in missing]
    if missing:
        print(f"not trained yet, skipped: {missing}", flush=True)
    for k, (name, part) in enumerate(todo, 1):
        t0 = time.time()
        result = evaluate(name, part, gt, device, body)
        torch.save(result, OUT / f"{name}_{part}.pt")
        print(f"[{k}/{len(todo)}] {name} {part}: {time.time() - t0:.0f} s", flush=True)
        if stop:
            print("stopped; the same command goes on", flush=True)
            return


def quintile_of(distance: np.ndarray) -> np.ndarray:
    edges = np.quantile(distance, np.linspace(0, 1, QUINTILES + 1)[1:-1])
    return np.searchsorted(edges, distance, side="right")


def report():
    cov = torch.load(COVERAGE, weights_only=False)["angle"]
    names = [n for n in RUNS if all((OUT / f"{n}_{p}.pt").exists() for p in PARTS)]
    res = {n: {p: torch.load(OUT / f"{n}_{p}.pt", weights_only=False) for p in PARTS} for n in names}
    print("Inpainting, 1000 validation poses, K = 10, mean over the four occlusions (mm, cm)")
    head = " | ".join(f"Q{q + 1}" for q in range(QUINTILES))
    print(f"| run | mean | best of 10 | APD | mean by quintile of distance to the data: {head} |")
    print("|---|---|---|---|" + "---|" * QUINTILES)
    summary = {}
    for n in names:
        mean = np.mean([res[n][o]["mpvpe"].mean() for o in OCCLUSIONS])
        best = np.mean([res[n][o]["mpvpe"].min(1).mean() for o in OCCLUSIONS])
        apd = np.mean([res[n][o]["apd"].mean() for o in OCCLUSIONS])
        by_q = []
        for q in range(QUINTILES):
            by_q.append(np.mean([res[n][o]["mpvpe"].mean(1)[quintile_of(cov[("valid", o, "c10")].numpy()) == q].mean()
                                 for o in OCCLUSIONS]))
        summary[n] = (mean, by_q[-1])
        print(f"| {n} | {mean:.2f} | {best:.2f} | {apd:.2f} | " + " | ".join(f"{v:.2f}" for v in by_q) + " |")
    print("\nPer occlusion: mean / best of 10 / APD")
    print("| run | " + " | ".join(OCCLUSIONS) + " |")
    print("|---|" + "---|" * len(OCCLUSIONS))
    for n in names:
        cells = [f"{res[n][o]['mpvpe'].mean():.1f} / {res[n][o]['mpvpe'].min(1).mean():.1f} / {res[n][o]['apd'].mean():.2f}"
                 for o in OCCLUSIONS]
        print(f"| {n} | " + " | ".join(cells) + " |")
    print("\nLog-likelihood (nats), mean, and by quintile of the distance of the whole pose to the data")
    print(f"| run | mean | {head} |")
    print("|---|---|" + "---|" * QUINTILES)
    qp = quintile_of(cov[("valid", "pose", "c10")].numpy())
    for n in names:
        ll = res[n]["loglik"]["loglik"]
        print(f"| {n} | {ll.mean():.2f} | " + " | ".join(f"{ll[qp == q].mean():.2f}" for q in range(QUINTILES)) + " |")
    if "uniform" in summary:
        base_mean, base_far = summary["uniform"]
        print("\nRule (far quintile -2 mm or more, all poses at most +0.5 mm):")
        for n in names:
            if n != "uniform":
                d_far, d_all = summary[n][1] - base_far, summary[n][0] - base_mean
                print(f"  {n}: far quintile {d_far:+.2f} mm, all {d_all:+.2f} mm -> "
                      f"{'passes' if d_far <= -2 and d_all <= 0.5 else 'fails'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only:
        compute(args.device)
    report()


if __name__ == "__main__":
    main()
