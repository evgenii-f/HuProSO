"""Run models through the test protocols: one command, the protocol first, the same flags for every protocol
(hp.evaluation.protocol_cli). For now the protocol is pose completion (PoseD-Flow's Tab. 2, hp.evaluation.completion);
generation and runtime are to follow.

    python scripts/evaluate.py completion --run ~/data/experiments/full/rsgm_posedflow_small_s10/run_0001 --split all
    python scripts/evaluate.py completion --run "my RSGM=~/data/experiments/x_s10/run_0001" --method inpaint \\
        --set steps=200 schedule=edm guidance=0.06 resample=5 replace=false --split valid --together
    python scripts/evaluate.py completion --run ~/data/posedflow_ckpt/checkpoints/epoch-009_step-0_loss-0.0000.ckpt \\
        --settings '{"lr": 0.05, "steps": 200}' --split test --occlusions left_leg
    python scripts/evaluate.py completion --models "PoseD-Flow: their checkpoint" --split test --dry-run
    python scripts/evaluate.py completion --run ~/data/experiments/x_s10/run_0001 --quick --together
    python scripts/evaluate.py completion --run ~/data/experiments/x_s10/run_0001 --quick \\
        --like "RSGM 512x4: their c10, inpaint t52"     # the method and settings of a row (or a point), on x
    python scripts/evaluate.py completion --points config/completion_test_points.yaml --fronts "RSGM dflow" \\
        --split valid --together     # the validation numbers again, next to the ones recorded with the points

Models: --run (a folder or a .ckpt, LABEL=PATH to name it), --models (rows of MODELS in evaluate_completion.py; all:
every row), --points (points of the fronts, scripts/select_test_points.py). --method and the settings (--settings
JSON, then --set key=value) apply to --run: for D-Flow lr, running_weight and steps go to D-Flow and the rest (e.g.
ode_steps, sampler, schedule, guess_weight) to its prior; for inpaint and dposer all go to the method. --like LABEL
starts from the method and settings of a row of MODELS or of a point of config/completion_test_points.yaml (the
settings chosen on the validation for that model; a new model gets them as they are), --settings and --set on top.
Without --like the settings are the defaults of the method: D-Flow those of their protocol, inpaint and dposer those
of the functions (not tuned).

Poses: --split valid is the fixed draw of 1000 validation poses on which the settings are chosen (the Optuna studies of
scripts/tune_completion.py), test their 5000 test poses, all both; --poses n takes the first n of them, as a quick
check in a folder of its own. --quick is --split valid --poses 200 (QUICK): the first 200 of the validation draw, a
random subset fixed by its seed. On random subsets of the test caches (71 models), 200 poses put the paired difference
of two models' mean MPVPE within about 1.4 mm (one sd; 100 poses: 1.9, 1000: 0.6) and swap 0.3% of the pairs that
differ by 2 mm or more: enough to see a change of a few mm, not to choose between close settings. Results: completions and metrics per (model, occlusion) in the folder of the split
(test: --out itself, where pose_completion.ipynb reads them; valid: <out>/valid), and one row per (model,
occlusion) in results.jsonl there; a table at the end. A result made with other settings is kept and reported
unless --recompute; a finished one is skipped unless --force (which computes everything again).

The studies ran with --together and their batch (500 unless the study says otherwise), so the same settings give the
same validation numbers only that way; without --together the hypotheses are drawn as theirs (other random draws,
the same method).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # the code of this checkout, before an editable install of another one
sys.path.insert(0, str(ROOT / "scripts"))

from hp.evaluation import protocol_cli as cli  # noqa: E402
from hp.evaluation.completion import BATCH, OCCLUSIONS  # noqa: E402

QUICK = 200  # poses of --quick
POINTS = ROOT / "config" / "completion_test_points.yaml"  # the points of the fronts, for --like


def completion_parser(sub) -> None:
    from evaluate_completion import CACHE

    p = sub.add_parser("completion", help="pose completion, PoseD-Flow's Tab. 2", description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_model_arguments(p)
    cli.add_common_arguments(p, out=CACHE, quick=QUICK)
    g = p.add_argument_group("completion")
    g.add_argument("--occlusions", nargs="+", default=list(OCCLUSIONS), choices=list(OCCLUSIONS))
    g.add_argument("--k", type=int, default=10, help="hypotheses per pose")
    g.add_argument("--valid-poses", type=int, default=1000, help="the size of the validation draw (the studies: 1000)")
    g.add_argument("--batch", type=int, default=BATCH, help=f"poses per batch (theirs: {BATCH}); speed only")
    g.add_argument("--together", action="store_true", help="the hypotheses of a batch in one GPU batch; speed only")
    g.add_argument("--no-metrics", action="store_true", help="only the completions (no SMPL-X body on this machine)")
    p.set_defaults(protocol=completion)


def poses_of(split: str, args):
    from hp.evaluation.completion import test_poses, valid_poses

    if split == "test":
        return test_poses(args.rotations, args.poses)
    return valid_poses(args.rotations, args.poses if args.poses is not None else args.valid_poses)


def merge_results(path: Path, rows: list) -> None:
    """results.jsonl with these rows, replacing earlier rows of the same (label, occlusion)."""
    keep = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                keep[row["label"], row["occlusion"]] = row
    for row in rows:
        keep[row["label"], row["occlusion"]] = row
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in keep.values()))
    temporary.replace(path)


def table(rows: list, occlusions: list, recorded: dict) -> str:
    """Mean MPVPE per occlusion, and the means over the occlusions of mean, best of k and APD (with the validation
    numbers recorded with a point, if any)."""
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], {})[row["occlusion"]] = row
    head = ["model", *occlusions, "mean", "best of k", "APD"]
    lines = ["| " + " | ".join(head + (["recorded (val)"] if recorded else [])) + " |",
             "|" + "---|" * (len(head) + bool(recorded))]
    for label, per in by_label.items():
        cells = [label] + [f"{per[o]['mpvpe_mean']:.2f}" if o in per else "-" for o in occlusions]
        if all(o in per for o in occlusions):
            mean = lambda key: sum(per[o][key] for o in occlusions) / len(occlusions)  # noqa: E731
            cells += [f"{mean('mpvpe_mean'):.2f}", f"{mean('mpvpe_min'):.2f}", f"{mean('apd'):.2f}"]
        else:
            cells += ["-", "-", "-"]
        if recorded:
            r = recorded.get(label)
            cells.append(f"{r['mean']:.2f} / {r['best_of_k']:.2f} / {r['apd']:.2f}" if r else "")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def completion(args) -> int:
    from evaluate_completion import MODELS, ROTATIONS
    from hp.evaluation.completion import OtherSettings, cache_state, cached_metrics, cached_run, summary

    args.rotations = args.rotations or ROTATIONS
    specs = cli.models(args, MODELS, known=cli.point_specs(POINTS))
    speed = {"batch": args.batch, "together": True} if args.together else (
        {"batch": args.batch} if args.batch != BATCH else {})
    recorded = cli.recorded(args.points)
    body = None
    for split in cli.splits(args.split):
        gt = poses_of(split, args)
        cache = cli.split_dir(args.out, split, args.poses)
        print(f"{split}: {len(gt)} poses, {len(specs)} model(s) x {len(args.occlusions)} occlusion(s) -> {cache}",
              flush=True)
        if args.dry_run:
            for label, spec in specs.items():
                states = {o: cache_state(cache, label, spec, len(gt), o, args.k, args.rotations, split, **speed)
                          for o in args.occlusions}
                print(f"  {label}: " + ", ".join(f"{o} {s}" for o, s in states.items()))
            continue
        rows, kept = [], []
        for label, spec in specs.items():
            for occlusion in args.occlusions:
                start = time.perf_counter()
                try:
                    pred = cached_run(cache, label, spec, gt, occlusion, args.device, args.k, args.rotations, split,
                                      overwrite=args.recompute or args.force, force=args.force, **speed)
                except OtherSettings:
                    kept.append(f"{label} / {occlusion}")
                    continue
                print(f"{label} / {occlusion}: {(time.perf_counter() - start) / 60:.1f} min", flush=True)
                if args.no_metrics:
                    continue
                if body is None:
                    from hp.evaluation.completion import Body
                    body = Body()
                plain, _ = cached_metrics(cache, label, occlusion, pred, gt, body)
                rows.append(dict(label=label, split=split, poses=len(gt), occlusion=occlusion, k=args.k,
                                 **summary(plain), spec=spec))
        if rows:
            merge_results(cache / "results.jsonl", rows)
            print(f"\n{split} ({len(gt)} poses; MPVPE mean per occlusion and mean / best of {args.k} in mm, "
                  f"APD in cm)\n" + table(rows, args.occlusions, recorded if split == "valid" else {}) + "\n")
        if kept:
            print(f"kept, made with other settings (--recompute replaces them): {kept}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="name", required=True, metavar="protocol")
    completion_parser(sub)
    args = parser.parse_args(argv)
    try:
        cli.resolve_poses(args)
        return args.protocol(args)
    except cli.UsageError as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
