"""Run models through the test protocols: one command, the protocol first, the same flags for every protocol
(hp.evaluation.protocol_cli). The protocols are pose completion (PoseD-Flow's Tab. 2, hp.evaluation.completion),
motion denoising (their Tab. 3, hp.evaluation.denoising) and unconditional generation (hp.evaluation.generation; both
at the end of this text); runtime is to follow.

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

Motion denoising (sequences of 60 frames, noise of 40 and 100 mm on the joints, MPJPE and MPVPE in mm):

    python scripts/evaluate.py denoising --points config/denoising_test_points.yaml --split test --device cuda:0
    python scripts/evaluate.py denoising --points config/denoising_test_points.yaml --groups "RSGM sds" --dry-run
    python scripts/evaluate.py denoising --models "DPoser: their full, DPoser method" --sigmas 0.1 --split valid
    python scripts/evaluate.py denoising --run ~/data/experiments/x_s10/run_0001 --like ours --quick
    python scripts/evaluate.py denoising --run ~/data/experiments/x_s10/run_0001 --set t_max=0.05 t_min=0.02 \\
        sds_weight=8 lr=0.04 steps_per_iter=67 --quick

--models are the rows of MODELS in evaluate_denoising.py, --points the rows of the test with their sources and
validation numbers (scripts/denoising_test_points.py; --groups is --fronts there). For --run the method is dposer
(DPoser's optimization, score distillation for RSGM) unless --method dflow, which a PoseD-Flow .ckpt gets by default;
the settings apply at every noise level: for dposer iterations, steps_per_iter, t_max, t_min, lr, sds_weight; for
dflow steps, lr, temp_weight, running_weight, guess_weight, optimizer, the rest (ode_steps, sampler, schedule) to the
prior. --like LABEL starts from the method and the settings per noise level of a row of MODELS or a point, or of a
ready name: ours (RSGM + SDS at the settings tuned on validation), "rsgm budget" (the same at the tuned DPoser's
steps), dposer (DPoser's own settings), "dposer tuned", posedflow (D-Flow with PoseD-Flow's settings); --settings and
--set apply on top at every noise level. Without --like a dposer run has DPoser's own settings, which are not tuned
for RSGM (a new RSGM wants --like ours). Both splits are in --out itself (<label>/<split>_<mm>mm.pt, where motion_denoising.ipynb reads them); --poses
n (the first n sequences) and --quick (the first 30 validation chunks) go to folders of their own. A stopped run
(Ctrl+C) continues from its last batch when it is run again.

Generation (samples without a condition; no --poses, the number of samples is part of the metrics):

    python scripts/evaluate.py generation --run ~/data/experiments/x_s10/run_0001 --like heun --quick
    python scripts/evaluate.py generation --run ~/data/experiments/x_s10/run_0001 --like heun --split valid
    python scripts/evaluate.py generation --models "RSGM 512x4: their full" --split test --dry-run

On the validation FID_val (the statistics of the validation poses), FID own, APD, dNN, our harness (Wasserstein,
recall, precision) and, without --quick, the log-likelihood; results in <out>/valid. On the test the stages and the
cache of hp.evaluation.pareto: their FID, FID own, APD, dNN and the harness with the log-likelihood. The settings
(--set) are those of the sampler: sample_steps, sampler, schedule, schedule_rho, eps, likelihood_steps. --like LABEL
takes those of a row of hp.evaluation.pareto.MODELS, or a ready name: own, heun, euler. Without --like the sampler is
the model's own. A stopped run (Ctrl+C) continues from its last finished stage when it is run again.

Everything at once, for a new RSGM, next to the references (config/references/posedflow.yaml, hp.evaluation.references):

    python scripts/evaluate.py all --run "my RSGM=~/data/experiments/x_s10" --like --quick    # then without --quick
    python scripts/evaluate.py all --run ~/data/experiments/x_s10 --like rsgm denoising="rsgm budget" --methods inpaint
    python scripts/evaluate.py all --run ~/data/experiments/x_s10 --tune grid --methods inpaint
    python scripts/evaluate.py all --run ~/data/experiments/x_s10 --tune inpaint=grid dflow=ready --dry-run
    python scripts/evaluate.py references --device cuda:0 --dry-run

all runs the three protocols on the validation (the test only with --split test). The completion has no default, it is
said on the command line (hp.evaluation.tuning): --like is ONE ready setting per procedure (--methods), that of the
reference of the same architecture, read from model.json (another architecture: that of timeres 1024x2, said so);
--tune grid is 10 settings of the inpainting, --tune study an Optuna study (120 or 60). A grid or study is announced
with its number of settings and asks before it starts (--yes: no question). Generation is heun, denoising ours. Then
a table per protocol: the model, the references with the differences and what of them is noise, the ranks, the
targets; also written to --reports. references computes the references themselves, once.
"""
import argparse
import json
import signal
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


def collect(args, label: str, spec: dict, result, complete: bool = True) -> None:
    """The results of a model for the caller of a protocol (all, references), if it gave args.collect."""
    if complete and getattr(args, "collect", None) is not None:
        args.collect[label] = (spec, result)


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
    soft = cli.SoftStop("occlusion")
    if not args.dry_run:
        signal.signal(signal.SIGINT, soft)
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
            parts = {}
            for occlusion in args.occlusions:
                if soft.requested:
                    break
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
                parts[occlusion] = plain
                rows.append(dict(label=label, split=split, poses=len(gt), occlusion=occlusion, k=args.k,
                                 **summary(plain), spec=spec))
            collect(args, label, spec, parts, len(parts) == len(OCCLUSIONS))
        if rows:
            merge_results(cache / "results.jsonl", rows)
            print(f"\n{split} ({len(gt)} poses; MPVPE mean per occlusion and mean / best of {args.k} in mm, "
                  f"APD in cm)\n" + table(rows, args.occlusions, recorded if split == "valid" else {}) + "\n")
        if kept:
            print(f"kept, made with other settings (--recompute replaces them): {kept}")
        if soft.requested:
            print("stopped; run the same command again to continue")
            return 1
    return 0


DENOISING_QUICK = 30  # validation chunks of --quick
DFLOW_DENOISING_KEYS = ("steps", "lr", "temp_weight", "running_weight", "guess_weight", "optimizer")


DENOISING_POINTS = ROOT / "config" / "denoising_test_points.yaml"  # the rows of the test, for --like
# ready names for --like: the rows of the report's comparison (any label of MODELS or of the points works too)
DENOISING_LIKE = {
    "ours": "RSGM 1536x6: their c10, DPoser method",               # RSGM + SDS, settings tuned on validation
    "rsgm budget": "RSGM 1536x6: their c10, DPoser method, DPoser's tuned budget",  # the same at the tuned DPoser's steps
    "dposer": "DPoser: their full, DPoser method",                  # DPoser's own settings
    "dposer tuned": "DPoser: their full, DPoser method, tuned",     # DPoser tuned like RSGM
    "posedflow": "PoseD-Flow: their checkpoint",                    # D-Flow with PoseD-Flow's settings
}


def denoising_settings_of(spec: dict, default_method: str = "dposer"):
    """(method, settings) of a row of evaluate_denoising.MODELS, for --like: its settings (per noise level if it has
    them per noise level) and the options of its prior in one dict; the rows without a method are D-Flow."""
    return spec.get("method", "dflow"), {**spec.get("prior", {}), **spec.get("settings", {})}


def denoising_spec(path: str, weights=None, method=None, settings=None, default_method: str = "dposer") -> dict:
    """The spec of evaluate_denoising.MODELS for a --run: the settings of the method under "settings", for D-Flow the
    options of the prior (ode_steps, sampler, ...) under "prior". Settings per noise level ({0.04: {...}, 0.1: {...}},
    from --like) keep their levels, and the other keys (--set, --settings) apply on top at every level."""
    spec = {"posedflow": path} if str(path).endswith(".ckpt") else {"run": path}
    if weights and "run" in spec:
        spec["weights"] = weights
    method = method or ("dflow" if "posedflow" in spec else default_method)
    settings = dict(settings or {})
    levels = {key: settings.pop(key) for key in list(settings) if isinstance(key, float)}
    if method == "dposer":
        spec["method"] = "dposer"
    elif method != "dflow":
        raise cli.UsageError(f"denoising has the methods dposer and dflow, not {method!r}")
    else:
        prior = {key: settings.pop(key) for key in list(settings) if key not in DFLOW_DENOISING_KEYS}
        if prior:
            spec["prior"] = prior
    if levels:
        spec["settings"] = {sigma: {**values, **settings} for sigma, values in levels.items()}
    elif settings:
        spec["settings"] = settings
    return spec


def denoising_parser(sub) -> None:
    from evaluate_denoising import CACHE

    from hp.evaluation.denoising import SIGMAS

    p = sub.add_parser("denoising", help="motion denoising, PoseD-Flow's Tab. 3", description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_model_arguments(p)
    p.add_argument("--groups", nargs="+", default=None, dest="fronts",
                   help='with --points: only these groups of rows, e.g. "RSGM sds" (as --fronts)')
    cli.add_common_arguments(p, out=CACHE, quick=DENOISING_QUICK)
    g = p.add_argument_group("denoising")
    g.add_argument("--sigmas", nargs="+", type=float, default=list(SIGMAS), choices=list(SIGMAS),
                   help="noise levels (m)")
    g.add_argument("--valid-sequences", type=int, default=100, help="the size of the validation draw (the studies: 100)")
    g.add_argument("--sequences-file", default="~/data/experiments/denoising/sequences.pt")
    g.add_argument("--batch", type=int, default=10, help="sequences per batch; speed and memory only")
    p.set_defaults(protocol=denoising)


def denoising_table(rows: list, sigmas: list, recorded: dict) -> str:
    """MPJPE / MPVPE per noise level (and those recorded on validation with a point, if any)."""
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], {})[row["sigma"]] = row
    head = ["model"] + [f"{round(s * 1000)} mm" for s in sigmas]
    lines = ["| " + " | ".join(head + (["recorded (val)"] if recorded else [])) + " |",
             "|" + "---|" * (len(head) + bool(recorded))]
    for label, per in by_label.items():
        cells = [label] + [f"{per[s]['mpjpe']:.2f} / {per[s]['mpvpe']:.2f}" if s in per else "-" for s in sigmas]
        if recorded:
            r = recorded.get(label, {})
            cells.append(" ; ".join(f"{r[s]['mpjpe']:.2f} / {r[s]['mpvpe']:.2f}" for s in sigmas if s in r))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def merge_rows(path: Path, rows: list, keys=("label", "occlusion")) -> None:
    """results.jsonl with these rows, replacing earlier rows of the same keys."""
    keep = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                keep[tuple(row[k] for k in keys)] = row
    for row in rows:
        keep[tuple(row[k] for k in keys)] = row
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in keep.values()))
    temporary.replace(path)


def denoising(args) -> int:
    from evaluate_denoising import MODELS

    from hp.evaluation import denoising as D

    args.like = DENOISING_LIKE.get(args.like, args.like)
    specs = cli.models(args, MODELS, default_method="dposer", known=cli.point_specs(DENOISING_POINTS),
                       spec_of=denoising_spec, settings_of=denoising_settings_of)
    recorded = cli.recorded(args.points)
    data = dict(sequences_file=args.sequences_file, first=args.poses, n_valid=args.valid_sequences)
    soft = cli.SoftStop("batch")  # Ctrl+C once (the run continues from there when run again)
    if not args.dry_run:
        signal.signal(signal.SIGINT, soft)
    for split in cli.splits(args.split):
        # the file names carry the split (<label>/<split>_<mm>mm.pt), so both splits share --out, as the caches of
        # evaluate_denoising.py; a check on the first n sequences goes to a folder of its own
        cache = Path(args.out).expanduser() if args.poses is None else cli.split_dir(args.out, split, args.poses)
        n = len(D.split_sequences(split, args.sequences_file, args.poses, args.valid_sequences)[1])
        print(f"{split}: {n} sequences, {len(specs)} model(s) x {len(args.sigmas)} noise level(s) -> {cache}",
              flush=True)
        if args.dry_run:
            for label, spec in specs.items():
                states = {s: D.cache_state(cache, label, spec, split, s, **data) for s in args.sigmas}
                print(f"  {label}: " + ", ".join(f"{round(s * 1000)} mm {v}" for s, v in states.items()))
            continue
        rows, kept = [], []
        for label, spec in specs.items():
            parts = {}
            for sigma in args.sigmas:
                if soft.requested:
                    break
                start = time.perf_counter()
                try:
                    m = D.cached_run(cache, label, spec, split, sigma, args.device, args.batch,
                                     stop=soft.stop, overwrite=args.recompute or args.force,
                                     force=args.force, **data)
                except D.OtherSettings:
                    kept.append(f"{label} / {round(sigma * 1000)} mm")
                    continue
                if m is None:
                    continue
                print(f"{label} / {round(sigma * 1000)} mm: {(time.perf_counter() - start) / 60:.1f} min", flush=True)
                parts[sigma] = m
                rows.append(dict(label=label, split=split, sequences=n, sigma=sigma, **D.summary(m), spec=spec))
            collect(args, label, spec, parts, len(parts) == len(D.SIGMAS))
        if rows:
            merge_rows(cache / "results.jsonl", rows, keys=("label", "split", "sigma"))
            print(f"\n{split} ({n} sequences; MPJPE / MPVPE in mm)\n"
                  + denoising_table(rows, args.sigmas, recorded if split == "valid" else {}) + "\n")
        if kept:
            print(f"kept, made with other settings (--recompute replaces them): {kept}")
        if soft.requested:
            print("stopped; run the same command again to continue")
            return 1
    return 0


# ready names for --like of the generation (any label of hp.evaluation.pareto.MODELS works too)
GENERATION_LIKE = {
    "own": None,                                                      # the model's own sampler, as it was trained
    "heun": "RSGM 512x4: their c10 (3 seeds), Heun EDM 50 NFE",       # RSGM: Heun on the EDM grid, 25 steps
    "euler": "RSGM 512x4: their c10 (3 seeds), ODE Euler 50",         # RSGM: the ODE, 50 uniform steps
}


def generation_settings_of(spec: dict, default_method: str = "sample"):
    """(method, settings) of a row of hp.evaluation.pareto.MODELS, for --like: the options of its sampler."""
    from hp.evaluation.pareto import OVERRIDES

    if "posedflow" in spec:
        raise cli.UsageError("--like takes the sampler of a --run; a PoseD-Flow row has its own")
    return default_method, {key: spec[key] for key in OVERRIDES if key in spec}


def generation_spec(path: str, weights=None, method=None, settings=None, default_method: str = "sample") -> dict:
    """The spec of a --run: {"run", "weights", "settings": the options of the sampler}."""
    from hp.evaluation.pareto import OVERRIDES

    if str(path).endswith(".ckpt"):
        raise cli.UsageError('a PoseD-Flow checkpoint is a row: --models "PoseD-Flow: their checkpoint, midpoint 100 (paper)"')
    if method not in (None, default_method):
        raise cli.UsageError(f"generation has no methods, not {method!r}; the sampler is set by --set or --like")
    unknown = set(settings or {}) - set(OVERRIDES)
    if unknown:
        raise cli.UsageError(f"generation has the settings {', '.join(OVERRIDES)}, not {sorted(unknown)}")
    spec = {"run": path}
    if weights:
        spec["weights"] = weights
    if settings:
        spec["settings"] = dict(settings)
    return spec


def generation_parser(sub) -> None:
    from hp.evaluation import pareto as P

    p = sub.add_parser("generation", help="unconditional generation, PoseD-Flow's protocol and our harness",
                       description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_model_arguments(p)
    cli.add_common_arguments(p, out=str(P.CACHE), poses=False,
                             quick_help="the validation without the log-likelihood (the other numbers are the same)")
    g = p.add_argument_group("generation")
    g.add_argument("--skeleton", default=P.SKELETON, help="where the SMPL skeleton is on this machine")
    g.add_argument("--gt-stats", default=P.GT_STATS, help="where their FID statistics are on this machine (test)")
    p.set_defaults(protocol=generation)


def generation(args) -> int:
    from hp.evaluation import generation as G
    from hp.evaluation import pareto as P

    if args.points or args.rotations:
        raise cli.UsageError("generation has no --points and its data are fixed (no --rotations)")
    if args.like is not None:
        args.like = GENERATION_LIKE.get(args.like, args.like)  # "own": no settings
    specs = cli.models(args, P.MODELS, default_method="sample", spec_of=generation_spec,
                       settings_of=generation_settings_of)
    zoo = P.Zoo({label: G.zoo_spec(spec) for label, spec in specs.items()}, args.device)
    if not zoo.builders:
        raise cli.UsageError("none of the models is on this machine")
    soft = cli.SoftStop("stage")
    if not args.dry_run:
        signal.signal(signal.SIGINT, soft)
    for split in cli.splits(args.split):
        arena = G.Arena(split, args.device, skeleton=args.skeleton, gt_stats=args.gt_stats)
        cache = cli.split_dir(args.out, split)
        stages = [s for s in G.STAGES[split] if not (args.quick and s == "likelihood")]
        print(f"{split}: {len(arena.poses)} poses, {len(zoo.builders)} run(s) x {len(stages)} stage(s) -> {cache}",
              flush=True)
        if args.dry_run:
            for label, identity in zoo.identities.items():
                states = [f"{s} {G.stage_state(cache, label, s, {'model': identity, **arena.settings(s)})}"
                          for s in stages]
                print(f"  {label}: " + ", ".join(states))
            continue
        rows, kept = [], []
        for label, build in zoo.builders.items():
            results = {}
            for stage in stages:
                if soft.requested:
                    break
                start = time.perf_counter()
                try:
                    results[stage] = G.run_stage(arena, label, stage, build, zoo.identities[label], cache,
                                                 overwrite=args.recompute, force=args.force)
                except G.OtherSettings:
                    kept.append(f"{label} / {stage}")
                    continue
                print(f"{label} / {stage}: {(time.perf_counter() - start) / 60:.1f} min", flush=True)
            if len(results) == len(stages):
                rows.append(dict(label=label, split=split, poses=len(arena.poses), **G.numbers(results),
                                 spec=specs[zoo.group_of[label]]))
                collect(args, label, rows[-1]["spec"], {k: v for k, v in rows[-1].items() if k != "spec"})
        if rows:
            merge_rows(cache / "results.jsonl", rows, keys=("label", "split"))
            real = [] if soft.requested else [dict(label=k, **v) for k, v in arena.real(cache).items()]
            print(f"\n{split} ({len(arena.poses)} poses; FID on joints, APD in cm, dNN in rad, Wasserstein / recall / "
                  f"precision in deg per joint)\n" + G.table(rows + real, split) + "\n")
        if kept:
            print(f"kept, made with other settings (--recompute replaces them): {kept}")
        if soft.requested:
            print("stopped; run the same command again to continue")
            return 1
    return 0


# ---- all protocols at once -----------------------------------------------------------------------------------------
PROFILES = {"generation": "heun", "denoising": "ours"}  # --like rsgm; the completion gets the reference sets


def likes_of(likes) -> dict:
    """{protocol: LABEL} of --like; "rsgm" stands for the reference sets of the completion."""
    from hp.evaluation import references as R

    chosen = {}
    for item in likes:
        protocol, sep, label = item.partition("=")
        if sep and protocol in R.PROTOCOLS:
            chosen[protocol] = label
        else:
            chosen["completion"] = item
    return chosen


def completion_modes(args) -> dict:
    """{method: ready | grid | study}, as said with --like or --tune: there is no default, so that nobody takes one
    setting for a search or the other way round."""
    from hp.evaluation import tuning as T

    ready = args.like is not None and (not args.like or "completion" in likes_of(args.like))
    if ready and args.tune:
        raise cli.UsageError("--like and --tune both set the completion: one of them (--like PROTOCOL=LABEL is free)")
    if ready:
        return {method: "ready" for method in args.methods}
    if not args.tune:
        raise cli.UsageError(T.MENU)
    result = T.modes(args.tune, args.methods)
    if any(m != "ready" for m in result.values()) and (args.quick or args.poses or args.split == "test"):
        raise cli.UsageError("a grid or a study chooses on the full validation draw: without --quick, --poses, --split")
    return result


def profile(likes, config: dict, arch: str, methods) -> list:
    """[(protocol, group, arguments, note)]: the runs of a model of this architecture with ready settings. likes:
    "rsgm" or nothing (the reference sets), a LABEL of the completion, and PROTOCOL=LABEL per protocol."""
    from hp.evaluation import references as R

    chosen = {k: v for k, v in likes_of(likes or []).items() if v != "rsgm"}
    runs = []
    for protocol in R.PROTOCOLS:
        if protocol != "completion":
            like = chosen.get(protocol, PROFILES[protocol])
            runs.append((protocol, "", ["--like", like], like))
        elif protocol in chosen:
            runs.append((protocol, completion_method(chosen[protocol]), ["--like", chosen[protocol]], chosen[protocol]))
        else:
            for method in methods:
                row, own = R.set_for(config, method, arch)
                note = f"{row['key']}, {row['source']}" + ("" if own else f"; no set of {arch}")
                runs.append((protocol, method, ["--method", method, "--settings", json.dumps(row["settings"])], note))
    return runs


def completion_method(label: str) -> str:
    from evaluate_completion import MODELS

    lookup = {**cli.point_specs(POINTS), **MODELS}
    if label not in lookup:
        raise cli.UsageError(f"--like {label!r}: not a row of the registry nor a point")
    return cli.method_of(lookup[label])[0]


def call(args, argv: list, parser) -> tuple:
    """Run a protocol with these arguments as its own command would: (its code, {label: (spec, result)})."""
    sub = parser.parse_args(argv)
    sub.collect = {}
    cli.resolve_poses(sub)
    code = sub.protocol(sub)
    return code, sub.collect


def run_label(base: str, protocol: str, group: str, arguments: list) -> str:
    """A label of its own for every setting of a model."""
    return f"{base}, {group or protocol} {cli.short_hash(arguments)}"


def common_argv(args, protocol: str) -> list:
    argv = ["--split", args.split, "--device", args.device]
    if args.out:
        argv += ["--out", str(Path(args.out).expanduser() / protocol)]
    if protocol == "completion":
        argv += ["--together", "--batch", str(args.batch)]
    if protocol != "generation" and (n := getattr(args, "poses" if protocol == "completion" else "sequences", None)):
        argv += ["--poses", str(n)]
    if getattr(args, "quick", None):
        argv += ["--quick"]
    return argv + [flag for flag in ("--recompute", "--force", "--dry-run")
                   if getattr(args, flag[2:].replace("-", "_"))]


def entry_of(protocol: str, group: str, result, **fields) -> dict:
    from hp.evaluation import references as R

    if protocol == "completion":
        return R.completion_entry(result, group=group, **fields)
    if protocol == "denoising":
        return R.denoising_entry(result, **fields)
    numbers = {k: v for k, v in result.items() if isinstance(v, (int, float)) and k != "poses"}
    return R.generation_entry(numbers, result["poses"], **fields)


def tune_plans(args, config: dict, path: str, arch: str, how: dict) -> dict:
    """{method: Plan} of the grids and studies, announced before anything runs."""
    from hp.evaluation import references as R
    from hp.evaluation import tuning as T

    plans = {}
    for method, mode in how.items():
        if mode != "ready":
            row, own = R.set_for(config, method, arch)
            plans[method] = T.make_plan(config, method, mode, path, row, own, args.n_trials,
                                        args.batch if args.batch != BATCH else None)
            print("\n" + T.announce(plans[method], len(trials_of(plans[method])[1]), args.device), flush=True)
    if plans and not args.dry_run and not args.yes:
        if not sys.stdin.isatty():
            raise cli.UsageError("a grid or a study is about to run: add --yes to start it without a question")
        if input("\nstart? [y/N] ").strip().lower() not in ("y", "yes"):
            raise cli.UsageError("not started")
    return plans


def trials_of(plan) -> tuple:
    """(all the trials, the finished ones as points) of the study of a plan, without creating it."""
    from tune_completion import STUDY_DIR, averages, open_study

    folder = Path(STUDY_DIR).expanduser()
    if not (folder / f"{plan.name}.db").is_file():
        return [], []
    trials = [dict(number=t.number, state=t.state.name, params=t.params or t.system_attrs.get("fixed_params", {}),
                   **({k: v for k, v in averages(t).items()} if t.state.name == "COMPLETE" else {}))
              for t in open_study(plan.name, folder).trials]
    return trials, [t for t in trials if t["state"] == "COMPLETE"]


def run_plan(plan, args) -> tuple:
    """Run (or continue) a grid or study: (the arguments of the completion at its best point, a note); (None, why) if
    it was stopped before its end."""
    from types import SimpleNamespace

    import yaml
    from hp.evaluation import tuning as T
    from tune_completion import STUDY_DIR, open_study, tune

    folder = Path(STUDY_DIR).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / f"{plan.name}.yaml"
    file.write_text(yaml.safe_dump(plan.config, sort_keys=False))
    total = plan.config["n_trials"]
    trials, points = trials_of(plan)
    if trials:  # trials stopped halfway count as finished for Optuna: they are run again
        total += sum(t["state"] in ("FAIL", "PRUNED") for t in trials)
        if again := T.missing(plan.grid, trials):
            study = open_study(plan.name, folder)
            for settings in again:
                study.enqueue_trial(settings)
    _, finished = tune(file, folder, SimpleNamespace(n_trials=total, summary=False, enqueue=[], device=args.device,
                                                     tf32=False))
    _, points = trials_of(plan)
    if not finished or len(points) < plan.config["n_trials"]:
        return None, f"stopped, {len(points)} of {plan.config['n_trials']} settings; the same command continues"
    point, warning = T.choose(points, plan.floor)
    settings = {**plan.config["fixed"], **point["params"]}
    note = f"{plan.mode} of {len(points)} settings, trial {point['number']} of {plan.name}"
    if plan.mode == "grid" and T.at_end(point, plan.grid):
        warning += ("; " if warning else "") + "the best point is at an end of the grid"
    print(f"\nchosen: {settings} ({point['mean']:.2f} / {point['best_of_k']:.2f} / {point['apd']:.2f})"
          + (f"; {warning}" if warning else ""))
    return ["--method", plan.method, "--settings", json.dumps(settings)], note + (f"; {warning}" if warning else "")


def all_parser(sub) -> None:
    from hp.evaluation import references as R

    p = sub.add_parser("all", help="generation, completion and denoising of a model, next to the references",
                       description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, metavar="[LABEL=]PATH", help="a run folder of RSGM (one model)")
    p.add_argument("--like", nargs="*", default=None, metavar="[PROTOCOL=]LABEL",
                   help="ready settings, ONE per procedure of the completion: without a value (or rsgm) the reference "
                        "set of the architecture, a LABEL that row or point; PROTOCOL=LABEL changes the generation "
                        '(heun) or the denoising (ours): denoising="rsgm budget"')
    p.add_argument("--tune", nargs="+", default=None, metavar="[METHOD=]MODE",
                   help="instead of --like for the completion: grid (10 settings, inpaint only), study (Optuna, 120 or "
                        "60 settings) or ready, for all the procedures or METHOD=MODE; one of --like and --tune must "
                        "be given when the completion runs, the number of settings is announced first")
    p.add_argument("--n-trials", type=int, default=None, help="of a study (default: 120 inpaint, 60 dflow)")
    p.add_argument("--yes", action="store_true", help="start a grid or study without the question")
    p.add_argument("--methods", nargs="+", default=list(R.METHODS), choices=R.METHODS,
                   help="the procedures of the completion (default both)")
    p.add_argument("--protocols", nargs="+", default=list(R.PROTOCOLS), choices=R.PROTOCOLS)
    p.add_argument("--split", choices=cli.SPLITS, default="valid", help="valid (default) or, explicitly, test")
    p.add_argument("--quick", action="store_true", help=f"{QUICK} poses, {DENOISING_QUICK} sequences, no likelihood")
    p.add_argument("--poses", type=int, default=None, help="the first n poses of the completion")
    p.add_argument("--sequences", type=int, default=None, help="the first n sequences of the denoising")
    add_all_common(p)
    p.add_argument("--reports", default="~/data/experiments/evaluate_all", help="where the summary is written (.md)")
    p.set_defaults(protocol=all_protocols)


def add_all_common(p) -> None:
    p.add_argument("--out", default=None, help="a results root with a folder per protocol (default: the usual caches)")
    p.add_argument("--device", default="cuda:0", help="cuda:N | mps | cpu")
    p.add_argument("--batch", type=int, default=BATCH, help="poses per batch of the completion; speed and memory only")
    p.add_argument("--recompute", action="store_true", help="replace results made with other settings")
    p.add_argument("--force", action="store_true", help="compute every result again")
    p.add_argument("--dry-run", action="store_true", help="print what would run and what is already there")


def all_protocols(args) -> int:
    from hp.evaluation import references as R

    config = R.load()
    label, sep, path = args.run.rpartition("=")
    try:
        path = R.resolve_run(path if sep else args.run)
        arch = R.architecture_of(path)
    except ValueError as error:
        raise cli.UsageError(str(error))
    base = label if sep else cli.default_label(path, {})
    if args.tune and "completion" not in args.protocols:
        raise cli.UsageError("--tune is of the completion, which is not among --protocols")
    how = completion_modes(args) if "completion" in args.protocols else {}
    if args.quick and (args.split == "test" or args.poses or args.sequences):
        raise cli.UsageError("--quick is a check on the validation with its own numbers of poses and sequences")
    print(f"{base}: {arch}, {path}")
    plans = tune_plans(args, config, path, arch, how)
    parser, entries, notes = build_parser(), [], []
    for protocol, group, arguments, note in profile(args.like, config, arch, args.methods):
        if protocol not in args.protocols:
            continue
        if protocol == "completion" and group in plans:
            if args.dry_run:
                continue
            arguments, note = run_plan(plans[group], args)
            if arguments is None:
                print(note)
                return 1
        name = run_label(base, protocol, group, arguments)
        print(f"\n== {protocol}{' ' + group if group else ''}: settings of {note}", flush=True)
        code, results = call(args, [protocol, "--run", f"{name}={path}", *arguments, *common_argv(args, protocol)], parser)
        if code:
            return code
        if name in results:
            spec, result = results[name]
            entries.append((note, entry_of(protocol, group, result, network=base, architecture=arch, label=base,
                                           spec=spec)))
        elif not args.dry_run:
            notes.append(f"{protocol} {group}: not complete, see above")
    if args.dry_run:
        return 0
    references = R.read()
    known = {R.entry_key(e) for e in references}
    references += [e for e in R.recorded_entries(config) if R.entry_key(e) not in known]
    if args.split == "test":  # the references, their front and the targets are numbers of the validation
        references, config = [], {**config, "front": [], "targets": {}}
    report = [f"# {base}\n", f"{arch}, `{path}`, {args.split}{' (--quick)' if args.quick else ''}, "
                             f"commit {cli.git_commit()}"]
    for note, entry in entries:
        title = f"{entry['protocol']}{' ' + entry['group'] if entry['group'] else ''}"
        report += [f"\n## {title}: n = {entry['n']}, settings of {note}\n", R.compare(entry, references, config)]
    own = {row["method"] for row in config["sets"] if R.resolve_or_none(row["run"]) == path}
    ready = [e["group"] for _, e in entries if e["protocol"] == "completion" and e["group"] not in own | set(plans)]
    if ready and args.split == "valid":
        report.append(f"\nCompletion {', '.join(ready)}: ready settings, not searched for this network, while the "
                      "references have their own: up to 2 mm of the mean may be the settings (see \"to the front\"). "
                      "--tune grid before a conclusion about small differences.")
    tuned = [e["group"] for n, e in entries if e["protocol"] == "completion" and e["group"] in plans]
    if tuned:
        report.append(f"\nCompletion {', '.join(tuned)}: the settings were chosen on these same validation poses, as "
                      "those of the references: a little optimistic for both, the test decides.")
    if entries and references:
        report.append("\nIn brackets: the model minus the reference; `~` within the noise of the poses and draws (not of "
                      "the training seeds), `?` noise unknown. MPVPE / MPJPE in mm, APD in cm, W / recall / precision in "
                      "deg per joint. \"to the front\": mean minus the mean of the front of the reference studies at "
                      "the same APD (negative: better).")
    report = "\n".join(report + [f"\n{n}" for n in notes])
    print("\n" + report)
    folder = Path(args.reports).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    amount = "_quick" if args.quick else "".join(
        f"_{k}{v}" for k, v in (("poses", args.poses), ("seq", args.sequences)) if v)
    file = folder / (base.replace("/", "_").replace(" ", "_") + f"_{args.split}{amount}.md")
    file.write_text(report + "\n")
    print(f"\nwritten to {file}")
    return 0


def references_parser(sub) -> None:
    from hp.evaluation import references as R

    p = sub.add_parser("references", help="the reference networks through all the protocols, on the validation",
                       description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocols", nargs="+", default=list(R.PROTOCOLS), choices=R.PROTOCOLS)
    p.add_argument("--networks", nargs="+", default=None, help='only these, e.g. "small c10"')
    add_all_common(p)
    p.set_defaults(protocol=references)


def references(args) -> int:
    """The rows of config/references/posedflow.yaml on the validation; the numbers are added to the files of
    hp.evaluation.references after every row, so a stopped run keeps the finished ones."""
    from hp.evaluation import references as R

    args.split = "valid"
    config, parser = R.load(), build_parser()
    if unknown := set(args.networks or []) - {row["network"] for row in config["sets"]}:
        raise cli.UsageError(f"no reference networks {sorted(unknown)}")
    rows = [row for row in config["sets"] if not args.networks or row["network"] in args.networks]
    networks = {row["network"]: row for row in rows}
    jobs = [(protocol, "", ["--like", PROFILES[protocol]], row) for protocol in ("generation", "denoising")
            for row in networks.values()]
    jobs[len(networks):len(networks)] = [
        ("completion", row["method"], ["--method", row["method"], "--settings", json.dumps(row["settings"])], row)
        for row in rows]
    for protocol, group, arguments, row in jobs:
        if protocol not in args.protocols:
            continue
        name = run_label(row["network"], protocol, group, arguments)
        print(f"\n== {protocol} {group}: {row['architecture']}, {row['network']}", flush=True)
        argv = [protocol, "--run", f"{name}={R.resolve_run(row['run'])}", *arguments, *common_argv(args, protocol)]
        code, results = call(args, argv, parser)
        if code:
            return code
        if name not in results:
            continue
        spec, result = results[name]
        entry = entry_of(protocol, group, result, network=row["network"], architecture=row["architecture"],
                         label=row["network"], spec=spec)
        entry.update(commit=cli.git_commit(), device=args.device)
        R.save([entry])
        if protocol == "completion":
            print("recorded / now: " + ", ".join(f"{k} {v:.2f} / {entry['numbers'][k]:.2f}"
                                                 for k, v in row["validation"].items()))
    if not args.dry_run:
        print(f"\n{len(R.read())} rows in {R.NUMBERS} and {R.ARRAYS}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="name", required=True, metavar="protocol")
    for add in (completion_parser, denoising_parser, generation_parser, all_parser, references_parser):
        add(sub)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.name not in ("all", "references"):
            cli.resolve_poses(args)
        return args.protocol(args)
    except cli.UsageError as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
