"""Evaluate the models of explore/evaluation/pareto_posedflow_vs_rsgm.ipynb on this machine (e.g. a GPU server) and write
the results into its cache: copy the cache folder back and the notebook reuses them. The models, settings and cache
are those of hp.evaluation.pareto, which the notebook uses too.

    python scripts/evaluate_pareto.py --list
    python scripts/evaluate_pareto.py --device cuda:0 --stages runtime
    python scripts/evaluate_pareto.py --device cuda:0 --models "RSGM 512x4" DPoser --recompute-runtime
    python scripts/evaluate_pareto.py --device cuda:0 --stages runtime --batches 1 16 256 1024 4096 16384

Stages: harness (Wasserstein, recall, precision, log-likelihood), paper (FID, FID own, APD, dNN) and runtime (the time
of model.sample(n) for each n of --batches, saved per chip and device as runtime_<chip>_<device>.json). A result
already in the cache with the same settings is not computed again (unless asked with --recompute*). The runtime file
keeps every n measured so far: new n are added to it, and a file of another torch (or other weights) is set aside as
runtime_<chip>_<device>.torch_<version>.json instead of being overwritten. Models whose runs are not on this machine
are skipped.

Needs on the machine: the runs (or their model.json and weights) at the paths of MODELS; for harness and paper also
the data (ROTATIONS, train and test), the SMPL skeleton and their FID statistics. SKELETON and GT_STATS are part of
the settings; if the files lie elsewhere here, say where with --skeleton and --gt-stats (only read from there).

Times on a GPU shared with other jobs are slower: each runtime result keeps the date, the load average and (CUDA) the
number of processes on the GPU, under "conditions".
"""
import argparse
import sys
from pathlib import Path

# the code of this checkout, before an editable install of another one
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hp.evaluation import pareto as P  # noqa: E402
from hp.evaluation.data import load_reference  # noqa: E402
from hp.evaluation.evaluate import EvalConfig  # noqa: E402

STAGES = ("harness", "paper", "runtime")


def recompute_set(values):
    """None: nothing; [] (the flag alone): all; names: those models (or runs)."""
    return frozenset() if values is None else (True if not values else frozenset(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None,
                        help="names or short labels of MODELS, or parts of them (case-insensitive); default: all")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--device", default=None, help="cuda:N | mps | cpu (default: mps, else cuda, else cpu)")
    parser.add_argument("--recompute", nargs="*", default=None, metavar="NAME",
                        help="compute harness and paper again: for these models (or runs), or all without names")
    parser.add_argument("--recompute-runtime", nargs="*", default=None, metavar="NAME",
                        help="time the n of --batches again: for these models, or all without names")
    parser.add_argument("--batches", nargs="+", type=int, default=list(P.BATCHES),
                        help="the n of model.sample(n) to time (those not in the cache yet)")
    parser.add_argument("--repeats", type=int, default=P.REPEATS, help="timed runs per n (an n timed with others is timed again)")
    parser.add_argument("--cache", default=str(P.CACHE))
    parser.add_argument("--skeleton", default=P.SKELETON, help="where the SMPL skeleton is on this machine")
    parser.add_argument("--gt-stats", default=P.GT_STATS, help="where their FID statistics are on this machine")
    parser.add_argument("--ll-chunk", type=int, default=250, help="poses per log_prob call in the harness (memory)")
    parser.add_argument("--smoke", action="store_true", help="tiny settings, to test the script (cached in <cache>_smoke, so the real results stay)")
    parser.add_argument("--list", action="store_true", help="list the models, whether they are here, and stop")
    args = parser.parse_args()

    device = args.device or P.default_device()
    cache = Path(args.cache).expanduser()
    try:
        models = P.select(args.models)
    except KeyError as error:
        sys.exit(f"{error.args[0]}; see --list")

    def log(message):
        print(message, flush=True)

    zoo = P.Zoo(models, device, log=log)
    if args.list:
        for name, spec in models.items():
            here = len(zoo.groups.get(name, []))
            log(f"{'+' if here else '-'} {name}   [{spec['short']}]   runs here: {here}/{len(P.members(name, spec))}")
        return
    if not zoo.builders:
        sys.exit("none of the chosen models is on this machine; see --list")

    harness, n_contrast, sets, set_size = P.HARNESS, P.N_CONTRAST, P.SETS, P.SET_SIZE
    batches, repeats = args.batches, args.repeats
    if args.smoke:
        harness = EvalConfig(n_wasserstein=100, n_queries=100, n_generated=500, n_likelihood=20, repeats=2)
        n_contrast, sets, set_size, batches, repeats = 2_000, 2, 100, [64, 1], 1
        cache = cache.with_name(cache.name + "_smoke")

    log(f"device {device}, {len(zoo.builders)} runs of {len(zoo.groups)} models, cache {cache}")
    quality = [s for s in ("harness", "paper") if s in args.stages]
    if quality:
        test = load_reference("test", P.ROTATIONS, "posedflow")
        train = load_reference("train", P.ROTATIONS, "posedflow")
        log(f"test {len(test):,}   train {len(train):,}")
        protocol = P.Protocol(train, device, sets, set_size, skeleton=args.skeleton, gt_stats=args.gt_stats)
        h, p = P.evaluate_quality(zoo, test, train, recompute_set(args.recompute), harness, n_contrast, protocol,
                                  stages=quality, build_wrapper=lambda m: P.chunked_log_prob(m, args.ll_chunk), cache=cache, log=log)
        for label in zoo.builders:
            parts = []
            if label in h:
                parts.append(f"Wasserstein {h[label]['wasserstein']['generated']['joint_mean_deg']['mean']:.2f}")
            if label in p:
                parts.append(f"FID {p[label]['FID']['mean']:.4f}, APD {p[label]['APD']['mean']:.2f}")
            log(f"{label}: {', '.join(parts)}")
    if "runtime" in args.stages:
        log(f"timing on {P.chip_name(device)} ({device}), saved as {P.runtime_stage(device)}.json")
        runtime = P.measure_runtime(zoo, recompute_set(args.recompute_runtime), batches, repeats, cache, log)
        for name, r in runtime.items():
            log(f"{name}: " + ", ".join(f"n={n} {b['us_per_pose']:,.1f} us/pose" for n, b in r["batches"].items()))


if __name__ == "__main__":
    main()
