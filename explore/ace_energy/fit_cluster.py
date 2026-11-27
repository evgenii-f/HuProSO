"""fit.py on the training poses of one cluster: a component of a mixture of linear energies (a probe of the mixture).
The clusters are the k-means centroids of mixture_clusters.py (--save; the 171 entries of the 19 local rotations); every
training pose goes to its nearest centroid, and fit.py sees only the poses of cluster --index, also for its joint
density. Everything after "--" goes to fit.py unchanged. From the root of the repository:

    python explore/ace_energy/fit_cluster.py --clusters ~/data/experiments/ace/mixture_clusters.pt --k 8 --index 4 -- \\
        --wigner 1 2 3 --pairs 1 2 3 --products --single --ridge 1e-3 --joint-density 5000 ... --device cuda:0 --out ...
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fit  # noqa: E402


def main() -> None:
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clusters", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args(argv[:cut])
    centroids = torch.load(Path(args.clusters).expanduser(), weights_only=True)[args.k]["centroids"]
    load = fit.load_reference

    def cluster_only(split, *rest, **kw):
        x = load(split, *rest, **kw)
        if split != "train":
            return x
        label = torch.cat([torch.cdist(x[i:i + 100_000].flatten(1).float(), centroids).argmin(1)
                           for i in range(0, len(x), 100_000)])
        keep = x[label == args.index]
        print(f"cluster {args.index} of K = {args.k}: {len(keep):,} of {len(x):,} training poses", flush=True)
        return keep

    fit.load_reference = cluster_only
    sys.argv = [sys.argv[0]] + argv[cut + 1:]
    fit.main()


if __name__ == "__main__":
    main()
