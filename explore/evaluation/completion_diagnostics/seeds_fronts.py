"""Band mean and hypervolume of the fact_RDW beta 0 test fronts per training seed (completion report, section 19).

Reads the result tables of the test logs (5000 test poses, K = 10): seed 10 from the data weighting test
(`~/data/experiments/data_weighting/test_fast.log`), seeds 11 and 12 from `~/data/experiments/seeds_rdw/test.log`,
and prints, per procedure and seed, the band mean (mm) and the hypervolume of best of 10 over APD 4.0-8.5 cm, as
`hp.evaluation.fronts.hypervolume` defines them (reference 115 / 65 mm).

    PYTHONPATH=src python explore/evaluation/completion_diagnostics/seeds_fronts.py
"""
import re
from pathlib import Path

from hp.evaluation.fronts import APD_RANGE, HV_REFERENCE, hypervolume

DATA = Path("~/data/experiments").expanduser()
LOGS = {"10": DATA / "data_weighting/test_fast.log", "11": DATA / "seeds_rdw/test.log",
        "12": DATA / "seeds_rdw/test.log"}
STUDIES = {"inpainting": "rsgm_inpaint_mean_apd_rdw_b0", "SDS": "rsgm_sds_mean_apd_rdw_b0"}


def points(log: Path, study: str, seed: str) -> list:
    """The test points of one seed and study: rows | model | 4 occlusions | mean | best of k | APD |."""
    rows = []
    for line in log.read_text().splitlines():
        if not line.startswith("|") or f"{study} " not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        found = re.search(r"seed (\d+)", cells[0])
        row_seed = "10" if "fact_RDW_s10" in cells[0] else (found.group(1) if found else None)
        if row_seed == seed:
            rows.append({"mean": float(cells[5]), "best_of_k": float(cells[6]), "apd": float(cells[7])})
    return rows


def main() -> None:
    width = APD_RANGE[1] - APD_RANGE[0]
    print("| procedure | seed | points | APD of points | band mean | HV best of 10 |")
    print("|---|---|---|---|---|---|")
    for name, study in STUDIES.items():
        for seed, log in LOGS.items():
            rows = points(log, study, seed)
            band = HV_REFERENCE["mean"] - hypervolume(rows, "mean") / width
            apd = [r["apd"] for r in rows]
            print(f"| {name} | s{seed} | {len(rows)} | {min(apd):.2f}-{max(apd):.2f} | {band:.2f} | "
                  f"{hypervolume(rows, 'best_of_k'):.2f} |")


if __name__ == "__main__":
    main()
