"""Evaluate a trained run from the command line (scripts/evaluate_model.py). Settings: config/evaluate.yaml."""
import sys
from pathlib import Path
from typing import Optional, Sequence

from omegaconf import OmegaConf

from hp.evaluation.data import load_reference
from hp.evaluation.evaluate import EvalConfig, evaluate_run
from hp.utils.device import resolve_device
from hp.utils.paths import REPO_ROOT


def format_report(report: dict) -> str:
    """The main numbers of a report as text."""
    w, m = report["wasserstein"], report["model"]
    pm = lambda x: f"{x['mean']:.2f} +- {x['std']:.2f}"
    lines = [
        f"{m['name']}: {m['parameters']:,} parameters, {report['sampling']['seconds_per_1000']:.2f} s per 1000 samples",
        f"real poses: {report['reference']['poses']:,}",
        f"wasserstein (deg per joint): generated {pm(w['generated']['joint_mean_deg'])}, same-set floor {pm(w['floor']['joint_mean_deg'])}",
    ]
    for name in ("recall", "precision", "baseline"):
        lines.append(f"{name if name != 'baseline' else 'same-set':9s} (deg per joint): mean {pm(report[name]['mean_deg'])}, p95 {report[name]['p95_deg']['mean']:.2f}, "
                     f"within 5 deg {report[name]['within_5deg']['mean']:.2f}")
    if "contrast" in w:
        lines.append(f"contrast  (deg per joint): wasserstein {pm(w['contrast']['joint_mean_deg'])}, recall {pm(report['contrast_recall']['mean_deg'])}, "
                     f"precision {pm(report['contrast_precision']['mean_deg'])}")
        lines.append("  contrast = real poses of other subjects: what exactly reproducing the training poses would score. "
                     "Compare the model with it; the same-set numbers are no goal (neighbouring frames are nearly identical).")
    likelihood = report["likelihood"]
    lines.append("log-likelihood: " + ("none (the model has no exact density)" if likelihood is None
                                       else f"{likelihood['mean']:.2f} +- {likelihood['std']:.2f} on {likelihood['space']}"))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> dict:
    """Evaluate as configured and write the report. Returns it."""
    cfg = OmegaConf.merge(OmegaConf.load(REPO_ROOT / "config" / "evaluate.yaml"),
                          OmegaConf.from_dotlist(list(sys.argv[1:] if argv is None else argv)))
    if not cfg.run:
        raise ValueError("Which run? Pass run=<folder with model.json and weights.pt>.")
    run = Path(str(cfg.run)).expanduser()
    reference = load_reference(cfg.split, cfg.data.rotations, cfg.data.split)
    contrast = None
    if cfg.contrast:
        contrast = load_reference(cfg.contrast, cfg.contrast_rotations or cfg.data.rotations, cfg.data.split)
    output = cfg.output or run / f"metrics_{cfg.split}.json"
    device = resolve_device(None if cfg.device == "auto" else cfg.device)
    report = evaluate_run(run, reference, EvalConfig(**OmegaConf.to_container(cfg.eval)), weights=cfg.weights,
                          output=output, device=device, contrast=contrast, progress=cfg.progress)
    print(format_report(report))
    print(f"Report written to {output}")
    return report
