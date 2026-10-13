"""Evaluate a trained run of the model zoo, e.g.

    python scripts/evaluate_model.py run=~/data/experiments/so3_flow/run_0001 split=test

The settings are in config/evaluate.yaml; command line arguments override them.
"""
from hp.evaluation.cli import main

if __name__ == "__main__":
    main()
