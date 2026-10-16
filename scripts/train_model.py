"""Train a model of the zoo, e.g.

    python scripts/train_model.py model=so3_flow train.max_iterations=2000

The settings are in config/train.yaml and config/models/<model>.yaml; command line arguments override them.

Go on with a run that stopped (the time limit, SIGTERM from a job scheduler, a crash) in its run folder:

    python scripts/train_model.py resume=<run folder>

The exit status is 0 when the run is finished and 75 when it stopped early and can be resumed.
"""
import sys

from hp.training.cli import exit_code, main

if __name__ == "__main__":
    sys.exit(exit_code(main()))
