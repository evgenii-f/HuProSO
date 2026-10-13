"""Train a model of the zoo, e.g.

    python scripts/train_model.py model=so3_flow train.max_iterations=2000

The settings are in config/train.yaml and config/models/<model>.yaml; command line arguments override them.
"""
from hp.training.cli import main

if __name__ == "__main__":
    main()
