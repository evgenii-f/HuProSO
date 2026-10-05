"""Crop and subsample the raw AMASS sequences, e.g. for quick experiments on a smaller dataset.

    python scripts/preprocess/compress_amass.py                      # settings of config/compress.yaml
    python scripts/preprocess/compress_amass.py stride=10 crop_start_s=0.5 datasets=[SSM,CMU]

The result has the layout of the raw data. To use it, set amass_raw (config/paths.yaml) to the
output_dir, set amass_rotations to a new folder and run the preprocessing as usual.
"""
from omegaconf import OmegaConf

from hp.data.compression import run, summary
from hp.utils.paths import REPO_ROOT


def main():
    cfg = OmegaConf.merge(OmegaConf.load(REPO_ROOT / "config" / "compress.yaml"), OmegaConf.from_cli())
    print(OmegaConf.to_yaml(cfg))
    print(summary(run(cfg)))


if __name__ == "__main__":
    main()
