import logging

import torch

from hp.utils.paths import get_path


def _available(device: str) -> bool:
    kind = torch.device(device).type
    if kind == "cuda":
        return torch.cuda.is_available() and torch.device(device).index in (
            None,
            *range(torch.cuda.device_count()),
        )
    if kind == "mps":
        return torch.backends.mps.is_available()
    return True


def resolve_device(requested: str = None) -> str:
    """Device to use. The 'device' setting (config/paths.yaml, HUPROSO_DEVICE) wins
    unless it is 'auto'. With 'auto' the requested device (e.g. the one stored in an
    experiment config) is used if available, otherwise cuda, otherwise cpu."""
    setting = get_path("device")
    if setting != "auto":
        return setting
    if requested and _available(requested):
        return str(requested)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested:
        logging.warning(f"Device {requested} is not available, using {device}.")
    return device
