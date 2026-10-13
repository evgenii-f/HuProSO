"""Models are registered by name, so that a config can say which one to build."""
from typing import Dict, List, Type

from hp.models.base import PoseModel

_REGISTRY: Dict[str, Type[PoseModel]] = {}


def register_model(name: str):
    """Class decorator: `@register_model("so3_flow")`."""

    def decorator(cls):
        if not (isinstance(cls, type) and issubclass(cls, PoseModel)):
            raise TypeError(f"{cls} is not a PoseModel.")
        if name in _REGISTRY:
            raise ValueError(f"A model named '{name}' is already registered ({_REGISTRY[name].__name__}).")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_models() -> List[str]:
    return sorted(_REGISTRY)


def get_model_class(name: str) -> Type[PoseModel]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Available: {available_models()}.") from None


def build_model(name: str, num_joints: int, **kwargs) -> PoseModel:
    """Build a registered model; the arguments are kept in `model.build_args` to rebuild it later."""
    model = get_model_class(name)(num_joints=num_joints, **kwargs)
    model.build_args = dict(name=name, num_joints=num_joints, **kwargs)
    return model
