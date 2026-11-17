"""Registry-based model construction.

New CNN variants can be added by registering a builder here and then
selecting its name in the training configuration; no training or inference
code needs to change.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from models.cnn import SimpleCNN

ModelBuilder = Callable[..., nn.Module]

MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "simple_cnn": SimpleCNN,
}


def list_models() -> tuple[str, ...]:
    """Return the available model names."""
    return tuple(MODEL_REGISTRY)


def create_model(
    name: str = "simple_cnn",
    *,
    num_classes: int = 10,
    **kwargs: object,
) -> nn.Module:
    """Instantiate a registered model.

    Args:
        name: Key in :data:`MODEL_REGISTRY`.
        num_classes: Number of output logits.
        **kwargs: Extra arguments forwarded to the model builder.

    Raises:
        ValueError: If the model name is not registered.
    """
    key = name.strip().lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available models: {', '.join(list_models())}.")
    return MODEL_REGISTRY[key](num_classes=num_classes, **kwargs)
