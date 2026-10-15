"""Quantitative Attribution with Counterfactuals."""

__version__ = "0.3.0"


def __getattr__(name):
    if name in {"train", "explain"}:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
