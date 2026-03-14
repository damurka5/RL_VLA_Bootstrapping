"""Embodiment-first RL to VLA bootstrapping framework."""

from .core.config import load_project_config
from .pipeline.bootstrap import BootstrapPipeline

__all__ = ["BootstrapPipeline", "load_project_config"]
