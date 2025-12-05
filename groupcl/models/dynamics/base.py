import importlib
import pkgutil
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Type, TypeVar
from config_dataclass import torch_dataclass, config_field
from groupcl.models.base import BaseModel

T = TypeVar("T", bound="BaseDynamicsModel")


@torch_dataclass
class BaseDynamicsModel(BaseModel, ABC):
    """Base class for all dynamics models. Mostly for type checking."""

    index_forward_offset: int = config_field(default=1)

    @classmethod
    def config_prefix(cls) -> str:
        """Prefix for the configuration file."""
        return "dynamics_model"

    @abstractmethod
    def forward(
        self,
        **kwargs,
    ):
        pass

    def to_gt_dynamics(self: T) -> T:
        """Create a new dynamics model that represents the same dynamics and can be used for training"""
        return self.clone()
