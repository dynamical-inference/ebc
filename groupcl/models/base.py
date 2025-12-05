from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Type

import torch

from config_dataclass import Configurable, torch_dataclass, config_field, check_initialized


@torch_dataclass
class BaseModel(Configurable, torch.nn.Module, ABC):
    """Base class for all models."""

    @classmethod
    def config_prefix(cls) -> str:
        """Prefix for the configuration file."""
        return "model"

    def __new__(cls, *args, **k):
        inst = super().__new__(cls)
        torch.nn.Module.__init__(inst)
        return inst

    @abstractmethod
    def forward(
        self,
        **kwargs,
    ):
        pass
