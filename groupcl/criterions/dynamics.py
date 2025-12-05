from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from torch import Tensor
from jaxtyping import Float
from jaxtyping import jaxtyped
from jaxtyping._typeguard import typechecked

from config_dataclass import Configurable, torch_dataclass, config_field, check_initialized

from groupcl.utils.datatypes import ContrastiveLossBatch


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class MSECriterion(Configurable, torch.nn.Module, ABC):
    """Wrapper around torch.nn.MSELoss."""

    reduction: Literal["mean", "sum"] = config_field(default="mean")

    def __new__(cls, *args, **k):
        inst = super().__new__(cls)
        torch.nn.Module.__init__(inst)
        return inst

    def __lazy_post_init__(self):
        super().__lazy_post_init__()
        self.criterion = torch.nn.MSELoss(reduction=self.reduction)

    @jaxtyped(typechecker=typechecked)
    def forward(
        self,
        input: Float[Tensor, "batch_size dim"],
        target: Float[Tensor, "batch_size dim"],
    ) -> Float[Tensor, " "]:
        """Compute the MSELoss.

        Args:
            input: The input samples of shape `(n, d)`.
            target: The target samples of shape `(n, d)`.

        Returns:
            The MSELoss.
        """
        return self.criterion(input, target)
