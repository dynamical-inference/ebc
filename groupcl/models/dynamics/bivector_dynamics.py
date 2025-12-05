from dataclasses import dataclass
from typing import Tuple, Literal, Optional

import torch
from jaxtyping import Float
from jaxtyping import Integer
from jaxtyping import jaxtyped
from torch import Tensor
from jaxtyping._typeguard import typechecked

from groupcl.models.dynamics.base import BaseDynamicsModel

from groupcl.utils.bivector import bivector_rotation, subspace_projection_matrix
from config_dataclass import torch_dataclass, config_field


@torch_dataclass
class BivectorDynamicsModel(BaseDynamicsModel):
    """
    A dynamics model that infers the bivector rotation matrix as a predictor of the transformed states.
    """

    dtype: Literal["torch.float32",
                   "torch.float64"] = config_field(default="torch.float64")
    eps: float = config_field(default=1e-8)
    projection_type: Literal["orthogonal",
                             "identity"] = config_field(default="orthogonal")
    norm_projection: bool = config_field(default=True)

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "torch.float32":
            return torch.float32
        elif self.dtype == "torch.float64":
            return torch.float64
        else:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    @jaxtyped(typechecker=typechecked)
    def forward(
        self,
        x: Float[Tensor, "batch dim"],
        y: Float[Tensor, "batch dim"],
        x_prime: Float[Tensor, "batch dim"],
        y_prime: Float[Tensor, "batch dim"],
        **kwargs,
    ) -> Tuple[
            Float[Tensor, "batch dim"],
            Float[Tensor, "batch dim"],
            Float[Tensor, "batch dim"],
    ]:
        """
        Computes the bivector rotation matrix Qx from x, x_prime. 
        Projects y and y_prime into the plane spanned by x and x_prime.
        Applies the rotation matrix Qx to y and y_prime in the projected space.
        Optionally, projects y_negs into the x, x_prime plane.

        Args: Paired Group Action data
            x: Pair 1 original state
            y: Pair 2 original state
            x_prime: Pair 1 transformed state
            y_prime: Pair 2 transformed state

        Returns:
            Tuple containing:
            - yp: y projected into span(x, x_prime)
            - yp_prime: y_prime projected into span(x, x_prime)
            - yp_prime_pred: yp_prime rotated by Qx
        """

        x = x.clone().to(self.torch_dtype)
        x_prime = x_prime.clone().to(self.torch_dtype)

        y = y.clone().to(self.torch_dtype)
        y_prime = y_prime.clone().to(self.torch_dtype)

        Qx = bivector_rotation(origin=x, target=x_prime, eps=self.eps)

        if self.projection_type == "orthogonal":
            P = subspace_projection_matrix(x, x_prime)
        elif self.projection_type == "identity":
            P = torch.eye(x.shape[-1], device=x.device, dtype=self.torch_dtype)
        else:
            raise ValueError(f"Invalid projection type: {self.projection_type}")

        yp = torch.einsum("...ij,...j->...i", P, y)
        yp_prime = torch.einsum("...ij,...j->...i", P, y_prime)

        if self.norm_projection:
            yp = yp / (torch.norm(yp, dim=-1, keepdim=True))
            yp_prime = yp_prime / (torch.norm(yp_prime, dim=-1, keepdim=True))

        y_prime_pred = torch.einsum("...ij,...j->...i", Qx, yp)

        return (yp, yp_prime, y_prime_pred)
