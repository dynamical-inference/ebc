from abc import abstractmethod

import cebra
import torch
from config_dataclass import config_field, torch_dataclass
from jaxtyping import Float, jaxtyped
from jaxtyping._typeguard import typechecked
from torch import Tensor, nn

from groupcl.models.base import BaseModel
from groupcl.models.utils import ModelStorageMixin
from groupcl.utils.complex import (
    complex_data_to_real,
    real_data_to_complex,
)


class NormLayer(nn.Module):
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return inp / torch.norm(inp, dim=-1, keepdim=True)


class NormLayerSplit(NormLayer):
    def __init__(self, content_dims: int, normalize_content: bool, normalize_group: bool):
        super().__init__()
        self.content_dims = content_dims
        self.normalize_content = normalize_content
        self.normalize_group = normalize_group

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        latent_dim = inp.shape[-1]
        # separate across samples_per_system dimension
        content_start = latent_dim - self.content_dims
        content_end = latent_dim
        content_slice = slice(content_start, content_end)
        content_embeddings = inp[..., content_slice]

        group_start = 0
        group_end = content_start
        group_slice = slice(group_start, group_end)
        group_embeddings = inp[..., group_slice]

        if self.normalize_content:
            content_embeddings = super().forward(content_embeddings)
        if self.normalize_group:
            group_embeddings = super().forward(group_embeddings)

        return torch.cat([group_embeddings, content_embeddings], dim=-1)


class NormComplexLayer(NormLayer):
    """
    Normalizes the across the group dimensions, such that when converting
    the embeddings to a complex domain, the resulting complex matrix is unitary.
    I.e. magnitude of the complex numbers are all 1.
    """

    def __init__(self, content_dims: int = 0, normalize_content: bool = False, epsilon: float = 1e-8):
        super().__init__()
        self.content_dims = content_dims
        self.normalize_content = normalize_content
        self.epsilon = epsilon

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        latent_dim = inp.shape[-1]
        # separate across samples_per_system dimension
        content_start = latent_dim - self.content_dims
        content_end = latent_dim
        content_slice = slice(content_start, content_end)
        content_embeddings = inp[..., content_slice]

        if self.normalize_content:
            content_embeddings = super().forward(content_embeddings)

        group_start = 0
        group_end = content_start
        group_slice = slice(group_start, group_end)
        group_embeddings = inp[..., group_slice]

        # to complex domain
        group_embeddings = real_data_to_complex(group_embeddings)

        # normalize the complex embeddings
        magnitude = torch.abs(group_embeddings)
        magnitude_safe = torch.clamp(magnitude, min=self.epsilon)
        group_embeddings = group_embeddings / magnitude_safe

        # to real domain
        group_embeddings = complex_data_to_real(group_embeddings)

        return torch.cat([group_embeddings, content_embeddings], dim=-1)


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class EncoderModel(ModelStorageMixin, BaseModel):
    """Base class for all encoder models."""

    input_dim: int = config_field(default=5)
    output_dim: int = config_field(default=2)
    normalize: bool = config_field(default=False)

    @classmethod
    def config_prefix(cls) -> str:
        """Prefix for the configuration file."""
        return "encoder_model"

    def __new__(cls, *args, **k):
        inst = super().__new__(cls)
        torch.nn.Module.__init__(inst)
        return inst

    def __lazy_post_init__(self):
        ModelStorageMixin.__lazy_post_init__(self)
        BaseModel.__lazy_post_init__(self)
        self.init_parameters()
        if self.normalize:
            self.norm_layer = NormLayer()

    @abstractmethod
    def init_parameters(self):
        pass

    @abstractmethod
    def _forward(self, x: Float[Tensor, " *batch {self.input_dim}"]) -> Float[Tensor, " *batch {self.output_dim}"]:
        pass

    @jaxtyped(typechecker=typechecked)
    def forward(self, x: Float[Tensor, " *batch {self.input_dim}"]) -> Float[Tensor, " *batch {self.output_dim}"]:
        x = self._forward(x)
        if self.normalize:
            x = self.norm_layer(x)
        return x


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class MLP(EncoderModel):
    """Simple MLP encoder model."""

    hidden_dim: int = config_field(default=128)
    num_layers: int = config_field(default=3)

    def init_parameters(self):
        first_layers = [
            nn.Linear(
                self.input_dim,
                self.hidden_dim,
            ),
            nn.GELU(),
        ]
        middle_layers = []
        for _ in range(self.num_layers - 1):
            middle_layers.extend(
                [
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                ]
            )
        last_layers = [
            nn.Linear(self.hidden_dim, self.output_dim),
        ]
        layers = first_layers + middle_layers + last_layers
        self.net = nn.Sequential(*layers)

    @jaxtyped(typechecker=typechecked)
    def _forward(self, x: Float[Tensor, " batch {self.input_dim}"]) -> Float[Tensor, " batch {self.output_dim}"]:
        return self.net(x)


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class MLPSubSpaceNormed(MLP):
    """
    Regular MLP encoder, but normalizing the group and content subspaces indivudually
    """

    content_dims: int = config_field(default=0)
    normalize_content: bool = config_field(default=True)
    normalize_group: bool = config_field(default=True)

    def __post_init__(self):
        super().__post_init__()
        # always use the norm layer
        self.normalize = True

    def init_parameters(self):
        super().init_parameters()

        # overwrite norm layer
        self.norm_layer = NormLayerSplit(
            content_dims=self.content_dims,
            normalize_content=self.normalize_content,
            normalize_group=self.normalize_group,
        )


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class MLPNormedComplex(MLP):
    content_dims: int = config_field(default=0)
    normalize_content: bool = config_field(default=False)
    epsilon: float = config_field(default=1e-8)

    def __post_init__(self):
        super().__post_init__()
        # always use the norm layer
        self.normalize = True

    def init_parameters(self):
        super().init_parameters()
        self.norm_layer = NormComplexLayer(
            content_dims=self.content_dims,
            normalize_content=self.normalize_content,
            epsilon=self.epsilon,
        )


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class Offset1ModelMLP(EncoderModel):
    """Simple MLP encoder model equivalent to Offset1Model-MSE from cebra."""

    hidden_dim: int = config_field(default=128)
    num_layers: int = config_field(default=3)

    def init_parameters(self):
        print("Ignoring hidden_dim and num_layers with Offset1ModelMLP")

        layers = [
            nn.Linear(
                self.input_dim,
                self.output_dim * 30,
            ),
            nn.GELU(),
            nn.Linear(self.output_dim * 30, self.output_dim * 30),
            nn.GELU(),
            nn.Linear(self.output_dim * 30, self.output_dim * 10),
            nn.GELU(),
            nn.Linear(int(self.output_dim * 10), self.output_dim),
        ]
        self.net = nn.Sequential(*layers)

    @jaxtyped(typechecker=typechecked)
    def _forward(
        self,
        x: Float[Tensor, " *batch  {self.input_dim}"],
    ) -> Float[Tensor, " *batch {self.output_dim}"]:
        return self.net(x)


@torch_dataclass
class IdentityEncoder(EncoderModel):
    """Identity model for debugging purposes."""

    def init_parameters(self):
        self.b = nn.Parameter(torch.zeros(self.input_dim))

    @jaxtyped(typechecker=typechecked)
    def _forward(self, x: Float[Tensor, " *batch  {self.input_dim}"]) -> Float[Tensor, " *batch {self.output_dim}"]:
        """Compute the embedding given the input signal.

        Args:
            inp: The input tensor of shape `num_samples x self.num_input x time`

        Returns:
            The output tensor of shape `num_samples x self.num_output x time`.

        """
        x = x + self.b
        x = x - self.b
        return x


@torch_dataclass
class LinearEncoderModel(EncoderModel):
    """Simple linear encoder model."""

    def init_parameters(self):
        self.linear = nn.Linear(self.input_dim, self.output_dim)

    @jaxtyped(typechecker=typechecked)
    def _forward(self, x: Float[Tensor, " *batch  {self.input_dim}"]) -> Float[Tensor, " *batch {self.output_dim}"]:
        """Linear transformation of input.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Transformed tensor of shape (batch_size, output_dim)
        """
        return self.linear(x)


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class CebraModel(EncoderModel):
    """Simple MLP encoder model."""

    architecture: str = config_field(default="offset10-model-mse")
    hidden_dim: int = config_field(default=128)

    def init_parameters(self):
        self.net = cebra.models.init(
            self.architecture,
            num_neurons=self.input_dim,
            num_units=self.hidden_dim,
            num_output=self.output_dim,
        )
        # the input dimension is actually not the num_neurons
        # rather num_neurons is input_dim // offset
        # to get the offset from the architecture name we first initialize the cebra model,
        # get the offset, then re-initialize the model with the correct num_neurons
        self.offset = len(self.net.get_offset())
        assert self.input_dim % self.offset == 0, "Input dimension must be divisible by offset"
        self.num_neurons = self.input_dim // self.offset

        self.net = cebra.models.init(
            self.architecture,
            num_neurons=self.num_neurons,
            num_units=self.hidden_dim,
            num_output=self.output_dim,
        )

    @jaxtyped(typechecker=typechecked)
    def _forward(self, x: Float[Tensor, " batch {self.input_dim}"]) -> Float[Tensor, " batch {self.output_dim}"]:
        # we need to reshape the input to (batch, num_neurons, offset)
        x = x.reshape(x.shape[0], self.offset, self.num_neurons).swapaxes(-2, -1)
        return self.net(x).reshape(x.shape[0], self.output_dim)


__all__ = [
    "EncoderModel",
    "MLP",
    "Offset1ModelMLP",
    "IdentityEncoder",
    "LinearEncoderModel",
    "CebraModel",
]
