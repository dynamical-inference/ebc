from matplotlib import colors

from idsprites.infinite_dsprites import (
    InfiniteDSprites,
    Factors,
)
import torch
import numpy as np
from config_dataclass import config_dataclass, config_field, check_initialized
from typing import List, Union
from pathlib import Path
import idsprites as ids
from tqdm import tqdm

generate_shape = ids.InfiniteDSprites().generate_shape
from groupcl.datasets.synthetic import BaseSyntheticDataset


@config_dataclass
class ConfigurableInfiniteDSprites(InfiniteDSprites, BaseSyntheticDataset):

    img_size: int = config_field(default=128)
    num_shapes: int = config_field(default=1)
    num_scales: int = config_field(default=4)
    num_orientations: int = config_field(default=4)
    num_position_x: int = config_field(default=4)
    num_position_y: int = config_field(default=4)

    color_range: List[str] = config_field(default_factory=lambda: ["white"])
    scale_range: List[float] = config_field(default_factory=lambda: [0.5, 1.0])
    orientation_range: List[float] = config_field(
        default_factory=lambda: [0.0, 2 * 3.14159])
    position_x_range: List[float] = config_field(
        default_factory=lambda: [0.0, 1.0])
    position_y_range: List[float] = config_field(
        default_factory=lambda: [0.0, 1.0])

    # we need to change these names, since the InfiniteDSprites.__init__ overwrites them
    shape_orientation_marker: bool = config_field(default=True)
    shape_orientation_color: str = config_field(default="black")
    background: str = config_field(default="gray")
    use_grayscale: bool = config_field(default=True)

    seed: int = config_field(default=876)

    root: Union[str, Path] = "./data"
    force_regenerate: bool = False

    def __lazy_post_init__(self):
        # important to seed before sampling shapes here
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        InfiniteDSprites.__init__(
            self,
            img_size=self.img_size,
            shapes=[generate_shape() for _ in range(self.num_shapes)],
            color_range=self.color_range,
            scale_range=np.linspace(*self.scale_range, self.num_scales),
            orientation_range=np.linspace(
                self.orientation_range[0], self.orientation_range[1] *
                (self.num_orientations / (self.num_orientations + 1)),
                self.num_orientations),
            position_x_range=np.linspace(*self.position_x_range,
                                         self.num_position_x),
            position_y_range=np.linspace(*self.position_y_range,
                                         self.num_position_y),
            orientation_marker=self.shape_orientation_marker,
            orientation_marker_color=self.shape_orientation_color,
            background_color=self.background,
            grayscale=self.use_grayscale,
        )

        assert self.shapes is not None, "shape must not be None"
        self.ranges = {
            "color": self.ranges["color"],
            "shape": self.shapes,
            "scale": self.ranges["scale"],
            "orientation": self.ranges["orientation"],
            "position_x": self.ranges["position_x"],
            "position_y": self.ranges["position_y"],
        }
        self.ranges["color"] = np.array(
            [colors.to_rgb(color) for color in self.ranges["color"]])
        super().__lazy_post_init__()

    @property
    def latent_names(self):
        return list(self.ranges.keys())

    @property
    def latent_sizes(self):
        return torch.tensor([len(v) for v in self.ranges.values()])

    def __iter__(self):
        for latent in self.generate_all_latents():
            yield latent

    def generate_all_latents(self):
        """Generates all possible combinations of factors."""
        # Create meshgrid for all indices
        indices = torch.meshgrid(
            [torch.arange(size) for size in self.latent_sizes], indexing='ij')

        # Stack and reshape to get all combinations
        all_factors = torch.stack([idx.flatten() for idx in indices], dim=1)

        return all_factors

    def latents_to_factors(self, latents: torch.Tensor):
        # latents should be of shape (batch_size, 6) and match the order of the range dict
        if len(latents.shape) == 1:
            latents = latents.unsqueeze(0)
        factors = []
        for latent in tqdm(latents, desc="Converting latents to factors"):
            factor_kwargs = {
                name: values[latent[i]]
                for i, (name, values) in enumerate(self.ranges.items())
            }
            factor_kwargs["shape_id"] = latent[0]
            factors.append(Factors(**factor_kwargs))
        return factors

    def draw(self, latents: torch.Tensor, **kwargs):
        imgs = []
        for factor in tqdm(self.latents_to_factors(latents),
                           desc="Drawing images"):
            img = super().draw(factor, **kwargs)
            # img is float32, wich takes waaaaay to much memory
            # convert to uint8
            img = (img * 255).astype(np.uint8)
            if self.grayscale:
                img = img.squeeze(-3)
            imgs.append(img)

        return np.stack(imgs)

    def _generate_data(self):
        latents = self.generate_all_latents()
        imgs = self.draw(latents)
        imgs = torch.from_numpy(imgs)
        return dict(
            latents=latents,
            imgs=imgs,
        )

    def __len__(self):
        return self._data["latents"].shape[0]

    def _validate_data_types(
        self,
        **kwargs,
    ):
        pass

    @property
    def _data_keys(self):
        return ["latents", "imgs"]
