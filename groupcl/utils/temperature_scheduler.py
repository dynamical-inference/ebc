import abc
import math
from dataclasses import dataclass

from jaxtyping import jaxtyped
from jaxtyping._typeguard import typechecked

from config_dataclass import Configurable, config_dataclass, config_field, check_initialized


@jaxtyped(typechecker=typechecked)
@config_dataclass
class TemperatureScheduler(Configurable, abc.ABC):

    max_epochs: int = config_field(default=100)  # Maximum number of epochs to
    initial_temp: float = config_field(
        default=1.0)  # Starting temperature value
    min_temp: float = config_field(
        default=1e-16)  # Minimum temperature value that can be reached

    def __lazy_post_init__(self):
        super().__lazy_post_init__()
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be greater than 0.")

        # temperatures must be > 0
        if self.initial_temp <= 0:
            raise ValueError("initial_temp must be greater than 0.")
        if self.min_temp <= 0:
            raise ValueError("min_temp must be greater than 0.")
        if self.initial_temp < self.min_temp:
            raise ValueError("initial_temp must be greater than min_temp.")

    @abc.abstractmethod
    def get_temperature(self, epoch: int) -> float:
        pass


@jaxtyped(typechecker=typechecked)
@config_dataclass
class DelayedTemperatureScheduler(TemperatureScheduler):
    delay_epoch: int = config_field(
        default=0
    )  # Number of epochs to wait before starting temperature scheduling

    def __lazy_post_init__(self):
        super().__lazy_post_init__()
        if self.delay_epoch < 0:
            raise ValueError("delay_epoch must be greater or equal to 0.")
        # Stricter check: delay_epoch must be less than max_epochs for decay to happen
        if self.delay_epoch >= self.max_epochs:
            raise ValueError(
                "delay_epoch must be strictly less than max_epochs.")

    @jaxtyped(typechecker=typechecked)
    def get_temperature(self, epoch: int) -> float:
        if epoch < self.delay_epoch:
            temperature = self.initial_temp
        elif epoch >= self.max_epochs:
            temperature = self.min_temp
        else:
            temperature = self._get_temperature(epoch)
        # bound with min and initial temperature
        temperature = max(self.min_temp, temperature)
        temperature = min(self.initial_temp, temperature)
        return temperature

    @abc.abstractmethod
    def _get_temperature(self, epoch: int) -> float:
        pass


@jaxtyped(typechecker=typechecked)
@config_dataclass
class LinearTemperatureScheduler(DelayedTemperatureScheduler):

    def _get_temperature(self, epoch: int) -> float:
        # linear scale means f(x) = ax + b, where x is the epoch, f(x) is the temperature
        # b is the initial temperature, a is the slope

        # slope = (self.initial_temp - self.min_temp)
        slope = (self.initial_temp - self.min_temp)
        # correct slope for reaching min_temp in (max_epochs-delay_epochs) steps
        corrected_slope = slope / (self.max_epochs - self.delay_epoch)

        # correct epoch for delay (correcting x)
        corrected_epoch = epoch - self.delay_epoch

        # calculate temperature
        epoch_temp = self.initial_temp - corrected_slope * corrected_epoch

        return epoch_temp


@jaxtyped(typechecker=typechecked)
@config_dataclass
class ExponentialTemperatureScheduler(DelayedTemperatureScheduler):

    def _get_temperature(self, epoch: int) -> float:
        corrected_epoch = epoch - self.delay_epoch
        total_decay_epochs = self.max_epochs - self.delay_epoch

        # Handle edge case where initial and min temperatures are the same
        if self.initial_temp <= self.min_temp:
            return self.initial_temp

        # Calculate decay rate 'k' such that temp reaches min_temp at max_epochs
        # temp = initial_temp * exp(-k * corrected_epoch)
        # min_temp = initial_temp * exp(-k * total_decay_epochs)
        # k = ln(initial_temp / min_temp) / total_decay_epochs
        # Note: total_decay_epochs > 0 due to check in DelayedTemperatureScheduler.__lazy_post_init__
        decay_rate = math.log(
            self.initial_temp / self.min_temp) / total_decay_epochs

        temperature = self.initial_temp * math.exp(
            -decay_rate * corrected_epoch)
        return temperature


@jaxtyped(typechecker=typechecked)
@config_dataclass
class CyclingTemperatureScheduler(DelayedTemperatureScheduler):
    num_cycles: int = config_field(
        default=1)  # Number of temperature cycles to complete over training

    def __lazy_post_init__(self):
        super().__lazy_post_init__()
        if self.num_cycles <= 0:
            raise ValueError("num_cycles must be greater than 0.")

    def _get_temperature(self, epoch: int) -> float:
        total_decay_epochs = self.max_epochs - self.delay_epoch

        # Handle edge case where cycle length might be zero if num_cycles is large
        if total_decay_epochs == 0:
            return self.initial_temp

        # Ensure cycle length is at least 1
        cycle_length = max(1, total_decay_epochs // self.num_cycles)

        # Corrected epoch for delay
        corrected_epoch = epoch - self.delay_epoch

        cycle_position = corrected_epoch % cycle_length
        cycle_progress = cycle_position / cycle_length  # goes from 0 to (cycle_length-1)/cycle_length

        # Cosine annealing formula: min_temp + 0.5 * (initial_temp - min_temp) * (1 + cos(pi * cycle_progress))
        # This ensures the temperature goes from initial_temp (at progress=0) to min_temp (at progress=1)
        temperature = self.min_temp + 0.5 * (
            self.initial_temp -
            self.min_temp) * (1 + math.cos(math.pi * cycle_progress))
        return temperature


@jaxtyped(typechecker=typechecked)
@config_dataclass
class ConstantTemperatureScheduler(TemperatureScheduler):

    def get_temperature(self, epoch: int) -> float:
        return self.initial_temp


@jaxtyped(typechecker=typechecked)
@config_dataclass
class CyclingExponentialTemperatureScheduler(DelayedTemperatureScheduler):
    num_cycles: int = config_field(
        default=1)  # Number of exponential decay cycles

    def __lazy_post_init__(self):
        super().__lazy_post_init__()
        if self.num_cycles <= 0:
            raise ValueError("num_cycles must be greater than 0.")

    def _get_temperature(self, epoch: int) -> float:
        total_decay_epochs = self.max_epochs - self.delay_epoch

        # Handle edge case where total decay duration is zero
        if total_decay_epochs == 0:
            return self.initial_temp

        # Calculate the duration of a single decay cycle
        # Use floating point division for potentially non-integer cycle lengths
        cycle_length = total_decay_epochs / self.num_cycles

        # Handle edge case where cycle length might be zero or negative (shouldn't happen with checks)
        if cycle_length <= 0:
            return self.initial_temp  # Or perhaps raise an error?

        # Corrected epoch relative to the start of delay
        corrected_epoch = epoch - self.delay_epoch

        # Determine the position within the current cycle
        # Use math.fmod for floating point modulus
        cycle_position = math.fmod(corrected_epoch, cycle_length)

        # Handle edge case where initial and min temperatures are the same or inverted
        if self.initial_temp <= self.min_temp:
            return self.initial_temp

        # Calculate decay rate 'k' for a single cycle duration
        # k = ln(initial_temp / min_temp) / cycle_length
        decay_rate = math.log(self.initial_temp / self.min_temp) / cycle_length

        # Apply exponential decay formula based on position within the current cycle
        temperature = self.initial_temp * math.exp(-decay_rate * cycle_position)

        return temperature
