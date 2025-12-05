from typing import List, Optional, Tuple, Union

import numpy as np
from jaxtyping._typeguard import typechecked

import torch
from jaxtyping import Float, Integer, jaxtyped


@jaxtyped(typechecker=typechecked)
def paired_differences_offset(
    auxiliary_data: Float[torch.Tensor, "num_samples"],
    offset: int = 1,
) -> Tuple[
    Float[torch.Tensor, "num_samples-{offset}"],
    Integer[torch.Tensor, "num_samples-{offset}"],
    Integer[torch.Tensor, "num_samples-{offset}"],
]:
    """
    Extract consecutive pairs of indices with a single offset for analysis.

    This function takes auxiliary data and returns consecutive pairs of indices
    separated by the specified offset. This is useful for analyzing temporal
    relationships in time series data.

    Args:
        auxiliary_data: 1D auxiliary data with shape (num_samples,)
        offset: Number of time steps between paired samples

    Returns:
        Tuple containing:
        - aux_data_diff: Difference in auxiliary data (auxiliary_data[i+offset] - auxiliary_data[i])
        - x_index: Indices for time i+offset
        - x_prime_index: Indices for time i

    Each output tensor has shape (num_samples - offset, ...) where ... depends on
    the dimensionality of the input.
    """
    num_samples = auxiliary_data.shape[0]
    index = torch.arange(num_samples)

    aux_data_1 = auxiliary_data[offset:]
    aux_data_2 = auxiliary_data[:-offset]
    aux_data_diff = aux_data_1 - aux_data_2
    x_index = index[offset:]
    x_prime_index = index[:-offset]

    return (
        aux_data_diff,
        x_index,
        x_prime_index,
    )


@jaxtyped(typechecker=typechecked)
def paired_differences(
    auxiliary_data: Float[torch.Tensor, "num_samples "],
    offsets: list[int] = [1],
) -> Tuple[
    Float[torch.Tensor, "total_pairs"],
    Integer[torch.Tensor, "total_pairs"],
    Integer[torch.Tensor, "total_pairs"],
    Integer[torch.Tensor, "total_pairs"],
]:
    """
    Extract consecutive pairs of indices with multiple offsets for analysis.

    This function takes auxiliary data and returns consecutive pairs of indices
    separated by the specified offsets. This is useful for analyzing temporal
    relationships in time series data at multiple time scales.

    Args:
        auxiliary_data: 1D auxiliary data with shape (num_samples,)
        offsets: List of time step offsets between paired samples (default: [1])

    Returns:
        Tuple containing:
        - aux_data_diff: Difference in auxiliary data (auxiliary_data[i+offset] - auxiliary_data[i])
        - x_index: Indices for time i+offset
        - x_prime_index: Indices for time i
        - time_offsets: Tensor indicating the time offset for each pair

    Each output tensor has shape (total_pairs, ...) where total_pairs is the sum
    of (num_samples - offset) for each offset, and ... depends on the dimensionality
    of the input.
    """
    all_aux_data_diff = []
    all_x_index = []
    all_x_prime_index = []
    all_time_offsets = []

    for offset in offsets:
        if offset >= auxiliary_data.shape[0]:
            continue
        (
            aux_data_diff,
            x_index,
            x_prime_index,
        ) = paired_differences_offset(
            auxiliary_data=auxiliary_data,
            offset=offset,
        )

        # Create time offset tensor for this offset
        time_offset_tensor = torch.full(
            (aux_data_diff.shape[0],), offset, dtype=torch.int64
        )

        all_aux_data_diff.append(aux_data_diff)
        all_x_index.append(x_index)
        all_x_prime_index.append(x_prime_index)
        all_time_offsets.append(time_offset_tensor)

    # Concatenate all results
    return (
        torch.cat(all_aux_data_diff, dim=0),
        torch.cat(all_x_index, dim=0),
        torch.cat(all_x_prime_index, dim=0),
        torch.cat(all_time_offsets, dim=0),
    )


def create_symlog_bins(
    data: np.ndarray, n_bins_per_side: int, alpha: float = 1.0
) -> np.ndarray:
    """
    Creates `2 * n_bins_per_side` total bins with a data-driven central bin.

    This version avoids a fixed zero-edge. To produce `n_bins + 1` total edges,
    it generates n_bins_per_side + 1 edges on the positive side and
    n_bins_per_side edges on the negative side. This results in a total of
    n_bins bins, but with a slight asymmetry in their count on each side.

    Args:
        data: The input data array.
        n_bins_per_side: Half the desired total number of bins.
        alpha: Controls the strength of the log scaling.

    Returns:
        An array of bin edges of length `(2 * n_bins_per_side) + 1`.
    """
    # Separate positive and negative non-zero data
    pos_data = data[data > 0]
    neg_data = data[data < 0]

    # Fallback to linear if data is one-sided
    if len(pos_data) == 0 or len(neg_data) == 0:
        total_bins = 2 * n_bins_per_side
        return np.linspace(data.min(), data.max(), total_bins + 1)

    # POSITIVE BINS: Generate n_bins_per_side + 1 edges
    min_pos, max_pos = pos_data.min(), pos_data.max()
    start_asinh = np.arcsinh(min_pos / alpha)
    stop_asinh = np.arcsinh(max_pos / alpha)
    # Generate k+1 edges to define k bins on the positive side
    asinh_pos_edges = np.linspace(start_asinh, stop_asinh, n_bins_per_side + 1)
    pos_edges = alpha * np.sinh(asinh_pos_edges)

    # NEGATIVE BINS: Generate n_bins_per_side edges
    min_neg_abs, max_neg_abs = -neg_data.max(), -neg_data.min()
    start_asinh = np.arcsinh(min_neg_abs / alpha)
    stop_asinh = np.arcsinh(max_neg_abs / alpha)
    # Generate k edges for the negative side
    asinh_neg_edges = np.linspace(start_asinh, stop_asinh, n_bins_per_side)
    neg_edges = -alpha * np.sinh(asinh_neg_edges)

    # Combine to get a total of (k) + (k+1) = 2k+1 edges
    return np.concatenate([neg_edges[::-1], pos_edges])


@jaxtyped(typechecker=typechecked)
def bin_differences(
    n_bins: int,
    aux_data_diff: Float[torch.Tensor, "num_samples"],
    symlog_scale_alpha: Optional[float] = None,
) -> Tuple[
    Integer[torch.Tensor, "num_samples"],
    Integer[torch.Tensor, "n_bins"],
    Float[torch.Tensor, "n_bins+1"],
]:
    """
    Bin auxiliary data differences into discrete bins.

    This function takes differences in auxiliary data and bins them into a specified
    number of bins using linear spacing between the minimum and maximum values.
    It handles edge cases by clipping values to valid bin ranges.

    Args:
        n_bins: Number of bins to create
        aux_data_diff: 1D tensor of auxiliary data differences to be binned

    Returns:
        Tuple containing:
        - bin_indices: Tensor of bin assignments for each sample (0 to n_bins-1)
        - bin_counts: Tensor with count of samples in each bin
        - bin_edges: Tensor of bin edge values (length n_bins+1)

    Note:
        The function clips bin indices to ensure all values fall within [0, n_bins-1].
        Samples exactly equal to the maximum value are assigned to the last bin.
    """

    if symlog_scale_alpha is not None:
        # check that n_bins is even
        assert n_bins % 2 == 0, "n_bins must be even for symlog scale"
        n_bins_per_side = n_bins // 2
        bin_edges = create_symlog_bins(
            aux_data_diff, n_bins_per_side=n_bins_per_side, alpha=symlog_scale_alpha
        )
    else:
        bin_edges = np.linspace(aux_data_diff.min(), aux_data_diff.max(), n_bins + 1)
    bin_indices = np.digitize(aux_data_diff, bin_edges)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)

    # there should be zero bins assigned to the frist bin
    # because that bin is for values outside smaller than the smallest bin edge
    # we allow for max 1 sample in the first bin to account for numeric errors
    assert bin_counts[0] <= 1, "There should be at most 1 sample in the first bin"
    # there may be multiple bins assigned to the last bin, which would be for those as large as aux_data_diff.max()
    # that should at least be one, usually exactly one. If more, show warning
    if bin_counts[-1] > 1:
        print(f"Warning: {bin_counts[-1]} bins assigned to the last bin")

    # let's drop the the first and last bin
    bin_indices = np.clip(bin_indices - 1, 0, n_bins - 1)

    bin_counts = np.bincount(bin_indices, minlength=n_bins)

    return (
        torch.from_numpy(bin_indices),
        torch.from_numpy(bin_counts),
        torch.from_numpy(bin_edges),
    )


# @jaxtyped(typechecker=typechecked)
def filter_binned_differences(
    auxiliary_diff: Float[torch.Tensor, " num_samples"],
    x_index: Integer[torch.Tensor, " num_samples"],
    x_prime_index: Integer[torch.Tensor, " num_samples"],
    time_offsets: Integer[torch.Tensor, " num_samples"],
    actions_idx: Integer[torch.Tensor, " num_samples"],
    actions_counts: Integer[torch.Tensor, " num_bins"],
    actions_bin_edges: Float[torch.Tensor, " num_bins+1"],
    min_samples_per_action: Optional[int] = None,
    min_abs_difference: Optional[float] = None,
    max_abs_difference: Optional[float] = None,
) -> Tuple[
    Float[torch.Tensor, " reduced_num_samples"],
    Integer[torch.Tensor, " reduced_num_samples"],
    Integer[torch.Tensor, " reduced_num_samples"],
    Integer[torch.Tensor, " reduced_num_samples"],
    Integer[torch.Tensor, " reduced_num_samples"],
    Integer[torch.Tensor, " reduced_num_bins"],
    Float[torch.Tensor, " reduced_num_bins+1"],
]:
    """
    Filter binned differences based on thresholds and minimum sample requirements.

    Args:
        auxiliary_diff: 1D tensor of auxiliary data differences
        x_index: 1D tensor of x indices
        x_prime_index: 1D tensor of x' indices
        time_offsets: 1D tensor of time offsets
        actions_idx: 1D tensor of action/bin indices
        actions_counts: 1D tensor with count of samples in each bin
        actions_bin_edges: 1D tensor of bin edge values
        min_samples_per_action: Minimum number of samples required per action/bin
        min_abs_difference: Minimum absolute auxiliary difference threshold
        max_abs_difference: Maximum absolute auxiliary difference threshold

    Returns:
        Tuple of filtered tensors with reduced dimensions
    """

    # Start with all actions being valid
    valid_actions_mask = torch.ones(len(actions_counts), dtype=torch.bool)

    # Filter based on min/max difference thresholds if provided
    if min_abs_difference is not None or max_abs_difference is not None:
        # Find bin centers to determine which bins to filter
        bin_centers = (actions_bin_edges[:-1] + actions_bin_edges[1:]) / 2
        abs_bin_centers = torch.abs(bin_centers)

        if min_abs_difference is not None:
            valid_actions_mask &= abs_bin_centers >= min_abs_difference

        if max_abs_difference is not None:
            valid_actions_mask &= abs_bin_centers <= max_abs_difference

    # Filter based on minimum samples per action if provided
    if min_samples_per_action is not None:
        valid_actions_mask &= actions_counts >= min_samples_per_action

    # Get the valid action indices
    valid_action_ids = torch.where(valid_actions_mask)[0]

    # Create mask for samples that belong to valid actions
    sample_mask = torch.isin(actions_idx, valid_action_ids)

    # Filter all sample-level data
    filtered_auxiliary_diff = auxiliary_diff[sample_mask]
    filtered_x_index = x_index[sample_mask]
    filtered_x_prime_index = x_prime_index[sample_mask]
    filtered_time_offsets = time_offsets[sample_mask]
    filtered_actions_idx = actions_idx[sample_mask]

    # Filter action-level data
    filtered_actions_counts = actions_counts[valid_actions_mask]

    # Filter bin edges to include only edges of valid bins
    if valid_actions_mask.any():
        # Collect all edges for valid bins
        filtered_actions_bin_edges = actions_bin_edges[valid_action_ids]
        # the edges expect the right border of the last bin as well
        filtered_actions_bin_edges = torch.cat(
            [
                filtered_actions_bin_edges,
                actions_bin_edges[valid_action_ids.max() + 1].unsqueeze(0),
            ]
        )
    else:
        # If no valid actions, return empty edge tensor
        filtered_actions_bin_edges = torch.empty(0, dtype=actions_bin_edges.dtype)

    # Remap action indices to be contiguous starting from 0
    if len(valid_action_ids) > 0:
        # Create mapping from old action indices to new contiguous indices
        action_mapping = torch.full((actions_idx.max() + 1,), -1, dtype=torch.long)
        action_mapping[valid_action_ids] = torch.arange(len(valid_action_ids))
        filtered_actions_idx = action_mapping[filtered_actions_idx]

    return (
        filtered_auxiliary_diff,
        filtered_x_index,
        filtered_x_prime_index,
        filtered_time_offsets,
        filtered_actions_idx,
        filtered_actions_counts,
        filtered_actions_bin_edges,
    )


def determine_variable_types(tensor):
    """
    Determine if each column/dimension of a tensor contains continuous or discrete variables.

    Args:
        tensor: torch.Tensor of shape (num_samples,) or (num_samples, dim)

    Returns:
        list: List of strings, either 'continuous' or 'discrete' for each dimension
    """
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(1)  # Make it 2D for uniform processing

    variable_types = []

    for col_idx in range(tensor.shape[1]):
        col_data = tensor[:, col_idx]

        # Remove any NaN values for analysis
        valid_data = col_data[~torch.isnan(col_data)]

        if len(valid_data) == 0:
            variable_types.append("unknown")
            continue

        # Check if all values are approximately integers
        is_integer_like = torch.allclose(valid_data, torch.round(valid_data), atol=1e-6)

        if is_integer_like:
            variable_types.append("discrete")
        else:
            variable_types.append("continuous")

    continous_mask = torch.tensor([x == "continuous" for x in variable_types])
    discrete_mask = torch.tensor([x == "discrete" for x in variable_types])
    return variable_types, continous_mask, discrete_mask


def _discretize_variable(
    variable: torch.Tensor,
    bin_size: Optional[float] = None,
    eps: float = 1e-6,
    return_boundaries: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Discretize a continuous variable into a categorical variable.

    Args:
        variable: torch.Tensor of shape (num_samples,)
        bin_size: float, size of the bins to discretize the variable into
        return_boundaries: bool, whether to return the boundaries used for discretization

    Returns:
        torch.Tensor of shape (num_samples,) or (num_samples, dim)
        If return_boundaries is True, returns tuple of (discretized, boundaries)
    """
    if bin_size is None:
        assert determine_variable_types(variable)[0][0] == "discrete", (
            "If bin_size is not provided, the variable must be discrete"
        )
        # Determine appropriate dtype based on max value
        max_val = variable.max().item()
        if max_val <= 127:
            dtype = torch.int8
        elif max_val <= 32767:
            dtype = torch.int16
        else:
            dtype = torch.int32
        discretized = variable.to(dtype)
        if return_boundaries:
            return discretized, discretized
        return discretized

    boundaries = torch.arange(
        variable.min(), variable.max() + eps, bin_size, device=variable.device
    )
    discretized = torch.bucketize(variable, boundaries, right=True, out_int32=True)
    discretized -= 1

    # Determine appropriate dtype based on number of bins
    max_bin_idx = len(boundaries) - 1
    if max_bin_idx <= 127:
        dtype = torch.int8
    elif max_bin_idx <= 32767:
        dtype = torch.int16
    else:
        dtype = torch.int32

    discretized = discretized.to(dtype)

    if return_boundaries:
        return discretized, boundaries

    return discretized


def discretize_variable(
    variable: torch.Tensor,
    bin_size: Union[Optional[float], List[Optional[float]]] = None,
    return_boundaries: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]]]:
    if variable.dim() == 1:
        return _discretize_variable(
            variable, bin_size, return_boundaries=return_boundaries
        )
    else:
        assert isinstance(bin_size, list), (
            "bin_size must be a list if the variable is multidimensional"
        )
        assert len(bin_size) == variable.shape[-1], (
            "bin_size must have the same length as the number of dimensions of the variable"
        )
        discretized_vars = []
        boundaries_list = []
        for i in range(variable.shape[-1]):
            disc, bounds = _discretize_variable(
                variable[:, i], bin_size[i], return_boundaries=True
            )
            discretized_vars.append(disc)
            boundaries_list.append(bounds)
        discretized = torch.stack(discretized_vars, dim=-1)
        if return_boundaries:
            return discretized, boundaries_list
        return discretized


def generate_group_factors(
    auxilary_variable: Integer[torch.Tensor, " num_samples *dim"],
    index_pairs: Integer[torch.Tensor, " num_pairs 2"],
    equivariant_dims: Optional[List[int]] = None,
) -> Tuple[
    Integer[torch.Tensor, " num_factors "],
    Integer[torch.Tensor, " num_factors 2"],
]:
    if equivariant_dims is None:
        equivariant_dims = list(range(auxilary_variable.shape[-1]))

    equivariant_col_mask = torch.tensor(
        [i in equivariant_dims for i in range(auxilary_variable.shape[-1])],
        device=auxilary_variable.device,
    )

    invariant_col_mask = ~equivariant_col_mask

    aux_pairs = auxilary_variable[index_pairs, ...]

    aux_differences = aux_pairs[:, 0, :] - aux_pairs[:, 1, :]
    aux_equal = aux_pairs[:, 0, :] == aux_pairs[:, 1, :]
    aux_value = aux_pairs[:, 0, :]

    group_factors = torch.zeros_like(aux_value)

    # group factors specifying equivariant pairs are determined by the difference in the auxiliary variable
    group_factors[:, equivariant_col_mask] = aux_differences[:, equivariant_col_mask]

    # group factors specifying invariant pairs are determined by the value
    group_factors[:, invariant_col_mask] = aux_value[:, invariant_col_mask]
    # but only for those pairs where the value is equal between the two samples of a pair
    # if the value is not the same, we don't want the pairs
    equal_mask = aux_equal[:, invariant_col_mask].all(dim=-1)
    group_factors = group_factors[equal_mask]
    filtered_index_pairs = index_pairs[equal_mask]

    return group_factors, filtered_index_pairs


def generate_group_factors_v2(
    auxilary_variable: Integer[torch.Tensor, " num_samples *dim"],
    index_pairs: Integer[torch.Tensor, " num_pairs 2"],
    group_action_dims: Optional[List[int]] = None,
    content_dims: Optional[List[int]] = None,
) -> Tuple[
    Integer[torch.Tensor, " num_factors "],
    Integer[torch.Tensor, " num_factors 2"],
]:
    if group_action_dims is None:
        group_action_dims = list(range(auxilary_variable.shape[-1]))
    if content_dims is None:
        content_dims = list()

    group_action_col_mask = torch.tensor(
        [i in group_action_dims for i in range(auxilary_variable.shape[-1])],
        device=auxilary_variable.device,
    )
    content_col_mask = torch.tensor(
        [i in content_dims for i in range(auxilary_variable.shape[-1])],
        device=auxilary_variable.device,
    )

    aux_pairs = auxilary_variable[index_pairs, ...]

    aux_differences = aux_pairs[:, 0, :] - aux_pairs[:, 1, :]
    aux_equal = aux_pairs[:, 0, :] == aux_pairs[:, 1, :]

    # group factors specify the identity of a group action
    # pairs with the same group factors are of the same group action
    # group action factors are determined by the change in the auxiliary variable
    group_factors = aux_differences[:, group_action_col_mask]

    # content factors may be different between pairs of the same group action
    # BUT content factors must be the same within a given sample pair
    # we filter out all pairs where the content factors are different
    equal_mask = aux_equal[:, content_col_mask].all(dim=-1)
    filtered_index_pairs = index_pairs[equal_mask]
    group_factors = group_factors[equal_mask]

    return group_factors, filtered_index_pairs
