from typing import Tuple

import pytest

from groupcl import criterions
import torch
from torch import nn
import torch.nn.functional as F


def setup_data(num_positives, num_negatives, dim):
    # set seed
    torch.manual_seed(42)
    ref = torch.randn(num_positives, dim).float()
    pos = torch.randn(num_positives, dim).float()
    neg = torch.randn(num_negatives, dim).float()
    ref.requires_grad_(True)
    pos.requires_grad_(True)
    neg.requires_grad_(True)
    return ref, pos, neg


def _compute_grads(output, inputs):
    for input_ in inputs:
        input_.grad = None
        assert input_.requires_grad
    output.backward()
    return [input_.grad for input_ in inputs]


def assert_grads(
    grads,
    ref_grads,
):
    for grad, ref_grad in zip(grads, ref_grads):
        assert grad is not None
        assert ref_grad is not None
        assert torch.allclose(grad, ref_grad)


@pytest.mark.parametrize("num_positives", [100, 500])
@pytest.mark.parametrize("num_negatives", [100, 500])
@pytest.mark.parametrize("dim", [5])
def test_dot_similarity_dual_ref(num_positives, num_negatives, dim):
    # input data
    ref, pos, neg = setup_data(num_positives, num_negatives, dim)
    ref_1, pos_1, neg_1 = ref.detach().clone(), pos.detach().clone(
    ), neg.detach().clone()
    ref_1.requires_grad_(True)
    pos_1.requires_grad_(True)
    neg_1.requires_grad_(True)

    pos_dist, neg_dist = criterions.dot_similarity_dual_ref(
        ref_pos=ref,
        ref_neg=ref,
        pos=pos,
        neg=neg,
    )
    pos_dist = pos_dist.mean()
    neg_dist = neg_dist.mean()
    combined_dist = (pos_dist + neg_dist) / 2
    grads_combined_dist = _compute_grads(combined_dist, [ref, pos, neg])

    pos_dist_1, neg_dist_1 = criterions.dot_similarity(
        ref=ref_1,
        pos=pos_1,
        neg=neg_1,
    )
    pos_dist_1 = pos_dist_1.mean()
    neg_dist_1 = neg_dist_1.mean()
    combined_dist_1 = (pos_dist_1 + neg_dist_1) / 2
    grads_combined_dist_1 = _compute_grads(combined_dist_1,
                                           [ref_1, pos_1, neg_1])

    assert torch.allclose(combined_dist, combined_dist_1)
    assert_grads(grads_combined_dist, grads_combined_dist_1)


@pytest.mark.parametrize("num_positives", [100, 500])
@pytest.mark.parametrize("num_negatives", [100, 500])
@pytest.mark.parametrize("dim", [5])
def test_euclidean_similarity_dual_ref(num_positives, num_negatives, dim):
    ref, pos, neg = setup_data(num_positives, num_negatives, dim)
    ref_1, pos_1, neg_1 = ref.detach().clone(), pos.detach().clone(
    ), neg.detach().clone()
    ref_1.requires_grad_(True)
    pos_1.requires_grad_(True)
    neg_1.requires_grad_(True)

    pos_dist, neg_dist = criterions.euclidean_similarity_dual_ref(
        ref_pos=ref,
        ref_neg=ref,
        pos=pos,
        neg=neg,
    )
    pos_dist = pos_dist.mean()
    neg_dist = neg_dist.mean()
    combined_dist = (pos_dist + neg_dist) / 2
    grads_combined_dist = _compute_grads(combined_dist, [ref, pos, neg])

    pos_dist_1, neg_dist_1 = criterions.euclidean_similarity(
        ref=ref_1,
        pos=pos_1,
        neg=neg_1,
    )
    pos_dist_1 = pos_dist_1.mean()
    neg_dist_1 = neg_dist_1.mean()
    combined_dist_1 = (pos_dist_1 + neg_dist_1) / 2
    grads_combined_dist_1 = _compute_grads(combined_dist_1,
                                           [ref_1, pos_1, neg_1])

    assert torch.allclose(combined_dist, combined_dist_1)
    assert_grads(grads_combined_dist, grads_combined_dist_1)


if __name__ == "__main__":
    test_dot_similarity_dual_ref(num_positives=100, num_negatives=100, dim=5)
    test_euclidean_similarity_dual_ref(num_positives=100,
                                       num_negatives=100,
                                       dim=5)
