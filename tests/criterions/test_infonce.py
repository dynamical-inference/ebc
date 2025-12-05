from typing import Tuple

import pytest

from groupcl.criterions.contrastive import infonce, infonce_full_denominator, euclidean_similarity

import torch
from torch import nn
import torch.nn.functional as F


def ref_infonce_full_denominator_simple(
        pos_dist: torch.Tensor,  # nxd
        neg_dist: torch.Tensor,  # nxd
):

    pos_dist = pos_dist
    neg_dist = neg_dist

    pos = 0
    for i in range(pos_dist.shape[0]):
        pos = pos - pos_dist[i]
    pos = pos / pos_dist.shape[0]

    exp_pos = torch.zeros_like(pos_dist)
    for i in range(pos_dist.shape[0]):
        exp_pos[i] = torch.exp(pos_dist[i])

    neg = torch.zeros(pos_dist.shape[0])
    for i in range(neg_dist.shape[0]):
        for j in range(neg_dist.shape[1]):
            neg[i] = neg[i] + torch.exp(neg_dist[i, j])

    denominator = 0
    for i in range(exp_pos.shape[0]):
        denominator += torch.log(exp_pos[i] + neg[i])
    denominator = denominator / exp_pos.shape[0]

    return pos + denominator, pos, denominator


def ref_infonce_full_denominator_not_stable(pos_dist, neg_dist):

    numerator = (-pos_dist).mean()
    denominator = torch.logsumexp(
        torch.concatenate([
            pos_dist.unsqueeze(1),
            neg_dist,
        ], dim=1),
        dim=1,
    ).mean()

    return numerator + denominator, numerator, denominator


def ref_infonce_not_stable(pos_dist: torch.Tensor, neg_dist: torch.Tensor):
    pos_dist = pos_dist
    neg_dist = neg_dist

    align = (-pos_dist).mean()
    uniform = torch.logsumexp(neg_dist, dim=1).mean()
    return align + uniform, align, uniform


def setup_data(num_positives, num_negatives, dim):
    # set seed
    torch.manual_seed(42)
    ref = torch.randn(num_positives, dim).float()
    pos = torch.randn(num_positives, dim).float()
    neg = torch.randn(num_negatives, dim).float()
    return ref, pos, neg


def setup_dist(ref, pos, neg):
    pos_dist, neg_dist = euclidean_similarity(ref, pos, neg)
    pos_dist.requires_grad_(True)
    neg_dist.requires_grad_(True)
    return pos_dist, neg_dist


def _compute_grads(output, inputs):
    for input_ in inputs:
        input_.grad = None
        assert input_.requires_grad
    output.backward()
    return [input_.grad for input_ in inputs]


def assert_loss(
    loss,
    align,
    uniform,
    ref_loss,
    ref_align,
    ref_uniform,
):
    assert torch.allclose(loss, ref_loss)
    assert torch.allclose(align, ref_align)
    assert torch.allclose(uniform, ref_uniform)


def assert_grads(
    grads,
    ref_grads,
):
    for grad, ref_grad in zip(grads, ref_grads):
        assert grad is not None
        assert ref_grad is not None
        assert torch.allclose(grad, ref_grad)


def _test_loss_impl(num_positives, num_negatives, dim, ref_fn, loss_fn):

    ref, pos, neg = setup_data(num_positives, num_negatives, dim)

    pos_dist, neg_dist = setup_dist(ref, pos, neg)
    ref_loss, ref_align, ref_uniform = ref_fn(pos_dist, neg_dist)
    ref_grads = _compute_grads(ref_loss, [pos_dist, neg_dist])

    pos_dist, neg_dist = setup_dist(ref, pos, neg)
    loss, align, uniform = loss_fn(pos_dist, neg_dist)
    grads = _compute_grads(loss, [pos_dist, neg_dist])

    assert_loss(loss, align, uniform, ref_loss, ref_align, ref_uniform)
    assert_grads(grads, ref_grads)


@pytest.mark.parametrize("num_samples", [(100, 100), (10, 1000), (1000, 10)])
@pytest.mark.parametrize("dim", [5, 10])
def test_infonce_full_denominator(num_samples, dim):
    num_positives, num_negatives = num_samples
    _test_loss_impl(
        dim=dim,
        num_positives=num_positives,
        num_negatives=num_negatives,
        ref_fn=ref_infonce_full_denominator_simple,
        loss_fn=infonce_full_denominator,
    )


@pytest.mark.parametrize("num_samples", [(100, 100), (10, 1000), (1000, 10)])
@pytest.mark.parametrize("dim", [5, 10])
def test_stable_infonce(num_samples, dim):
    num_positives, num_negatives = num_samples
    _test_loss_impl(
        dim=dim,
        num_positives=num_positives,
        num_negatives=num_negatives,
        ref_fn=ref_infonce_not_stable,
        loss_fn=infonce,
    )


@pytest.mark.parametrize("num_samples", [(100, 100), (10, 1000), (1000, 10)])
@pytest.mark.parametrize("dim", [5, 10])
def test_stable_infonce_full_denominator(num_samples, dim):
    num_positives, num_negatives = num_samples
    _test_loss_impl(
        dim=dim,
        num_positives=num_positives,
        num_negatives=num_negatives,
        ref_fn=ref_infonce_full_denominator_not_stable,
        loss_fn=infonce_full_denominator,
    )


if __name__ == "__main__":
    test_stable_infonce(num_samples=(100, 100), dim=5)
    test_infonce_full_denominator(num_samples=(100, 100), dim=5)
