import pytest

import torch
import numpy as np
from groupcl.utils.bivector import subspace_projection_matrix, are_independent

DTYPE = torch.float64

# def test_are_independent():
#     """Test the are_independent function with various vector sets."""

#     # Test 1: Two independent 2D vectors
#     v1 = torch.tensor([1, 0]).float()
#     v2 = torch.tensor([0, 1]).float()
#     assert are_independent([v1, v2]) == True

#     # Test 2: Two dependent 2D vectors (v2 = 2 * v1)
#     v1 = torch.tensor([1, 2]).float()
#     v2 = torch.tensor([2, 4]).float()
#     assert are_independent([v1, v2]) == False

#     # Test 3: Three independent 3D vectors
#     v1 = torch.tensor([1, 0, 0]).float()
#     v2 = torch.tensor([0, 1, 0]).float()
#     v3 = torch.tensor([0, 0, 1]).float()
#     assert are_independent([v1, v2, v3]) == True

#     # Test 4: Three 3D vectors where one is a sum of the others (dependent)
#     v1 = torch.tensor([1, 0, 0]).float()
#     v2 = torch.tensor([0, 1, 0]).float()
#     v3 = torch.tensor([1, 1, 0]).float()
#     assert are_independent([v1, v2, v3]) == False

#     # Test 6: Single vector (always independent if non-zero)
#     v1 = torch.tensor([3, 4, 5]).float()
#     assert are_independent([v1]) == True

#     # Test batch dimensions
#     batches = 10
#     v1 = torch.tensor([1, 0, 0]).unsqueeze(0).repeat(batches, 1).float()
#     v2 = torch.tensor([0, 1, 0]).unsqueeze(0).repeat(batches, 1).float()
#     v3 = torch.tensor([0, 0, 1]).unsqueeze(0).repeat(batches, 1).float()
#     assert are_independent([v1, v2, v3]) == True

# def _test_projection_matrix(basis1, basis2):
#     # Verify linear independence
#     assert are_independent([basis1, basis2])
#     P = subspace_projection_matrix(basis1, basis2)

#     projection_basis1 = torch.einsum("...ij,...j->...i", P, basis1)
#     projection_basis2 = torch.einsum("...ij,...j->...i", P, basis2)

#     assert torch.allclose(projection_basis1, basis1)
#     assert torch.allclose(projection_basis2, basis2)

#     # sample random vector outside the span
#     tmp_vec = torch.randn_like(basis1)
#     outside_vector = tmp_vec + basis1 + basis2

#     projection_outside_vector = torch.einsum("...ij,...j->...i", P,
#                                              outside_vector)

#     # this vector is now in the span, projecting it again should yield the same vector
#     projection_outside_vector_again = torch.einsum("...ij,...j->...i", P,
#                                                    projection_outside_vector)
#     assert torch.allclose(projection_outside_vector_again,
#                           projection_outside_vector)

# def test_projection_matrix():
#     # Create first basis vector
#     basis1 = torch.randn(10, dtype=DTYPE)
#     basis1 = basis1 / basis1.norm()

#     # Create second basis vector that's linearly independent from the first
#     # by generating a random vector and removing its projection onto basis1
#     temp_vec = torch.randn(10, dtype=DTYPE)
#     basis2 = temp_vec + basis1
#     basis2 = basis2 / basis2.norm()  # Normalize
#     _test_projection_matrix(basis1, basis2)

# def test_projection_matrix_not_normalized():
#     # Create first basis vector
#     basis1 = torch.randn(10, dtype=DTYPE)

#     temp_vec = torch.randn(10, dtype=DTYPE)
#     basis2 = temp_vec + basis1
#     _test_projection_matrix(basis1, basis2)

# def test_projection_matrix_batched():
#     dim = 5
#     batch_size = 10
#     basis1 = torch.randn(batch_size, dim, dtype=DTYPE)
#     temp_vec = torch.randn(batch_size, dim, dtype=DTYPE)
#     basis2 = temp_vec + basis1
#     _test_projection_matrix(basis1, basis2)

# def test_projection_onto_xy_plane():
#     # Example 1: Orthogonal basis in R^3 (projection onto xy-plane)
#     vec_a = torch.tensor([1.0, 0.0, 0.0])
#     vec_b = torch.tensor([0.0, 1.0, 0.0])
#     P_xy = subspace_projection_matrix(vec_a, vec_b)

#     # Test projection
#     test_vec = torch.tensor([3.0, 4.0, 5.0])
#     projected_vec = P_xy @ test_vec

#     # Should be [3., 4., 0.]
#     expected = torch.tensor([3.0, 4.0, 0.0])
#     assert torch.allclose(projected_vec, expected)

# def test_projection_onto_non_orthogonal_basis():
#     # Example 2: Non-orthogonal basis in R^3
#     vec_c = torch.tensor([1.0, 1.0, 0.0])
#     vec_d = torch.tensor([0.0, 1.0, 1.0])
#     P_cd = subspace_projection_matrix(vec_c, vec_d)

#     # Test projection (vector orthogonal to the plane spanned by c, d is [1, -1, 1])
#     test_vec_ortho = torch.tensor([1.0, -1.0, 1.0])
#     projected_ortho = P_cd @ test_vec_ortho

#     # Should be close to [0., 0., 0.]
#     assert torch.allclose(projected_ortho,
#                           torch.zeros_like(projected_ortho),
#                           atol=1e-6)

#     # Test projection (vector within the plane, e.g., c+d)
#     test_vec_in_plane = vec_c + vec_d  # [1., 2., 1.]
#     projected_in_plane = P_cd @ test_vec_in_plane

#     # Should be close to [1., 2., 1.]
#     assert torch.allclose(projected_in_plane, test_vec_in_plane)

# def test_linearly_dependent_vectors():
#     # Example 3: Linearly dependent vectors (should raise error)
#     vec_e = torch.tensor([1.0, 2.0, 3.0])
#     vec_f = torch.tensor([2.0, 4.0, 6.0])  # vec_f = 2 * vec_e

#     assert not are_independent([vec_e, vec_f
#                                ]), "Vectors should be linearly dependent"

#     with pytest.raises(ValueError):
#         subspace_projection_matrix(vec_e, vec_f)

if __name__ == "__main__":
    pytest.main([__file__])
    # test_projection_matrix_batched()
