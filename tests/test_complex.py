import numpy as np
import pytest
import torch
from scipy.stats import unitary_group

from groupcl.utils.complex import (
    complex_data_to_real,
    complex_matrix_to_real,
    real_data_to_complex,
    real_matrix_to_complex,
    solve_complex_matrix_system,
    solve_real_matrix_system,
)

# Test constants
ATOL = 1e-5
RTOL = 1e-5
STRICT_ATOL = 1e-7
STRICT_RTOL = 1e-7
NUM_SEEDS = 5
DTYPE_REAL = torch.float32
DTYPE_COMPLEX = torch.complex64


def _assert_allclose(a, b, atol=ATOL, rtol=RTOL, err_msg=""):
    """Helper function for tensor comparison with custom error message."""
    assert torch.allclose(
        a, b, atol=atol,
        rtol=rtol), f"{err_msg}: max diff: {torch.max(torch.abs(a - b))}"


def _assert_allclose_strict(a,
                            b,
                            atol=STRICT_ATOL,
                            rtol=STRICT_RTOL,
                            err_msg=""):
    """Helper function for strict tensor comparison."""
    _assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=err_msg)


def _make_random_complex_matrix(n,
                                seed=None,
                                unitary=False,
                                device='cpu',
                                batch_dims=None):
    """Helper function to create random complex matrices for testing."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    shape = (*batch_dims, n, n) if batch_dims else (n, n)

    if unitary:
        if batch_dims:
            raise NotImplementedError(
                "Batched unitary matrix generation not supported in test helper."
            )
        # Create a random unitary matrix using scipy
        U_np = unitary_group.rvs(n, random_state=seed)
        U = torch.from_numpy(U_np).to(DTYPE_COMPLEX).to(device)
        return U
    else:
        # Create a random complex matrix
        real_part = torch.randn(*shape, dtype=DTYPE_REAL, device=device)
        imag_part = torch.randn(*shape, dtype=DTYPE_REAL, device=device)
        return torch.complex(real_part, imag_part)


def _make_random_complex_data(m, n, seed=None, device='cpu', batch_dims=None):
    """Helper function to create random complex data for testing.
    
    Args:
        m: number of samples
        n: number of dimensions
    """
    if seed is not None:
        torch.manual_seed(seed)

    shape = (*batch_dims, m, n) if batch_dims else (m, n)
    real_part = torch.randn(*shape, dtype=DTYPE_REAL, device=device)
    imag_part = torch.randn(*shape, dtype=DTYPE_REAL, device=device)
    return torch.complex(real_part, imag_part)


def _verify_real_matrix_block_structure(R, n):
    """Verify that a real matrix has the correct block structure for complex representation."""
    A = R[..., :n, :n]  # Top-left: real part
    B = R[..., n:, :n]  # Bottom-left: imaginary part
    C = R[..., :n, n:]  # Top-right: should be -imaginary part
    D = R[..., n:, n:]  # Bottom-right: should be real part

    # Check block structure: [[A, -B], [B, A]]
    _assert_allclose(A,
                     D,
                     err_msg="Top-left and bottom-right blocks should be equal")
    _assert_allclose(B,
                     -C,
                     err_msg="Bottom-left should be negative of top-right")


class TestComplexMatrixConversion:
    """Test suite for complex matrix conversion functions."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_matrix_round_trip_conversion(self, n, seed):
        """Test that complex -> real -> complex conversion is identity."""
        C_original = _make_random_complex_matrix(n, seed=seed)

        # Convert to real and back
        R = complex_matrix_to_real(C_original)
        C_recovered = real_matrix_to_complex(R)

        _assert_allclose_strict(C_original,
                                C_recovered,
                                err_msg="Round-trip conversion failed")

    @pytest.mark.parametrize("n", [1, 2, 4])
    @pytest.mark.parametrize("batch_dims", [(2,), (3, 2)])
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_batched_matrix_round_trip_conversion(self, n, batch_dims, seed):
        """Test that batched complex -> real -> complex conversion is identity."""
        C_original = _make_random_complex_matrix(n,
                                                 seed=seed,
                                                 batch_dims=batch_dims)

        # Convert to real and back
        R = complex_matrix_to_real(C_original)
        C_recovered = real_matrix_to_complex(R)

        assert R.shape == (*batch_dims, 2 * n, 2 * n)

        _assert_allclose_strict(
            C_original,
            C_recovered,
            err_msg="Batched round-trip matrix conversion failed")

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_matrix_block_structure(self, n, seed):
        """Test that converted real matrix has correct block structure."""
        C = _make_random_complex_matrix(n, seed=seed)
        R = complex_matrix_to_real(C)

        # Check output shape
        assert R.shape == (2 * n, 2 * n)

        # Verify block structure
        _verify_real_matrix_block_structure(R, n)

    @pytest.mark.parametrize("n", [2, 3, 4])
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_unitary_matrix_properties(self, n, seed):
        """Test that unitary matrices preserve orthogonality after conversion."""
        U_complex = _make_random_complex_matrix(n, seed=seed, unitary=True)
        U_real = complex_matrix_to_real(U_complex)

        # Check that complex matrix is unitary
        UUH = U_complex @ U_complex.conj().T
        I_complex = torch.eye(n, dtype=DTYPE_COMPLEX)
        _assert_allclose(UUH, I_complex, err_msg="Original matrix not unitary")

        # Check that real matrix is orthogonal
        UUT = U_real @ U_real.T
        I_real = torch.eye(2 * n, dtype=DTYPE_REAL)
        _assert_allclose(UUT, I_real, err_msg="Converted matrix not orthogonal")

    def test_matrix_specific_examples(self):
        """Test conversion with specific known examples."""
        # Test identity matrix
        I = torch.eye(2, dtype=DTYPE_COMPLEX)
        I_real = complex_matrix_to_real(I)
        expected_real = torch.eye(4, dtype=DTYPE_REAL)
        _assert_allclose_strict(I_real, expected_real)

        # Test simple complex matrix
        C = torch.tensor([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]],
                         dtype=DTYPE_COMPLEX)
        R = complex_matrix_to_real(C)
        expected = torch.tensor(
            [[1, 3, -2, -4], [5, 7, -6, -8], [2, 4, 1, 3], [6, 8, 5, 7]],
            dtype=DTYPE_REAL)
        _assert_allclose_strict(R, expected)

    @pytest.mark.parametrize("n", [1, 2, 4])
    def test_matrix_edge_cases(self, n):
        """Test edge cases for matrix conversion."""
        # Test zero matrix
        C_zero = torch.zeros(n, n, dtype=DTYPE_COMPLEX)
        R_zero = complex_matrix_to_real(C_zero)
        expected_zero = torch.zeros(2 * n, 2 * n, dtype=DTYPE_REAL)
        _assert_allclose_strict(R_zero, expected_zero)

        # Test purely real matrix
        C_real = torch.randn(n, n, dtype=DTYPE_REAL)
        C_real_complex = C_real.to(DTYPE_COMPLEX)
        R = complex_matrix_to_real(C_real_complex)

        # Should have zero imaginary blocks
        assert torch.allclose(R[:n, n:], torch.zeros(n, n))  # Top-right
        assert torch.allclose(R[n:, :n], torch.zeros(n, n))  # Bottom-left

    def test_matrix_invalid_input(self):
        """Test error handling for invalid inputs."""
        # Odd dimension should raise error for real_matrix_to_complex
        R_odd = torch.randn(3, 3, dtype=DTYPE_REAL)
        with pytest.raises(ValueError,
                           match="Matrix last dimension must be even"):
            real_matrix_to_complex(R_odd)


class TestComplexDataConversion:
    """Test suite for complex data conversion functions."""

    @pytest.mark.parametrize("m", [1, 3, 10, 50])  # num_samples
    @pytest.mark.parametrize("n", [1, 2, 3, 5])  # num_dimensions
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_data_round_trip_conversion(self, m, n, seed):
        """Test that complex -> real -> complex data conversion is identity."""
        D_original = _make_random_complex_data(m, n, seed=seed)

        # Convert to real and back
        R = complex_data_to_real(D_original)
        D_recovered = real_data_to_complex(R)

        _assert_allclose_strict(D_original,
                                D_recovered,
                                err_msg="Data round-trip conversion failed")

    @pytest.mark.parametrize("m", [1, 10])
    @pytest.mark.parametrize("n", [1, 3])
    @pytest.mark.parametrize("batch_dims", [(2,), (4, 3)])
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_batched_data_round_trip_conversion(self, m, n, batch_dims, seed):
        """Test that batched complex -> real -> complex data conversion is identity."""
        D_original = _make_random_complex_data(m,
                                               n,
                                               seed=seed,
                                               batch_dims=batch_dims)

        # Convert to real and back
        R = complex_data_to_real(D_original)
        D_recovered = real_data_to_complex(R)

        assert R.shape == (*batch_dims, m, 2 * n)

        _assert_allclose_strict(
            D_original,
            D_recovered,
            err_msg="Batched data round-trip conversion failed")

    @pytest.mark.parametrize("m", [5, 10])  # num_samples
    @pytest.mark.parametrize("n", [2, 3, 4])  # num_dimensions
    def test_data_shape_consistency(self, m, n):
        """Test that data conversion produces correct shapes."""
        D = _make_random_complex_data(m, n)
        R = complex_data_to_real(D)

        assert R.shape == (m, 2 * n)
        assert R.dtype == DTYPE_REAL

    def test_data_specific_examples(self):
        """Test data conversion with specific known examples."""
        # Data with shape (2 samples, 2 dimensions)
        D = torch.tensor([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]],
                         dtype=DTYPE_COMPLEX)
        R = complex_data_to_real(D)
        expected = torch.tensor(
            [
                [1, 3, 2, 4
                ],  # sample 1: [real_dim1, real_dim2, imag_dim1, imag_dim2]
                [5, 7, 6, 8
                ]  # sample 2: [real_dim1, real_dim2, imag_dim1, imag_dim2]
            ],
            dtype=DTYPE_REAL)
        _assert_allclose_strict(R, expected)

    def test_data_invalid_input(self):
        """Test error handling for invalid data inputs."""
        # Odd dimension should raise error for real_data_to_complex
        R_odd = torch.randn(
            5, 3, dtype=DTYPE_REAL)  # (5 samples, 3 dimensions - odd!)
        with pytest.raises(ValueError,
                           match="Data's last dimension must be even"):
            real_data_to_complex(R_odd)


class TestComplexLinearSystemSolving:
    """Test suite for linear system solving with complex matrices."""

    @pytest.mark.parametrize("n", [2, 3, 4])  # num_dimensions
    @pytest.mark.parametrize("m", [10, 50, 100])  # num_samples
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_complex_matrix_recovery(self, n, m, seed):
        """Test that we can perfectly recover complex matrices from linear systems."""
        # Generate random system: Y = X @ U.T
        U_true = _make_random_complex_matrix(n, seed=seed)
        X = _make_random_complex_data(m, n, seed=seed + 1000)
        Y = X @ U_true.T

        # Solve for the matrix
        U_estimated = solve_complex_matrix_system(X, Y)

        _assert_allclose(U_true,
                         U_estimated,
                         err_msg="Complex matrix recovery failed")

    @pytest.mark.parametrize("n", [2, 3])
    @pytest.mark.parametrize("m", [10, 50])
    @pytest.mark.parametrize("batch_dims", [(2,), (5, 3)])
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_batched_complex_matrix_recovery(self, n, m, batch_dims, seed):
        """Test that we can recover batched complex matrices from linear systems."""
        # Generate random system: Y = X @ U.T
        U_true = _make_random_complex_matrix(n,
                                             seed=seed,
                                             batch_dims=batch_dims)
        X = _make_random_complex_data(m,
                                      n,
                                      seed=seed + 1000,
                                      batch_dims=batch_dims)
        Y = X @ U_true.transpose(-1, -2)

        # Solve for the matrix
        U_estimated = solve_complex_matrix_system(X, Y)

        _assert_allclose(U_true,
                         U_estimated,
                         err_msg="Batched complex matrix recovery failed")

    @pytest.mark.parametrize("n", [2, 3, 4])  # num_dimensions
    @pytest.mark.parametrize("m", [10, 50])  # num_samples
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_real_matrix_recovery(self, n, m, seed):
        """Test that we can perfectly recover real matrices from linear systems."""
        # Generate random system: Y = X @ U.T
        U_true = torch.randn(n, n, dtype=DTYPE_REAL)
        X = torch.randn(m, n, dtype=DTYPE_REAL)
        Y = X @ U_true.T

        # Solve for the matrix
        U_estimated = solve_real_matrix_system(X, Y)

        _assert_allclose(U_true,
                         U_estimated,
                         err_msg="Real matrix recovery failed")

    @pytest.mark.parametrize("n", [2, 3])
    @pytest.mark.parametrize("m", [10, 50])
    @pytest.mark.parametrize("batch_dims", [(2,), (5, 3)])
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_batched_real_matrix_recovery(self, n, m, batch_dims, seed):
        """Test that we can recover batched real matrices from linear systems."""
        # Generate random system: Y = X @ U.T
        U_true = torch.randn(*batch_dims, n, n, dtype=DTYPE_REAL)
        X = torch.randn(*batch_dims, m, n, dtype=DTYPE_REAL)
        Y = X @ U_true.transpose(-1, -2)

        # Solve for the matrix
        U_estimated = solve_real_matrix_system(X, Y)

        _assert_allclose(U_true,
                         U_estimated,
                         err_msg="Batched real matrix recovery failed")

    @pytest.mark.parametrize("n", [2, 3])  # num_dimensions
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_unitary_matrix_recovery(self, n, seed):
        """Test recovery of unitary matrices specifically."""
        # Generate unitary matrix and test data: Y = X @ U.T
        U_true = _make_random_complex_matrix(n, seed=seed, unitary=True)
        X = _make_random_complex_data(100, n, seed=seed + 2000)
        Y = X @ U_true.T

        # Solve for the matrix
        U_estimated = solve_complex_matrix_system(X, Y)

        _assert_allclose(U_true,
                         U_estimated,
                         err_msg="Unitary matrix recovery failed")

    @pytest.mark.parametrize("n", [2, 3])  # num_dimensions
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_conversion_consistency_in_solving(self, n, seed):
        """Test that solving in complex vs real domain gives consistent results."""
        # Generate complex system: Y = X @ U.T
        U_complex_true = _make_random_complex_matrix(n, seed=seed)
        X_complex = _make_random_complex_data(100, n, seed=seed + 3000)
        Y_complex = X_complex @ U_complex_true.T

        # Convert to real representation
        U_real_true = complex_matrix_to_real(U_complex_true)
        X_real = complex_data_to_real(X_complex)
        Y_real = complex_data_to_real(Y_complex)

        # Solve in both domains
        U_complex_estimated = solve_complex_matrix_system(X_complex, Y_complex)
        U_real_estimated = solve_real_matrix_system(X_real, Y_real)

        # Convert real solution back to complex
        U_complex_from_real = real_matrix_to_complex(U_real_estimated)

        # Both approaches should give the same result
        _assert_allclose(U_complex_true,
                         U_complex_estimated,
                         err_msg="Complex domain solving failed")
        _assert_allclose(U_real_true,
                         U_real_estimated,
                         err_msg="Real domain solving failed")
        _assert_allclose(
            U_complex_estimated,
            U_complex_from_real,
            err_msg="Complex and real domain solutions don't match")


class TestComplexAlgebraicProperties:
    """Test suite for verifying algebraic properties of the conversions."""

    @pytest.mark.parametrize("n", [2, 3])
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_matrix_multiplication_consistency(self, n, seed):
        """Test that (AB)_real = A_real @ B_real for complex matrices A, B."""
        # Generate two complex matrices
        A = _make_random_complex_matrix(n, seed=seed)
        B = _make_random_complex_matrix(n, seed=seed + 100)

        # Compute product in complex domain
        AB_complex = A @ B
        AB_real_direct = complex_matrix_to_real(AB_complex)

        # Compute product in real domain
        A_real = complex_matrix_to_real(A)
        B_real = complex_matrix_to_real(B)
        AB_real_indirect = A_real @ B_real

        _assert_allclose(
            AB_real_direct,
            AB_real_indirect,
            err_msg="Matrix multiplication not consistent between domains")

    @pytest.mark.parametrize("n", [2, 3])
    @pytest.mark.parametrize("batch_dims", [(2,), (4, 2)])
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_batched_matrix_multiplication_consistency(self, n, batch_dims,
                                                       seed):
        """Test that (AB)_real = A_real @ B_real for batched complex matrices A, B."""
        # Generate two complex matrices
        A = _make_random_complex_matrix(n, seed=seed, batch_dims=batch_dims)
        B = _make_random_complex_matrix(n,
                                        seed=seed + 100,
                                        batch_dims=batch_dims)

        # Compute product in complex domain
        AB_complex = A @ B
        AB_real_direct = complex_matrix_to_real(AB_complex)

        # Compute product in real domain
        A_real = complex_matrix_to_real(A)
        B_real = complex_matrix_to_real(B)
        AB_real_indirect = A_real @ B_real

        _assert_allclose(
            AB_real_direct,
            AB_real_indirect,
            err_msg=
            "Batched matrix multiplication not consistent between domains")

    @pytest.mark.parametrize("n", [2, 3])  # num_dimensions
    @pytest.mark.parametrize("m", [5, 10])  # num_samples
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_matrix_vector_multiplication_consistency(self, n, m, seed):
        """Test that (Ax)_real = A_real @ x_real for complex matrix A and data x."""
        # Generate complex matrix and data
        A = _make_random_complex_matrix(n, seed=seed)
        x = _make_random_complex_data(m, n, seed=seed + 200)

        # Compute product in complex domain: x @ A.T (since our data is samples-as-rows)
        Ax_complex = x @ A.T
        Ax_real_direct = complex_data_to_real(Ax_complex)

        # Compute product in real domain
        A_real = complex_matrix_to_real(A)
        x_real = complex_data_to_real(x)
        Ax_real_indirect = x_real @ A_real.T

        _assert_allclose(
            Ax_real_direct,
            Ax_real_indirect,
            err_msg="Matrix-vector multiplication not consistent between domains"
        )

    @pytest.mark.parametrize("n", [2, 3])
    @pytest.mark.parametrize("m", [5, 10])
    @pytest.mark.parametrize("batch_dims", [(2,), (4, 2)])
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_batched_matrix_vector_multiplication_consistency(
            self, n, m, batch_dims, seed):
        """Test that (Ax)_real = A_real @ x_real for batched complex matrix A and data x."""
        # Generate complex matrix and data
        A = _make_random_complex_matrix(n, seed=seed, batch_dims=batch_dims)
        x = _make_random_complex_data(m,
                                      n,
                                      seed=seed + 200,
                                      batch_dims=batch_dims)

        # Compute product in complex domain: x @ A.T
        Ax_complex = x @ A.transpose(-1, -2)
        Ax_real_direct = complex_data_to_real(Ax_complex)

        # Compute product in real domain
        A_real = complex_matrix_to_real(A)
        x_real = complex_data_to_real(x)
        Ax_real_indirect = x_real @ A_real.transpose(-1, -2)

        _assert_allclose(
            Ax_real_direct,
            Ax_real_indirect,
            err_msg=
            "Batched matrix-vector multiplication not consistent between domains"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestComplexCUDACompatibility:
    """Test suite for CUDA device compatibility."""

    def test_cuda_device_consistency(self):
        """Test that conversions work correctly on CUDA devices."""
        device = 'cuda'
        n, m = 3, 20

        # Test on CUDA
        C_cuda = _make_random_complex_matrix(n, seed=42, device=device)
        X_cuda = _make_random_complex_data(m, n, seed=43, device=device)

        # Test conversions
        R_cuda = complex_matrix_to_real(C_cuda)
        C_recovered_cuda = real_matrix_to_complex(R_cuda)

        assert C_cuda.device.type == 'cuda'
        assert R_cuda.device.type == 'cuda'
        assert C_recovered_cuda.device.type == 'cuda'

        _assert_allclose_strict(C_cuda,
                                C_recovered_cuda,
                                err_msg="CUDA round-trip conversion failed")

    def test_cuda_solving_consistency(self):
        """Test that matrix solving works on CUDA and gives same results as CPU."""
        n, m = 3, 50

        # Generate on CPU
        U_true_cpu = _make_random_complex_matrix(n, seed=42)
        X_cpu = _make_random_complex_data(m, n, seed=43)
        Y_cpu = X_cpu @ U_true_cpu.T

        # Move to CUDA
        U_true_cuda = U_true_cpu.cuda()
        X_cuda = X_cpu.cuda()
        Y_cuda = Y_cpu.cuda()

        # Solve on both devices
        U_estimated_cpu = solve_complex_matrix_system(X_cpu, Y_cpu)
        U_estimated_cuda = solve_complex_matrix_system(X_cuda, Y_cuda)

        # Results should be equivalent
        _assert_allclose(U_estimated_cpu,
                         U_estimated_cuda.cpu(),
                         err_msg="CUDA and CPU solving results differ")


def test_documentation_examples():
    """Test that examples in docstrings work correctly."""
    # Test complex_matrix_to_real example
    C = torch.tensor([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=DTYPE_COMPLEX)
    R = complex_matrix_to_real(C)
    assert R.shape == torch.Size([4, 4])

    # Test complex_data_to_real example
    D = torch.tensor([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]],
                     dtype=DTYPE_COMPLEX)  # (2 samples, 2 dimensions)
    R_data = complex_data_to_real(D)
    assert R_data.shape == torch.Size([2, 4])  # (2 samples, 4 real dimensions)

    # Test real_data_to_complex example
    D_real = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]],
                          dtype=DTYPE_REAL)  # (2 samples, 4 real dims)
    C_recovered = real_data_to_complex(D_real)
    assert C_recovered.shape == torch.Size([2,
                                            2])  # (2 samples, 2 complex dims)
    expected = torch.tensor([[1. + 5.j, 3. + 7.j], [2. + 6.j, 4. + 8.j]],
                            dtype=DTYPE_COMPLEX)
    _assert_allclose_strict(C_recovered, expected)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
