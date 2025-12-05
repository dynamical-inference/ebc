import pytest
import torch
import numpy as np
from scipy.stats import unitary_group

from groupcl.models.groups import LeastSquaresSolver, ComplexLeastSquaresSolver
from groupcl.utils.complex import complex_matrix_to_real, complex_data_to_real

# Test constants
ATOL = 1e-4
RTOL = 1e-4
STRICT_ATOL = 1e-6
STRICT_RTOL = 1e-6
NUM_SEEDS = 3
DTYPE_REAL = torch.float32
DTYPE_COMPLEX = torch.complex64


def _assert_allclose(a, b, atol=ATOL, rtol=RTOL, err_msg=""):
    """Helper function for tensor comparison with custom error message."""
    assert torch.allclose(
        a, b, atol=atol, rtol=rtol
    ), f"{err_msg}: max diff: {torch.max(torch.abs(a - b))}"


def _make_random_complex_transformation_data(num_samples, complex_dim, seed=None):
    """Create test data for complex transformations."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate a random complex transformation matrix
    real_part = torch.randn(complex_dim, complex_dim, dtype=DTYPE_REAL)
    imag_part = torch.randn(complex_dim, complex_dim, dtype=DTYPE_REAL)
    U_complex = torch.complex(real_part, imag_part)
    
    # Generate random complex input data
    X_real_part = torch.randn(num_samples, complex_dim, dtype=DTYPE_REAL)
    X_imag_part = torch.randn(num_samples, complex_dim, dtype=DTYPE_REAL)
    X_complex = torch.complex(X_real_part, X_imag_part)
    
    # Apply transformation: X_prime = X @ U.T
    X_prime_complex = X_complex @ U_complex.T
    
    # Convert to real representation for solver input
    X_real = complex_data_to_real(X_complex)  # (num_samples, 2*complex_dim)
    X_prime_real = complex_data_to_real(X_prime_complex)  # (num_samples, 2*complex_dim)
    U_real = complex_matrix_to_real(U_complex)  # (2*complex_dim, 2*complex_dim)
    
    return X_real, X_prime_real, U_real


class TestComplexLeastSquaresSolver:
    """Test suite for ComplexLeastSquaresSolver."""
    
    def test_basic_instantiation(self):
        """Test that ComplexLeastSquaresSolver can be instantiated."""
        solver = ComplexLeastSquaresSolver()
        assert solver.orthogonal_solution is False
        assert isinstance(solver, LeastSquaresSolver)
    
    @pytest.mark.parametrize("complex_dim", [1, 2, 3, 4])
    @pytest.mark.parametrize("num_samples", [10, 50])
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_single_system_solving(self, complex_dim, num_samples, seed):
        """Test that ComplexLeastSquaresSolver can solve single systems accurately."""
        solver = ComplexLeastSquaresSolver()
        
        # Generate test data
        X, X_prime, U_true = _make_random_complex_transformation_data(
            num_samples, complex_dim, seed=seed
        )
        
        # Solve for the transformation matrix
        U_estimated = solver._solve_system(X, X_prime)
        
        # Check that we recover the original transformation matrix
        _assert_allclose(
            U_true, U_estimated,
            err_msg=f"Complex solver failed to recover transformation matrix (dim={complex_dim}, samples={num_samples})"
        )
        
        # Verify that the solution actually works
        X_prime_predicted = X @ U_estimated.T
        _assert_allclose(
            X_prime, X_prime_predicted,
            err_msg="Predicted transformation doesn't match original"
        )
    
    @pytest.mark.parametrize("complex_dim", [2, 3])
    @pytest.mark.parametrize("num_systems", [2, 5])
    @pytest.mark.parametrize("num_samples", [20, 50])
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_multiple_systems_solving(self, complex_dim, num_systems, num_samples, seed):
        """Test that ComplexLeastSquaresSolver can solve multiple systems."""
        solver = ComplexLeastSquaresSolver()
        
        # Generate test data for multiple systems
        torch.manual_seed(seed)
        X_batch = []
        X_prime_batch = []
        U_true_batch = []
        
        for i in range(num_systems):
            X, X_prime, U_true = _make_random_complex_transformation_data(
                num_samples, complex_dim, seed=seed + i * 1000
            )
            X_batch.append(X)
            X_prime_batch.append(X_prime)
            U_true_batch.append(U_true)
        
        X_batch = torch.stack(X_batch, dim=0)  # (num_systems, num_samples, 2*complex_dim)
        X_prime_batch = torch.stack(X_prime_batch, dim=0)
        U_true_batch = torch.stack(U_true_batch, dim=0)
        
        # Solve for all transformation matrices
        U_estimated_batch = solver._solve_systems(X_batch, X_prime_batch)
        
        # Check that we recover all transformation matrices
        _assert_allclose(
            U_true_batch, U_estimated_batch,
            err_msg=f"Complex solver failed to recover multiple transformation matrices"
        )
    
    def test_odd_dimension_error(self):
        """Test that odd dimensions raise appropriate errors."""
        solver = ComplexLeastSquaresSolver()
        
        # Create data with odd dimension (should fail)
        X_odd = torch.randn(10, 3, dtype=DTYPE_REAL)  # 3 is odd
        X_prime_odd = torch.randn(10, 3, dtype=DTYPE_REAL)
        
        with pytest.raises(ValueError, match="must be even for complex conversion"):
            solver._solve_system(X_odd, X_prime_odd)
        
        # Test with batch data
        X_odd_batch = torch.randn(2, 10, 3, dtype=DTYPE_REAL)
        X_prime_odd_batch = torch.randn(2, 10, 3, dtype=DTYPE_REAL)
        
        with pytest.raises(ValueError, match="must be even for complex conversion"):
            solver._solve_systems(X_odd_batch, X_prime_odd_batch)
    
    @pytest.mark.parametrize("complex_dim", [2, 3])
    @pytest.mark.parametrize("num_samples", [20, 50])
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_consistency_with_regular_solver(self, complex_dim, num_samples, seed):
        """Test that ComplexLeastSquaresSolver gives reasonable results compared to regular solver."""
        complex_solver = ComplexLeastSquaresSolver()
        real_solver = LeastSquaresSolver(orthogonal_solution=False)
        
        # Generate test data
        X, X_prime, U_true = _make_random_complex_transformation_data(
            num_samples, complex_dim, seed=seed
        )
        
        # Solve with both solvers
        U_complex_estimated = complex_solver._solve_system(X, X_prime)
        U_real_estimated = real_solver._solve_system(X, X_prime)
        
        # Both should solve the same linear system, but the complex approach 
        # might find a different (but equally valid) solution
        # So let's verify both solutions work equally well
        X_prime_complex_pred = X @ U_complex_estimated.T
        X_prime_real_pred = X @ U_real_estimated.T
        
        # Both predictions should be close to the target
        complex_error = torch.norm(X_prime - X_prime_complex_pred)
        real_error = torch.norm(X_prime - X_prime_real_pred)
        
        # Complex solver should be at least as good as real solver
        assert complex_error <= real_error + 1e-3, f"Complex solver error ({complex_error}) significantly worse than real solver error ({real_error})"
    
    def test_solve_method_integration(self):
        """Test that the solver integrates properly with the SystemSolver.solve method."""
        solver = ComplexLeastSquaresSolver()
        
        # Generate test data for a single system (adding batch dimension)
        X, X_prime, U_true = _make_random_complex_transformation_data(30, 2, seed=42)
        
        # Add system dimension
        X_batch = X.unsqueeze(0)  # (1, num_samples, dim)
        X_prime_batch = X_prime.unsqueeze(0)
        
        # Use the inherited solve method
        U_estimated_batch = solver.solve(X_batch, X_prime_batch)
        
        # Check shape and accuracy
        assert U_estimated_batch.shape == (1, 4, 4)  # (1 system, 4x4 matrix)
        
        U_estimated = U_estimated_batch[0]
        _assert_allclose(
            U_true, U_estimated,
            err_msg="Integration with solve method failed"
        )
    

class TestComplexSolverComparison:
    """Compare ComplexLeastSquaresSolver against other solvers."""
    
    @pytest.mark.parametrize("complex_dim", [2, 3])
    @pytest.mark.parametrize("seed", range(NUM_SEEDS))
    def test_all_solvers_consistency(self, complex_dim, seed):
        """Test that all solvers produce valid solutions for the same problem."""
        complex_solver = ComplexLeastSquaresSolver()
        real_solver = LeastSquaresSolver(orthogonal_solution=False)
        
        # Generate test data
        X, X_prime, _ = _make_random_complex_transformation_data(50, complex_dim, seed=seed)
        
        # Solve with different methods
        U_complex = complex_solver._solve_system(X, X_prime)
        U_real = real_solver._solve_system(X, X_prime)
        
        # All solutions should solve the linear system accurately
        X_prime_complex_pred = X @ U_complex.T
        X_prime_real_pred = X @ U_real.T
        
        _assert_allclose(X_prime, X_prime_complex_pred, err_msg="Complex solver prediction failed")
        _assert_allclose(X_prime, X_prime_real_pred, err_msg="Real solver prediction failed")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 