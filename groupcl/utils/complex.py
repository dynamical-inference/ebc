import torch
from jaxtyping import Complex, Float, jaxtyped
from jaxtyping._typeguard import typechecked
from torch import Tensor


@jaxtyped(typechecker=typechecked)
def complex_matrix_to_real(
    C: Complex[torch.Tensor, "*batch n n"]
) -> Float[torch.Tensor, "*batch 2*n 2*n"]:
    """
    Converts a batch of n x n complex matrices into their 2n x 2n real equivalents.
    
    The complex matrix C = A + iB is converted to the real matrix:
    [ Real(C)  -Imag(C) ]   [ A  -B ]
    [ Imag(C)   Real(C) ] = [ B   A ]
    
    This representation preserves the algebraic structure such that complex 
    matrix multiplication corresponds to real matrix multiplication.
    
    Args:
        C: Complex matrix of shape (*batch, n, n)
        
    Returns:
        Real matrix of shape (*batch, 2n, 2n) representing the complex matrix
        
    Example:
        >>> C = torch.tensor([[1+2j, 3+4j], [5+6j, 7+8j]])
        >>> R = complex_matrix_to_real(C)
        >>> R.shape
        torch.Size([4, 4])
    """
    *batch_dims, n, _ = C.shape
    A_real = torch.zeros((*batch_dims, 2 * n, 2 * n),
                         dtype=torch.float32,
                         device=C.device)

    A_real[..., :n, :n] = C.real  # Real part
    A_real[..., :n, n:] = -C.imag  # -Imaginary part
    A_real[..., n:, :n] = C.imag  # Imaginary part
    A_real[..., n:, n:] = C.real  # Real part

    return A_real


@jaxtyped(typechecker=typechecked)
def complex_data_to_real(
        D: Complex[torch.Tensor,
                   "*batch n"]) -> Float[torch.Tensor, "*batch 2*n"]:
    """
    Converts a batch of complex data vectors/matrices into their real equivalent.
    
    The complex data matrix is converted by concatenating real and imaginary parts
    along the last dimension: [ Real(D) | Imag(D) ]
    
    This representation allows complex vector operations to be performed 
    using real arithmetic when combined with complex_matrix_to_real.
    
    Args:
        D: Complex data of shape (*batch, n)
        
    Returns:
        Real data of shape (*batch, 2n) where:
        - First n columns are the real parts
        - Last n columns are the imaginary parts
        
    Example:
        >>> D = torch.tensor([[1+2j, 3+4j], [5+6j, 7+8j]])  # (2 samples, 2 dimensions)
        >>> R = complex_data_to_real(D)
        >>> R.shape
        torch.Size([2, 4])
    """
    return torch.cat([D.real, D.imag], dim=-1)


@jaxtyped(typechecker=typechecked)
def real_matrix_to_complex(
    real_matrix: Float[Tensor, "*batch n n"]
) -> Complex[Tensor, "*batch n//2 n//2"]:
    """
    Converts a batch of 2n x 2n real matrices into their n x n complex equivalent.
    
    The real matrix must have the block structure:
    [ A  -B ]
    [ B   A ]
    where A and B are n x n real matrices representing the real and imaginary parts.
    
    This function is the inverse of complex_matrix_to_real.
    
    Args:
        real_matrix: Real matrix of shape (*batch, 2n, 2n) with the required block structure
        
    Returns:
        Complex matrix of shape (*batch, n, n) where C = A + iB
        
    Raises:
        ValueError: If the matrix dimension is not even
        
    Note:
        If the input matrix doesn't have the exact required structure,
        a warning is printed but the conversion proceeds using the 
        top-left and bottom-left blocks as real and imaginary parts.
        
    Example:
        >>> R = torch.tensor([[1, 0, -2, 0], [0, 1, 0, -2], 
        ...                   [2, 0, 1, 0], [0, 2, 0, 1]])
        >>> C = real_matrix_to_complex(R)
        >>> C.shape
        torch.Size([2, 2])
    """
    dim = real_matrix.shape[-1]
    if dim % 2 != 0:
        raise ValueError("Matrix last dimension must be even.")
    n = dim // 2

    A = real_matrix[..., :n, :n]  # Real part
    B = real_matrix[..., n:, :n]  # Imaginary part

    # Optional: Check for the required structure
    O12 = real_matrix[..., :n, n:]
    O22 = real_matrix[..., n:, n:]
    if not torch.allclose(A, O22, atol=1e-6) or not torch.allclose(
            O12, -B, atol=1e-6):
        print(
            "Warning: Matrix does not have the ideal structure for a complex map."
        )

    return torch.complex(A, B)


@jaxtyped(typechecker=typechecked)
def real_data_to_complex(
        D: Float[torch.Tensor,
                 "*batch n"]) -> Complex[torch.Tensor, "*batch n//2"]:
    """
    Converts a batch of real data vectors/matrices into their complex equivalent.
    
    The real data represents complex data where:
    - First half of the last dimension are the real parts
    - Second half of the last dimension are the imaginary parts
    
    This function is the inverse of complex_data_to_real.
    
    Args:
        D: Real data of shape (*batch, 2n)
        
    Returns:
        Complex data of shape (*batch, n)
        
    Raises:
        ValueError: If the data's last dimension is not even
        
    Example:
        >>> D = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]], dtype=torch.float32)  # (2 samples, 4 real dims)
        >>> C = real_data_to_complex(D)
        >>> C.shape
        torch.Size([2, 2])
        >>> C
        tensor([[1.+5.j, 3.+7.j],
                [2.+6.j, 4.+8.j]])
    """
    dim = D.shape[-1]
    if dim % 2 != 0:
        raise ValueError("Data's last dimension must be even.")
    n = dim // 2

    real_part = D[..., :n]
    imag_part = D[..., n:]

    return torch.complex(real_part, imag_part)


@jaxtyped(typechecker=typechecked)
def solve_complex_matrix_system(
    X: Complex[torch.Tensor, "*batch m n"],
    Y: Complex[torch.Tensor, "*batch m n"],
) -> Complex[torch.Tensor, "*batch n n"]:
    """
    Solve for transformation matrix U in the equation Y = X @ U.T using complex arithmetic.
    
    This function solves the linear system where each row of X is transformed
    by matrix U.T to produce the corresponding row of Y. This is solved for a batch of systems.
    
    Args:
        X: Complex input data of shape (*batch, m, n) where m is num_samples and n is num_dimensions
        Y: Complex output data of shape (*batch, m, n) where Y = X @ U.T
        
    Returns:
        U: Complex transformation matrix of shape (*batch, n, n) such that Y ≈ X @ U.T
        
    Example:
        >>> U_true = torch.tensor([[1+1j, 2+0j], [0+1j, 1-1j]])
        >>> X = torch.randn(100, 2, dtype=torch.complex64)  # (100 samples, 2 dimensions)
        >>> Y = X @ U_true.T
        >>> U_estimated = solve_complex_matrix_system(X, Y)
        >>> torch.allclose(U_true, U_estimated, atol=1e-4)
        True
    """
    # X and Y are already in samples-as-rows format (*batch, m, n)
    # Solve Y = X @ U.T for U.T
    U_T = torch.linalg.lstsq(X, Y).solution
    U = U_T.transpose(-1, -2)

    return U


@jaxtyped(typechecker=typechecked)
def solve_real_matrix_system(
    X: Float[torch.Tensor, "*batch m n"],
    Y: Float[torch.Tensor, "*batch m n"],
) -> Float[torch.Tensor, "*batch n n"]:
    """
    Solve for transformation matrix U in the equation Y = X @ U.T using real arithmetic.
    
    This function solves the linear system where each row of X is transformed
    by matrix U.T to produce the corresponding row of Y.
    
    Args:
        X: Real input data of shape (*batch, m, n) where m is num_samples and n is num_dimensions
        Y: Real output data of shape (*batch, m, n) where Y = X @ U.T
        
    Returns:
        U: Real transformation matrix of shape (*batch, n, n) such that Y ≈ X @ U.T
        
    Example:
        >>> U_true = torch.randn(3, 3)
        >>> X = torch.randn(100, 3)  # (100 samples, 3 dimensions)
        >>> Y = X @ U_true.T
        >>> U_estimated = solve_real_matrix_system(X, Y)
        >>> torch.allclose(U_true, U_estimated, atol=1e-4)
        True
    """
    # X and Y are already in samples-as-rows format (*batch, m, n)
    # Solve Y = X @ U.T for U.T
    U_T = torch.linalg.lstsq(X, Y).solution
    U = U_T.transpose(-1, -2)

    return U
