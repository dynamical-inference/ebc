import pytest
import numpy as np
from groupcl.metrics.utils import (
    fit_ortho_procrustes,
    predict_ortho_procrustes,
    fit_linear_regression,
    apply_identifiability_constant,
)
from scipy.stats import ortho_group
from sklearn.metrics import r2_score
import torch


def apply_identifiability_constant_np(A, x):
    return np.array(
        apply_identifiability_constant(A=torch.tensor(A), x=torch.tensor(x)))


# @pytest.mark.parametrize("fitting_method", ["lsqt", "ortho-procrustes"])
# def test_r2_ortho_procrustes(fitting_method):
#     pass


def generate_data(
    n_samples=100,
    n_features=2,
    orthogonal_matrix=True,
    seed=42,
):
    np.random.seed(seed)

    if orthogonal_matrix:
        A = ortho_group.rvs(n_features)
    else:
        max_tries = 10000
        for _ in range(max_tries):
            A = np.random.randn(n_features, n_features)
            # accept if det is not close to 0 and also not close to 1
            not_orthogonal = not np.allclose(
                np.abs(np.linalg.det(A)), 1, atol=1e-1)
            not_singular = not np.allclose(
                np.abs(np.linalg.det(A)), 0, atol=1e-3)
            if not_orthogonal and not_singular:
                break
        else:
            raise ValueError(
                "Failed to generate a non-singular matrix, non-orthogonal matrix"
            )

    x = np.random.randn(n_samples, n_features)
    y = apply_identifiability_constant_np(A=A, x=x)
    return x, y, A


# linear fit should always work perfectly
@pytest.mark.parametrize("n_samples", [100, 1000])
@pytest.mark.parametrize("n_features", [
    3,
    10,
])
@pytest.mark.parametrize("orthogonal_matrix", [
    False,
    True,
])
def test_fit_linear_regression(n_samples, n_features, orthogonal_matrix):
    x, y, A = generate_data(n_samples,
                            n_features,
                            orthogonal_matrix=orthogonal_matrix)
    lr = fit_linear_regression(x, y, bias=False)

    # y_pred = lr.predict(x)
    y_pred = apply_identifiability_constant_np(A=lr.coef_, x=x)
    r2 = r2_score(y, y_pred)

    assert r2 > 0.99, f"R2 score is too low: {r2}"
    np.testing.assert_allclose(A,
                               lr.coef_,
                               atol=1e-6,
                               err_msg="A matrix is not recovered")


@pytest.mark.parametrize("n_samples", [100, 1000])
@pytest.mark.parametrize("n_features", [
    3,
    10,
])
@pytest.mark.parametrize("orthogonal_matrix", [
    True,
    False,
])
def test_fit_ortho_procrustes(n_samples, n_features, orthogonal_matrix):
    x, y, A = generate_data(n_samples,
                            n_features,
                            orthogonal_matrix=orthogonal_matrix)
    A_hat = fit_ortho_procrustes(x, y)
    y_pred = predict_ortho_procrustes(A_hat, x)
    r2 = r2_score(y, y_pred)

    if orthogonal_matrix:
        assert r2 > 0.99, f"R2 score is too low: {r2}"
        np.testing.assert_allclose(A,
                                   A_hat,
                                   atol=1e-6,
                                   err_msg="A matrix is not recovered")
    else:
        assert r2 < 0.7, f"R2 score is too high: {r2}"


from groupcl.metrics.identifiability import R2


@pytest.mark.parametrize("n_samples", [100, 1000])
@pytest.mark.parametrize("n_features", [
    3,
    10,
])
def test_r2_ortho_procrustes(n_samples, n_features):
    R2_ortho = R2(fitting_method="ortho-procrustes",
                  direction="backward",
                  bias=False)
    R2_lsqt = R2(
        fitting_method="lsqt",
        direction="backward",
        bias=False,
    )

    z_true, z_pred, A = generate_data(n_samples,
                                      n_features,
                                      orthogonal_matrix=True)
    z_true = torch.tensor(z_true)
    z_pred = torch.tensor(z_pred)

    r2_ortho = R2_ortho._compute(z_true, z_pred)
    r2_lsqt = R2_lsqt._compute(z_true, z_pred)
    assert r2_lsqt > 0.99, f"R2_lsqt score is too low: {r2_lsqt}"
    assert r2_ortho > 0.99, f"R2_ortho score is too low: {r2_ortho}"


if __name__ == "__main__":
    pytest.main([__file__])
