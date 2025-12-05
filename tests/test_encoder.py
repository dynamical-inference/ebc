import pytest
import torch

from groupcl.models.encoder import MLPNormedComplex, NormComplexLayer, NormLayer
from groupcl.utils.complex import real_data_to_complex

# Test constants
ATOL = 1e-6
RTOL = 1e-5


def _assert_allclose(a, b, atol=ATOL, rtol=RTOL, err_msg=""):
    """Helper function for tensor comparison with custom error message."""
    assert torch.allclose(
        a, b, atol=atol,
        rtol=rtol), f"{err_msg}: max diff: {torch.max(torch.abs(a - b))}"


class TestNormComplexLayer:

    @pytest.mark.parametrize("batch_size", [1, 8])
    @pytest.mark.parametrize("latent_dim", [4, 16])
    @pytest.mark.parametrize("content_dims", [0, 2, 4])
    def test_output_shape(self, batch_size, latent_dim, content_dims):
        """Test that the output shape is the same as the input shape."""
        if content_dims >= latent_dim or (latent_dim - content_dims) % 2 != 0:
            pytest.skip(
                "content_dims must be smaller than latent_dim and group dim must be even."
            )

        layer = NormComplexLayer(content_dims=content_dims)
        inp = torch.randn(batch_size, latent_dim)
        out = layer(inp)
        assert out.shape == inp.shape

    @pytest.mark.parametrize("batch_size", [1, 8])
    @pytest.mark.parametrize("latent_dim", [4, 16, 32])
    def test_normalization_full(self, batch_size, latent_dim):
        """Test that the full output is normalized when content_dims is 0."""
        layer = NormComplexLayer(content_dims=0)
        inp = torch.randn(batch_size, latent_dim)
        out = layer(inp)

        # Check that the group embeddings are normalized in the complex domain
        out_complex = real_data_to_complex(out)
        magnitudes = torch.abs(out_complex)
        expected_magnitudes = torch.ones_like(magnitudes)
        _assert_allclose(magnitudes,
                         expected_magnitudes,
                         err_msg="Complex magnitudes should be 1")

    @pytest.mark.parametrize("batch_size", [1, 8])
    @pytest.mark.parametrize("latent_dim", [8, 16])
    @pytest.mark.parametrize("content_dims", [2, 4])
    def test_normalization_split(self, batch_size, latent_dim, content_dims):
        """Test normalization with group and content subspaces."""
        layer = NormComplexLayer(content_dims=content_dims,
                                 normalize_content=False)
        inp = torch.randn(batch_size, latent_dim)
        out = layer(inp)

        group_dim = latent_dim - content_dims
        group_slice = slice(0, group_dim)
        content_slice = slice(group_dim, latent_dim)

        group_part = out[..., group_slice]
        content_part = out[..., content_slice]

        # Check group part normalization
        group_complex = real_data_to_complex(group_part)
        magnitudes = torch.abs(group_complex)
        expected_magnitudes = torch.ones_like(magnitudes)
        _assert_allclose(
            magnitudes,
            expected_magnitudes,
            err_msg="Group part complex magnitudes should be 1",
        )

        # Check content part is unchanged
        original_content_part = inp[..., content_slice]
        _assert_allclose(content_part,
                         original_content_part,
                         err_msg="Content part should be unchanged")

    @pytest.mark.parametrize("batch_size", [1, 8])
    @pytest.mark.parametrize("latent_dim", [8, 16])
    @pytest.mark.parametrize("content_dims", [2, 4])
    def test_normalization_with_content_normalized(self, batch_size, latent_dim,
                                                   content_dims):
        """Test normalization with group and normalized content subspaces."""
        layer = NormComplexLayer(content_dims=content_dims,
                                 normalize_content=True)
        inp = torch.randn(batch_size, latent_dim)
        out = layer(inp)

        group_dim = latent_dim - content_dims
        group_slice = slice(0, group_dim)
        content_slice = slice(group_dim, latent_dim)

        group_part = out[..., group_slice]
        content_part = out[..., content_slice]

        # Check group part normalization
        group_complex = real_data_to_complex(group_part)
        magnitudes = torch.abs(group_complex)
        expected_magnitudes = torch.ones_like(magnitudes)
        _assert_allclose(
            magnitudes,
            expected_magnitudes,
            err_msg="Group part complex magnitudes should be 1",
        )

        # Check content part is normalized
        content_norm = torch.norm(content_part, dim=-1)
        expected_norm = torch.ones_like(content_norm)
        _assert_allclose(content_norm,
                         expected_norm,
                         err_msg="Content part should be normalized")

    def test_epsilon(self):
        """Test that epsilon prevents division by zero and handles small values."""
        latent_dim = 4
        layer = NormComplexLayer(epsilon=1e-8)

        # Test with zero input
        inp = torch.zeros(1, latent_dim)
        out = layer(inp)
        assert not torch.isnan(out).any() and not torch.isinf(out).any()
        out_complex = real_data_to_complex(out)
        magnitudes = torch.abs(out_complex)
        expected = torch.zeros_like(magnitudes)
        _assert_allclose(
            magnitudes,
            expected,
        )
        _assert_allclose(
            out,
            torch.zeros_like(out),
        )

        # Test with very small, non-zero input (magnitude < epsilon)
        inp = torch.rand(1, latent_dim) * 1e-6
        out = layer(inp)
        assert not torch.isnan(out).any() and not torch.isinf(out).any()
        out_complex = real_data_to_complex(out)
        magnitudes = torch.abs(out_complex)
        expected = torch.ones_like(magnitudes)
        _assert_allclose(magnitudes,
                         expected,
                         err_msg="Small inputs should be normalized to 1")


class TestMLPNormedComplex:

    @pytest.mark.parametrize("input_dim", [10])
    @pytest.mark.parametrize("output_dim", [4, 8])
    @pytest.mark.parametrize("content_dims", [0, 2])
    @pytest.mark.parametrize("batch_size", [1, 16])
    def test_forward_pass_and_shape(self, input_dim, output_dim, content_dims,
                                    batch_size):
        """Test forward pass and output shape."""
        if content_dims >= output_dim or (output_dim - content_dims) % 2 != 0:
            pytest.skip(
                "content_dims must be smaller than output_dim and group dim must be even."
            )

        model = MLPNormedComplex(input_dim=input_dim,
                                 output_dim=output_dim,
                                 content_dims=content_dims)
        model.init_parameters()
        inp = torch.randn(batch_size, input_dim)
        out = model(inp)
        assert out.shape == (batch_size, output_dim)

    @pytest.mark.parametrize("input_dim", [10])
    @pytest.mark.parametrize("output_dim", [4, 8])
    @pytest.mark.parametrize("batch_size", [1, 16])
    def test_normalization_properties(self, input_dim, output_dim, batch_size):
        """Test that the output has normalized complex magnitudes."""
        content_dims = 0
        model = MLPNormedComplex(input_dim=input_dim,
                                 output_dim=output_dim,
                                 content_dims=content_dims)
        model.init_parameters()
        inp = torch.randn(batch_size, input_dim)
        out = model(inp)

        out_complex = real_data_to_complex(out)
        magnitudes = torch.abs(out_complex)
        expected_magnitudes = torch.ones_like(magnitudes)
        _assert_allclose(
            magnitudes,
            expected_magnitudes,
            err_msg="Complex magnitudes of output should be 1",
        )

    @pytest.mark.parametrize("input_dim", [10])
    @pytest.mark.parametrize("output_dim", [8])
    @pytest.mark.parametrize("content_dims", [2, 4])
    @pytest.mark.parametrize("batch_size", [1, 16])
    def test_normalization_properties_with_content(self, input_dim, output_dim,
                                                   content_dims, batch_size):
        """Test that the group part of the output is normalized."""
        model = MLPNormedComplex(
            input_dim=input_dim,
            output_dim=output_dim,
            content_dims=content_dims,
            normalize_content=False,
        )
        model.init_parameters()
        inp = torch.randn(batch_size, input_dim)
        out = model(inp)

        group_dim = output_dim - content_dims
        group_slice = slice(0, group_dim)

        group_part = out[..., group_slice]

        # Check group part normalization
        group_complex = real_data_to_complex(group_part)
        magnitudes = torch.abs(group_complex)
        expected_magnitudes = torch.ones_like(magnitudes)
        _assert_allclose(
            magnitudes,
            expected_magnitudes,
            err_msg="Group part complex magnitudes should be 1",
        )

    @pytest.mark.parametrize("input_dim", [10])
    @pytest.mark.parametrize("output_dim", [8])
    @pytest.mark.parametrize("content_dims", [2, 4])
    @pytest.mark.parametrize("batch_size", [1, 16])
    def test_mlp_normalization_with_content_normalized(self, input_dim,
                                                       output_dim, content_dims,
                                                       batch_size):
        """Test that the group part is normalized and content part is normalized."""
        model = MLPNormedComplex(
            input_dim=input_dim,
            output_dim=output_dim,
            content_dims=content_dims,
            normalize_content=True,
        )
        model.init_parameters()
        inp = torch.randn(batch_size, input_dim)
        out = model(inp)

        group_dim = output_dim - content_dims
        group_slice = slice(0, group_dim)
        content_slice = slice(group_dim, output_dim)

        group_part = out[..., group_slice]
        content_part = out[..., content_slice]

        # Check group part normalization
        group_complex = real_data_to_complex(group_part)
        magnitudes = torch.abs(group_complex)
        expected_magnitudes = torch.ones_like(magnitudes)
        _assert_allclose(
            magnitudes,
            expected_magnitudes,
            err_msg="Group part complex magnitudes should be 1",
        )

        # Check content part is normalized
        content_norm = torch.norm(content_part, dim=-1)
        expected_norm = torch.ones_like(content_norm)
        _assert_allclose(content_norm,
                         expected_norm,
                         err_msg="Content part should be normalized")
