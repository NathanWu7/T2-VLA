import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from openpi.models import tactile_encoder as tactile_encoder_jax
from openpi.models_pytorch.tactile_encoder_pytorch import (
    MarkerTokenEncoder,
    TactilePrefixCrossAttention,
    TactileTCNEncoder,
)


def _encoder() -> TactileTCNEncoder:
    return TactileTCNEncoder(
        input_dim=6,
        hidden_dim=8,
        output_dim=4,
        history_len=2,
        has_reference_frame=True,
        diff_from_reference=False,
    )


def test_tactile_tcn_encoder_builds_one_prefix_embedding():
    output = _encoder()(torch.zeros(3, 3, 6))

    assert output.shape == (3, 4)


def test_tactile_tcn_encoder_rejects_wrong_feature_width():
    with pytest.raises(ValueError, match="expected input_dim=6"):
        _encoder()(torch.zeros(1, 3, 5))


def test_legacy_tactile_tcn_state_dict_keys_remain_stable():
    keys = set(_encoder().state_dict())
    assert keys == {
        f"blocks.{block}.kernels.{kernel}.{param}"
        for block in range(2)
        for kernel in range(3)
        for param in ("weight", "bias")
    } | {"blocks.0.residual_proj.weight", "blocks.0.residual_proj.bias", "out_proj.weight", "out_proj.bias"}


def _copy_jax_marker_params(jax_module, torch_module):
    """Copy NNX MarkerTokenEncoder params using the documented path mapping."""
    jax_params = nnx.state(jax_module, nnx.Param).flat_state()
    target = torch_module.state_dict()
    copied = {}
    for path, value in jax_params.items():
        pieces = list(path)
        pieces = [piece.removeprefix("block_").removeprefix("kernel_") for piece in pieces]
        if pieces[-1] == "kernel":
            pieces[-1] = "weight"
        name = ".".join(pieces)
        tensor = torch.as_tensor(np.asarray(value.value))
        copied[name] = tensor.T if name.endswith("weight") else tensor
    target.update(copied)
    torch_module.load_state_dict(target)


def test_marker_token_encoder_matches_jax_with_transposed_linear_kernels():
    kwargs = dict(
        in_dim=6,
        history_len=2,
        has_reference_frame=True,
        diff_from_reference=False,
        prefix_width=8,
        suffix_width=4,
        prefix_num_tokens=3,
        suffix_num_tokens=2,
    )
    jax_module = tactile_encoder_jax.MarkerTokenEncoder(rngs=nnx.Rngs(0), **kwargs)
    torch_module = MarkerTokenEncoder(**kwargs)
    _copy_jax_marker_params(jax_module, torch_module)
    tactile = jax.random.normal(jax.random.key(1), (2, 3, 6))
    expected = jax_module(tactile)
    actual = torch_module(torch.from_numpy(np.asarray(tactile)))
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(got.detach().numpy(), np.asarray(want), rtol=2e-5, atol=2e-5)


def _copy_jax_cross_params(jax_module, torch_module):
    target = torch_module.state_dict()
    for path, value in nnx.state(jax_module, nnx.Param).flat_state().items():
        pieces = list(path)
        if pieces[-1] == "scale":
            pieces[-1] = "weight"
        elif pieces[-1] == "kernel":
            pieces[-1] = "weight"
        name = ".".join(pieces)
        tensor = torch.as_tensor(np.asarray(value.value))
        if name == "alpha":
            tensor = tensor.reshape(1)
        target[name] = tensor.T if pieces[-1] == "weight" and path[-1] == "kernel" else tensor
    torch_module.load_state_dict(target)


@pytest.mark.parametrize("contact", [True, False])
@pytest.mark.parametrize("alpha", [0.0, 0.7])
def test_tactile_cross_attention_matches_jax_and_zeroes_closed_rows(contact, alpha):
    jax_module = tactile_encoder_jax.TactilePrefixCrossAttention(8, 2, rngs=nnx.Rngs(0))
    torch_module = TactilePrefixCrossAttention(8, 2)
    assert jax_module.alpha.value.shape == ()
    assert torch_module.alpha.shape == (1,)
    assert torch_module.state_dict()["alpha"].shape == (1,)
    torch.testing.assert_close(torch_module.alpha, torch.zeros(1))
    jax_module.alpha.value = jnp.array(alpha, dtype=jnp.float32)
    _copy_jax_cross_params(jax_module, torch_module)
    torch.testing.assert_close(torch_module.alpha, torch.tensor([alpha]))
    base = jax.random.normal(jax.random.key(2), (2, 4, 8))
    tactile = jax.random.normal(jax.random.key(3), (2, 3, 8))
    mask = jnp.array([[True, True, True], [contact, contact, contact]])
    expected = jax_module(base, tactile, mask)
    actual = torch_module(
        torch.from_numpy(np.asarray(base)),
        torch.from_numpy(np.asarray(tactile)),
        torch.from_numpy(np.asarray(mask)),
    )
    np.testing.assert_allclose(actual.detach().numpy(), np.asarray(expected), rtol=2e-5, atol=2e-5)
