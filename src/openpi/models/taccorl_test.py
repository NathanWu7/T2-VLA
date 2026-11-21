"""Synthetic JAX tests: no external checkpoints, datasets, or training configuration imports."""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from openpi.models import gemma
from openpi.models import model as model_lib
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models import siglip
from openpi.models import tactile_encoder
from openpi.shared import nnx_utils
from openpi.shared.tactile_type import TactileType


def _config(**overrides):
    kwargs = dict(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
        action_dim=6,
        action_horizon=3,
        max_token_len=4,
        tactile_type=TactileType.EXPERT_HIS_C_FUT,
        tactile_dim=2,
        tactile_fusion="taccorl",
        tactile_streams=("tactile_prefix", "tactile_suffix"),
        tactile_prefix_dim_in=9 * 20 * 2,
        tactile_prefix_history=8,
        tactile_prefix_use_reference_frame=True,
        tactile_prefix_diff_from_reference=False,
    )
    return pi0_config.Pi0Config(**(kwargs | overrides))


@pytest.fixture
def tiny_backbones(monkeypatch):
    # Exercise real Gemma/SigLIP attention with small dimensions and vocabulary.
    # Pi0 hard-codes a large vision variant, so shrink only the test factory.
    original_siglip = siglip.Module

    def tiny_siglip(*args, **kwargs):
        kwargs.update(width=16, depth=1, mlp_dim=32, num_heads=2, patch_size=(112, 112), head_zeroinit=False)
        return original_siglip(*args, **kwargs)

    monkeypatch.setattr(siglip, "Module", tiny_siglip)
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 128)


def _observation(config, contact=None):
    obs = config.fake_obs(batch_size=2)
    return dataclasses.replace(
        obs,
        tactile_prefix=jax.random.normal(jax.random.key(1), (2, 9, 40)),
        tactile_contact=None if contact is None else jnp.array(contact, dtype=jnp.bool_),
        tokenized_prompt_mask=jnp.array([[True, True, False, False], [True, False, False, False]]),
        image_masks={name: jnp.array([True, name == "base_0_rgb"]) for name in obs.images},
    )


def _open_action_attention(model):
    # Pi05 adaRMS residual gates initialize to zero. Open them to ensure the
    # equivalence tests actually exercise attention, including cached attention.
    state = nnx.state(model, nnx.Param)
    changed = 0
    for path, param in state.flat_state().items():
        if path[-2:] == ("Dense_0", "bias") and any(
            name in path for name in ("pre_attention_norm_1", "pre_ffw_norm_1")
        ):
            width = param.value.shape[-1] // 3
            param.value = param.value.at[..., 2 * width :].set(0.5)
            changed += 1
    assert changed == 2
    nnx.update(model, state)


def _run(model, config, obs):
    key = jax.random.key(2)
    actions = config.fake_act(batch_size=2)
    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, actions)
    samples = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=3)
    assert loss.shape == (2, config.action_horizon)
    assert samples.shape == (2, config.action_horizon, config.action_dim)
    assert np.isfinite(loss).all()
    assert np.isfinite(samples).all()
    return loss, samples


@pytest.mark.usefixtures("tiny_backbones")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("contact", [None, (True, False)])
def test_loss_and_cached_sampling(dtype, contact):
    config = _config(dtype=dtype)
    model = config.create(jax.random.key(0))
    _open_action_attention(model)
    model.tactile_cross_attn.alpha.value = jnp.array(0.7)
    obs = _observation(config, contact)
    assert obs.tactile_suffix is None
    _run(model, config, obs)
    # An unspecified contact gate has exactly the same embedding/masks as open.
    if contact is None:
        implicit = model.embed_prefix(obs)
        explicit = model.embed_prefix(dataclasses.replace(obs, tactile_contact=jnp.ones(2, dtype=jnp.bool_)))
        for actual, expected in zip(implicit, explicit, strict=True):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.usefixtures("tiny_backbones")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_closed_gate_matches_absent_tokens(dtype):
    config = _config(dtype=dtype)
    model = config.create(jax.random.key(0))
    _open_action_attention(model)
    model.tactile_cross_attn.alpha.value = jnp.array(0.7)
    closed_obs = _observation(config, (False, False))
    closed = _run(model, config, closed_obs)
    opened = _run(model, config, dataclasses.replace(closed_obs, tactile_contact=jnp.ones(2, dtype=jnp.bool_)))
    # This guards against a vacuous comparison with zero adaRMS attention gates.
    assert not np.allclose(opened[1], closed[1], rtol=1e-4, atol=1e-4)

    # Reuse the very same model/weights and loss weighting; omit both sets of
    # tactile tokens using the legacy no-stream path. No second random init.
    model.tactile_fusion = "none"
    model.tactile_streams = ()
    absent = _run(model, config, dataclasses.replace(closed_obs, tactile_prefix=None, tactile_contact=None))
    tolerance = 2e-5 if dtype == "float32" else 2e-2
    for actual, expected in zip(closed, absent, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.usefixtures("tiny_backbones")
def test_prefix_identity_masks_positions_and_cache():
    config = _config()
    model = config.create(jax.random.key(0))
    _open_action_attention(model)
    obs = _observation(config, (True, False))
    prefix, prefix_mask, prefix_ar = model.embed_prefix(obs)
    suffix, suffix_mask, suffix_ar, cond = model.embed_suffix(obs, config.fake_act(2), jnp.ones(2))
    n_pre, n_suf = config.tactile_prefix_num_tokens, config.tactile_suffix_num_tokens
    assert float(model.tactile_cross_attn.alpha.value) == 0
    np.testing.assert_array_equal(prefix_ar, False)
    np.testing.assert_array_equal(suffix_ar, [True] + [False] * (n_suf - 1) + [True, False, False])
    np.testing.assert_array_equal(prefix_mask[:, -n_pre:], np.broadcast_to([[True], [False]], (2, n_pre)))
    np.testing.assert_array_equal(suffix_mask[:, :n_suf], np.broadcast_to([[True], [False]], (2, n_suf)))

    mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar = jnp.concatenate([prefix_ar, suffix_ar])
    positions = jnp.cumsum(mask, axis=1) - 1
    attention = pi0.make_attn_mask(mask, ar)
    cached_positions = jnp.sum(prefix_mask, axis=1)[:, None] + jnp.cumsum(suffix_mask, axis=1) - 1
    np.testing.assert_array_equal(cached_positions, positions[:, prefix.shape[1] :])
    cached_attention = jnp.concatenate(
        [prefix_mask[:, None, :] & suffix_mask[:, :, None], pi0.make_attn_mask(suffix_mask, suffix_ar)], axis=-1
    )
    np.testing.assert_array_equal(cached_attention, attention[:, prefix.shape[1] :, :])

    # Compare actual joint and KV-cached expert outputs, not just mask formulas.
    (_, joint_suffix), _ = model.PaliGemma.llm(
        [prefix, suffix], mask=attention, positions=positions, adarms_cond=[None, cond]
    )
    _, cache = model.PaliGemma.llm(
        [prefix, None],
        mask=pi0.make_attn_mask(prefix_mask, prefix_ar),
        positions=jnp.cumsum(prefix_mask, axis=1) - 1,
    )
    (_, cached_suffix), _ = model.PaliGemma.llm(
        [None, suffix], mask=cached_attention, positions=cached_positions, kv_cache=cache, adarms_cond=[None, cond]
    )
    np.testing.assert_allclose(
        cached_suffix[:, -config.action_horizon :], joint_suffix[:, -config.action_horizon :], rtol=2e-5, atol=2e-5
    )

    model.tactile_fusion = "none"
    model.tactile_streams = ()
    base_prefix, base_mask, base_ar = model.embed_prefix(obs)
    base_suffix, base_suffix_mask, base_suffix_ar, _ = model.embed_suffix(obs, config.fake_act(2), jnp.ones(2))
    # alpha=0 must leave Z_base exactly unchanged, even in the open row.
    np.testing.assert_array_equal(prefix[:, :-n_pre], base_prefix)
    np.testing.assert_array_equal(suffix[:, n_suf:], base_suffix)
    base_positions = jnp.cumsum(jnp.concatenate([base_mask, base_suffix_mask], axis=1), axis=1) - 1
    retained = np.r_[np.arange(base_prefix.shape[1]), np.arange(prefix.shape[1] + n_suf, mask.shape[1])]
    # Closing the gate does not shift any of the remaining tokens' positions.
    np.testing.assert_array_equal(positions[1, retained], base_positions[1])
    base_attention = pi0.make_attn_mask(
        jnp.concatenate([base_mask, base_suffix_mask], axis=1), jnp.concatenate([base_ar, base_suffix_ar])
    )
    np.testing.assert_array_equal(attention[1][retained[:, None], retained[None, :]], base_attention[1])
    removed = np.setdiff1d(np.arange(mask.shape[1]), retained)
    assert not np.asarray(attention[1, removed, :]).any()
    assert not np.asarray(attention[1, :, removed]).any()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_cross_attention_identity_and_masked_gradients(dtype):
    cross = tactile_encoder.TactilePrefixCrossAttention(32, 4, rngs=nnx.Rngs(0))
    base = jax.random.normal(jax.random.key(1), (2, 3, 32)).astype(dtype)
    tactile = jax.random.normal(jax.random.key(2), (2, 5, 32)).astype(dtype)
    mask = jnp.array([[True] * 5, [False] * 5])
    np.testing.assert_array_equal(cross(base, tactile, mask), base)
    cross.alpha.value = jnp.array(0.7)
    result = cross(base, tactile, mask)
    assert result.dtype == dtype
    assert np.isfinite(np.asarray(result, dtype=np.float32)).all()
    np.testing.assert_array_equal(result[1], base[1])
    assert not np.array_equal(result[0], base[0])
    grad = jax.grad(lambda x: cross(base, x, mask).astype(jnp.float32).sum())(tactile)
    assert np.isfinite(np.asarray(grad, dtype=np.float32)).all()
    np.testing.assert_array_equal(grad[1], 0)


@pytest.mark.usefixtures("tiny_backbones")
def test_cross_attention_can_be_disabled_and_encoding_is_shared(monkeypatch):
    config = _config(tactile_prefix_cross_attn=False, tactile_prefix_num_tokens=3, tactile_suffix_num_tokens=2)
    model = config.create(jax.random.key(0))
    assert model.tactile_cross_attn is None
    assert not any(path[0] == "tactile_cross_attn" for path in nnx.state(model, nnx.Param).flat_state())
    calls = []
    original_call = tactile_encoder.MarkerTokenEncoder.__call__

    def counted_call(self, tactile):
        calls.append(tactile.shape)
        return original_call(self, tactile)

    monkeypatch.setattr(tactile_encoder.MarkerTokenEncoder, "__call__", counted_call)
    obs = _observation(config)
    model.compute_loss(jax.random.key(0), obs, config.fake_act(2))
    assert calls == [(2, 9, 40)]
    calls.clear()
    model.sample_actions(jax.random.key(0), obs, num_steps=3)
    # Encoding in the traced while-loop body would produce an extra call here.
    assert calls == [(2, 9, 40)]


def test_observation_contact_roundtrip_and_preprocessing():
    obs = _observation(_config(), (True, False))
    restored = model_lib.Observation.from_dict(obs.to_dict())
    np.testing.assert_array_equal(restored.tactile_contact, obs.tactile_contact)
    processed = model_lib.preprocess_observation(None, restored, tactile_type=TactileType.EXPERT_HIS_C_FUT)
    np.testing.assert_array_equal(processed.tactile_prefix, obs.tactile_prefix)
    np.testing.assert_array_equal(processed.tactile_contact, obs.tactile_contact)
    disabled = model_lib.preprocess_observation(None, restored, tactile_type=TactileType.NO)
    assert disabled.tactile_prefix is None
    assert disabled.tactile_contact is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tactile_type": TactileType.NO}, "EXPERT_HIS_C_FUT"),
        ({"tactile_streams": ("tactile_prefix",)}, "both"),
        ({"tactile_prefix_dim_in": None}, "tactile_prefix_dim_in"),
        ({"tactile_prefix_history": None}, "tactile_prefix_history"),
        ({"tactile_prefix_use_reference_frame": None}, "tactile_prefix_use_reference_frame"),
        ({"tactile_prefix_diff_from_reference": None}, "tactile_prefix_diff_from_reference"),
        ({"tactile_prefix_dim_in": 359}, "frames"),
        ({"tactile_prefix_num_tokens": 0}, "tactile_prefix_num_tokens"),
        ({"tactile_cross_attn_heads": 3}, "divide"),
        ({"tactile_suffix_placement": "prefix"}, "tactile_suffix_placement"),
    ],
)
def test_invalid_marker_contract(overrides, message):
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


@pytest.mark.usefixtures("tiny_backbones")
def test_missing_markers_are_an_error():
    config = _config()
    model = config.create(jax.random.key(0))
    missing = dataclasses.replace(_observation(config), tactile_prefix=None)
    with pytest.raises(ValueError, match="observation.tactile_prefix"):
        model.compute_loss(jax.random.key(0), missing, config.fake_act(2))
    with pytest.raises(ValueError, match="observation.tactile_prefix"):
        model.sample_actions(jax.random.key(0), missing, num_steps=1)


@pytest.mark.usefixtures("tiny_backbones")
def test_legacy_tacfield_parameter_paths_and_single_token():
    config = _config(
        tactile_fusion="none", tactile_streams=("tactile_prefix",), tactile_prefix_encoder_type="tcn"
    )
    model = config.create(jax.random.key(0))
    # Recorded from HEAD fe1b4884's TactileTCNEncoder/create_tactile_encoder:
    # two blocks, three Linear kernels each; only block_0 needs a residual
    # projection (40 -> 2 * 64); final out_proj is 128 -> 64.
    expected = {
        f"blocks/block_{block}/kernels/kernel_{kernel}/{param}"
        for block in range(2)
        for kernel in range(3)
        for param in ("kernel", "bias")
    }
    expected |= {f"blocks/block_0/residual_proj/{param}" for param in ("kernel", "bias")}
    expected |= {f"out_proj/{param}" for param in ("kernel", "bias")}
    flat = nnx.state(model, nnx.Param).flat_state()
    actual = {"/".join(path[1:]) for path in flat if path[0] == "tactile_prefix_encoder"}
    assert actual == expected
    assert not any(path[0] in ("marker_token_encoder", "tactile_cross_attn") for path in flat)
    obs = _observation(config, (False, False))
    prefix, mask, _ = model.embed_prefix(obs)
    absent, _, _ = model.embed_prefix(dataclasses.replace(obs, tactile_prefix=None))
    assert prefix.shape[1] == absent.shape[1] + 1
    # The new contact field has no effect on legacy behavior.
    np.testing.assert_array_equal(mask[:, -1], True)
    expected_token = model.tactile_prefix_encoder(obs.tactile_prefix)
    np.testing.assert_array_equal(prefix[:, -1], expected_token)


@pytest.mark.parametrize(("reference", "difference"), [(True, False), (True, True), (False, False)])
@pytest.mark.parametrize("frames", [3, 9, 11])
def test_tcn_frame_selection_preserves_legacy_outputs(reference, difference, frames):
    encoder = tactile_encoder.TactileTCNEncoder(
        in_dim=4,
        history_len=8,
        hidden_dim=8,
        emb_dim=6,
        has_reference_frame=reference,
        diff_from_reference=difference,
        rngs=nnx.Rngs(0),
    )
    markers = jax.random.normal(jax.random.key(1), (2, frames, 4))
    # Explicit pre-refactor selection, followed by the same legacy parameters.
    if reference:
        selected = markers[:, 1:9] - markers[:, :1] if difference else markers[:, :9]
    else:
        selected = markers[:, -8:]
    hidden = selected
    for block in encoder.blocks.values():
        hidden = block(hidden)
    np.testing.assert_array_equal(encoder(markers), encoder.out_proj(hidden[:, -1]))


def test_marker_heads_and_token_embeddings():
    encoder = tactile_encoder.MarkerTokenEncoder(
        in_dim=40,
        history_len=8,
        has_reference_frame=True,
        diff_from_reference=False,
        prefix_width=32,
        suffix_width=16,
        prefix_num_tokens=3,
        suffix_num_tokens=2,
        rngs=nnx.Rngs(0),
    )
    prefix, suffix = encoder(jnp.zeros((2, 9, 40)))
    assert prefix.shape == (2, 3, 32)
    assert suffix.shape == (2, 2, 16)
    # With zero input and zero Linear biases, these are exactly the learned
    # per-token embeddings, independently parameterized in the two heads.
    np.testing.assert_array_equal(prefix[0], encoder.prefix_token_embedding.value)
    np.testing.assert_array_equal(suffix[0], encoder.suffix_token_embedding.value)
    assert not np.array_equal(prefix[0, 0], prefix[0, 1])
    assert not np.array_equal(suffix[0, 0], suffix[0, 1])


@pytest.mark.usefixtures("tiny_backbones")
def test_taccorl_params_checkpoint_roundtrip(tmp_path):
    config = _config()
    model = config.create(jax.random.key(0))
    original_state = nnx.state(model, nnx.Param).flat_state()

    # save_state stores the inference params as a separate Orbax item whose
    # value is the NNX state under the "params" key.
    params_path = tmp_path / "params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(params_path, {"params": nnx.state(model)})

    restored_params = model_lib.restore_params(params_path, restore_type=np.ndarray)
    restored_model = config.load(restored_params)
    restored_state = nnx.state(restored_model, nnx.Param).flat_state()

    for path, original in original_state.items():
        if path[0] not in ("marker_token_encoder", "tactile_cross_attn"):
            continue
        assert path in restored_state
        np.testing.assert_array_equal(restored_state[path].value, original.value)
