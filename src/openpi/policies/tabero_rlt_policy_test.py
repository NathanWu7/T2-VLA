import json
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.torch import save_file
import torch

from openpi.policies import policy_config
from openpi.policies import tabero_rlt_policy
import openpi.transforms as transforms


def _write_bundle(
    tmp_path,
    *,
    normalized_action_bound=4.0,
    ref_num_action_chunks=2,
):
    encoder_path = tmp_path / "rlt_encoder.safetensors"
    actor_path = tmp_path / "rlt_actor.safetensors"
    save_file({"encoder.weight": torch.eye(2)}, encoder_path)
    save_file(
        {
            "backbone.0.weight": torch.zeros(2, 5),
            "backbone.0.bias": torch.zeros(2),
            "backbone.2.weight": torch.eye(2),
            "backbone.2.bias": torch.zeros(2),
            "backbone.4.weight": torch.eye(2),
            "backbone.4.bias": torch.zeros(2),
            "actor_mean.weight": torch.ones(2, 2),
            "actor_mean.bias": torch.zeros(2),
        },
        actor_path,
    )
    manifest = {
        "format": "tabero_rlt_t2vla",
        "action_parameterization": "residual",
        "action_composition": "reference_plus_bounded_delta",
        "residual_action_bound": 4.0,
        "action_space": "model_normalized",
        "normalized_action_bound": normalized_action_bound,
        "z_dim": 2,
        "proprio_dim": 1,
        "action_dim": 1,
        "num_action_chunks": 2,
        "ref_num_action_chunks": ref_num_action_chunks,
        "actor_hidden_dim": 2,
        "rlt_input_dim": 2,
        "rlt_embed_dim": 2,
        "rlt_num_rl_tokens": 1,
        "rlt_prefix_seq_len": 4,
        "rlt_num_layers": 1,
        "rlt_num_heads": 1,
        "rlt_mlp_ratio": 1.0,
        "rlt_image_only": False,
        "rlt_use_mask": True,
        "rlt_use_normalized_proprio": True,
        "state_indices": None,
        "reference_num_steps": 2,
        "reference_sampling_method": "flow_ode",
        "encoder_weights": encoder_path.name,
        "actor_weights": actor_path.name,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_bundle_manifest_loads_current_model_normalized_format(tmp_path):
    expected = _write_bundle(tmp_path)

    bundle = tabero_rlt_policy.TaberoRLTBundle.load(tmp_path)

    assert bundle.manifest.action_space == "model_normalized"
    assert bundle.manifest.normalized_action_bound == 4.0
    assert bundle.manifest.action_dim == 1
    assert bundle.manifest.num_action_chunks == 2
    assert bundle.manifest.rlt_use_normalized_proprio is True
    assert bundle.manifest.reference_num_steps == 2
    assert bundle.manifest.to_dict() == expected


def test_bundle_rejects_unknown_action_space(tmp_path):
    manifest = _write_bundle(tmp_path)
    manifest["action_space"] = "environment"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with np.testing.assert_raises_regex(ValueError, "model_normalized"):
        tabero_rlt_policy.TaberoRLTBundle.load(tmp_path)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"rlt_use_normalized_proprio": False}, "normalized proprio"),
        ({"reference_sampling_method": "flow_sde"}, "flow_ode"),
        ({"state_indices": [0, 2]}, "proprio_dim"),
    ],
)
def test_bundle_rejects_unsupported_input_semantics(tmp_path, updates, message):
    manifest = _write_bundle(tmp_path)
    manifest.update(updates)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        tabero_rlt_policy.TaberoRLTBundle.load(tmp_path)


@pytest.mark.parametrize("bound", [0.0, -1.0, float("inf"), float("nan")])
def test_bundle_rejects_non_finite_or_non_positive_action_bound(tmp_path, bound):
    _write_bundle(tmp_path, normalized_action_bound=bound)

    with pytest.raises(ValueError, match="finite and positive"):
        tabero_rlt_policy.TaberoRLTBundle.load(tmp_path)


def test_bundle_actor_input_uses_policy_chunks_not_available_reference_chunks(tmp_path):
    _write_bundle(tmp_path, ref_num_action_chunks=3)

    actor = tabero_rlt_policy.TaberoRLTBundle.load(tmp_path).build_actor()

    assert actor.backbone[0].in_features == 5


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"num_action_chunks": 0}, "positive"),
        ({"ref_num_action_chunks": 1}, "ref_num_action_chunks"),
        ({"z_dim": 3}, "z_dim"),
        ({"rlt_embed_dim": 3, "rlt_num_heads": 2, "z_dim": 3}, "divisible"),
    ],
)
def test_bundle_rejects_inconsistent_manifest_dimensions(tmp_path, updates, message):
    manifest = _write_bundle(tmp_path)
    manifest.update(updates)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        tabero_rlt_policy.TaberoRLTBundle.load(tmp_path)


def test_actor_uses_current_scaled_tanh_and_actor_checkpoint_layout():
    actor = tabero_rlt_policy.RLTActorMLP(
        input_dim=4,
        output_dim=2,
        hidden_dim=2,
        normalized_action_bound=4.0,
    )
    with torch.no_grad():
        for module in actor.backbone:
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
        actor.actor_mean.weight.zero_()
        actor.actor_mean.bias.copy_(torch.tensor([4.0, -4.0]))

    output = actor(torch.ones(1, 4))

    expected = 4.0 * torch.tanh(torch.tensor([[1.0, -1.0]]))
    torch.testing.assert_close(output, expected)
    assert set(actor.state_dict()) == {
        "backbone.0.weight",
        "backbone.0.bias",
        "backbone.2.weight",
        "backbone.2.bias",
        "backbone.4.weight",
        "backbone.4.bias",
        "actor_mean.weight",
        "actor_mean.bias",
    }


def test_actor_input_order_matches_rlinf_stage2():
    ref_chunk = torch.tensor([[[1.0], [2.0]]])
    z_rl = torch.tensor([[3.0]])
    proprio = torch.tensor([[4.0]])

    actor_input = tabero_rlt_policy.make_actor_input(
        ref_chunk=ref_chunk,
        z_rl=z_rl,
        proprio=proprio,
        num_action_chunks=2,
        action_dim=1,
    )

    torch.testing.assert_close(actor_input, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))


def test_flow_ode_step_uses_rlinf_floating_point_operation_order():
    x_t = torch.tensor([[[0.12345679, -1.2345679]]], dtype=torch.float32)
    velocity = torch.tensor([[[3.1415927, -2.7182817]]], dtype=torch.float32)
    timestep = torch.tensor([0.7], dtype=torch.float32)
    delta = torch.tensor([0.1], dtype=torch.float32)
    expanded_time = timestep[:, None, None].expand_as(x_t)
    expanded_delta = delta[:, None, None].expand_as(x_t)
    expected = (x_t - velocity * expanded_time) * (1 - (expanded_time - expanded_delta)) + (
        x_t + velocity * (1 - expanded_time)
    ) * (expanded_time - expanded_delta)

    actual = tabero_rlt_policy.rlinf_flow_ode_step(x_t, velocity, timestep, delta)

    assert torch.equal(actual, expected)


class _FakePaliGemma(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = SimpleNamespace(
            language_model=SimpleNamespace(config=SimpleNamespace(_attn_implementation=None))
        )

    def forward(self, *, inputs_embeds, **kwargs):
        del kwargs
        return (inputs_embeds[0] + 0.5, None), ("prefix-cache",)


class _FakePI0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(action_horizon=2, action_dim=1)
        self.paligemma_with_expert = _FakePaliGemma()

    def _preprocess_observation(self, observation, *, train):
        assert not train
        batch_size = observation.state.shape[0]
        return (
            [torch.zeros(batch_size, 1, 1)],
            [torch.ones(batch_size, dtype=torch.bool)],
            torch.zeros(batch_size, 1, dtype=torch.long),
            torch.ones(batch_size, 1, dtype=torch.bool),
            observation.state,
            None,
        )

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, tactile_prefix):
        del images, img_masks, lang_tokens, lang_masks, tactile_prefix
        return (
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            torch.tensor([[True, True]]),
            torch.tensor([[False, False]]),
        )

    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        del state, prefix_pad_masks, past_key_values, timestep
        return torch.zeros_like(x_t)


def _write_inference_bundle(tmp_path):
    encoder = tabero_rlt_policy.RLTTokenEncoder(
        input_dim=2,
        embed_dim=2,
        num_rl_tokens=1,
        prefix_seq_len=4,
        num_layers=1,
        num_heads=1,
        mlp_ratio=1.0,
    )
    save_file(
        {f"encoder.{key}": value for key, value in encoder.state_dict().items()}, tmp_path / "encoder.safetensors"
    )

    actor = tabero_rlt_policy.RLTActorMLP(
        input_dim=5,
        output_dim=2,
        hidden_dim=2,
        normalized_action_bound=4.0,
    )
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.actor_mean.bias.copy_(torch.tensor([4.0, -4.0]))
    save_file(actor.state_dict(), tmp_path / "actor.safetensors")

    manifest = {
        "format": "tabero_rlt_t2vla",
        "action_parameterization": "residual",
        "action_composition": "reference_plus_bounded_delta",
        "residual_action_bound": 4.0,
        "action_space": "model_normalized",
        "normalized_action_bound": 4.0,
        "z_dim": 2,
        "proprio_dim": 1,
        "action_dim": 1,
        "num_action_chunks": 2,
        "ref_num_action_chunks": 2,
        "rlt_input_dim": 2,
        "rlt_embed_dim": 2,
        "rlt_num_rl_tokens": 1,
        "rlt_prefix_seq_len": 4,
        "rlt_num_layers": 1,
        "rlt_num_heads": 1,
        "rlt_mlp_ratio": 1.0,
        "rlt_image_only": False,
        "rlt_use_mask": True,
        "rlt_use_normalized_proprio": True,
        "state_indices": None,
        "reference_num_steps": 2,
        "reference_sampling_method": "flow_ode",
        "actor_hidden_dim": 2,
        "encoder_weights": "encoder.safetensors",
        "actor_weights": "actor.safetensors",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))


def test_rlt_policy_model_runs_prefix_encoder_reference_and_actor(tmp_path):
    _write_inference_bundle(tmp_path)
    model = tabero_rlt_policy.TaberoRLTPolicyModel.from_bundle(_FakePI0(), tmp_path)
    observation = SimpleNamespace(state=torch.tensor([[0.25]]))

    actions = model.sample_actions(
        "cpu",
        observation,
        noise=torch.zeros(1, 2, 1),
        num_steps=2,
    )

    expected = 4.0 * torch.tanh(torch.tensor([[[1.0], [-1.0]]]))
    torch.testing.assert_close(actions, expected)


@pytest.mark.parametrize("mode", ["absolute", "residual"])
@pytest.mark.parametrize("zero_residual", [False, True])
def test_explicit_action_mode_controls_reference_addition(tmp_path, zero_residual, mode):
    _write_inference_bundle(tmp_path)
    if mode == "absolute":
        manifest_path = tmp_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(action_parameterization="absolute", action_composition="direct")
        manifest.pop("residual_action_bound")
        manifest_path.write_text(json.dumps(manifest))
    model = tabero_rlt_policy.TaberoRLTPolicyModel.from_bundle(_FakePI0(), tmp_path)
    if zero_residual:
        with torch.no_grad():
            model.actor.actor_mean.bias.zero_()
    # FakePI0 has zero velocity, so the reference equals the supplied noise.
    reference = torch.tensor([[[5.0], [-6.0]]])
    actions = model.sample_actions(
        "cpu", SimpleNamespace(state=torch.tensor([[0.25]])), noise=reference.clone(), num_steps=2
    )
    delta = torch.zeros_like(reference) if zero_residual else 4 * torch.tanh(torch.tensor([[[1.0], [-1.0]]]))
    expected = reference + delta if mode == "residual" else delta
    torch.testing.assert_close(actions, expected)


@pytest.mark.parametrize(
    "updates",
    [
        {"action_parameterization": "absolute"},
        {"action_composition": "direct"},
        {"residual_action_bound": 2.0},
        {"residual_action_bound": None},
    ],
)
def test_bundle_rejects_ambiguous_or_inconsistent_action_contract(tmp_path, updates):
    manifest = _write_bundle(tmp_path)
    manifest.update(updates)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="RLT bundle"):
        tabero_rlt_policy.TaberoRLTBundle.load(tmp_path)


def test_zero_residual_preserves_reference_precision_with_bf16_actor(tmp_path):
    _write_inference_bundle(tmp_path)
    model = tabero_rlt_policy.TaberoRLTPolicyModel.from_bundle(_FakePI0(), tmp_path)
    model.actor.to(dtype=torch.bfloat16)
    with torch.no_grad():
        model.actor.actor_mean.bias.zero_()
    reference = torch.tensor([[[5.01], [-6.03]]])
    actual = model.sample_actions(
        "cpu", SimpleNamespace(state=torch.tensor([[0.25]])), noise=reference.clone(), num_steps=2
    )
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_rlt_policy_rejects_reference_step_override_that_differs_from_bundle(tmp_path):
    _write_inference_bundle(tmp_path)
    model = tabero_rlt_policy.TaberoRLTPolicyModel.from_bundle(_FakePI0(), tmp_path)
    observation = SimpleNamespace(state=torch.tensor([[0.25]]))

    with pytest.raises(ValueError, match="reference_num_steps"):
        model.sample_actions(
            "cpu",
            observation,
            noise=torch.zeros(1, 2, 1),
            num_steps=3,
        )


def test_rlt_policy_rejects_bundle_for_different_base_checkpoint(tmp_path):
    _write_inference_bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["base_model"] = str(tmp_path / "expected_base")
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="base checkpoint"):
        tabero_rlt_policy.TaberoRLTPolicyModel.from_bundle(
            _FakePI0(),
            tmp_path,
            base_model_path=tmp_path / "different_base",
        )


def test_rlt_bundle_uses_base_checkpoint_content_hash_after_relocation(tmp_path):
    bundle_path = tmp_path / "bundle"
    bundle_path.mkdir()
    _write_inference_bundle(bundle_path)
    first_base = tmp_path / "first-base"
    moved_base = tmp_path / "moved-base"
    first_base.mkdir()
    moved_base.mkdir()
    (first_base / "model.safetensors").write_bytes(b"same model")
    (moved_base / "model.safetensors").write_bytes(b"same model")
    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["base_model"] = str(first_base)
    manifest["base_model_sha256"] = tabero_rlt_policy.checkpoint_sha256(first_base)
    manifest_path.write_text(json.dumps(manifest))

    tabero_rlt_policy.TaberoRLTPolicyModel.from_bundle(
        _FakePI0(),
        bundle_path,
        base_model_path=moved_base,
    )

    (moved_base / "model.safetensors").write_bytes(b"changed model")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        tabero_rlt_policy.TaberoRLTPolicyModel.from_bundle(
            _FakePI0(),
            bundle_path,
            base_model_path=moved_base,
        )


def test_rlt_policy_rejects_bundle_for_different_base_action_horizon(tmp_path):
    _write_inference_bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["base_action_horizon"] = 3
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="action horizon"):
        tabero_rlt_policy.TaberoRLTPolicyModel.from_bundle(_FakePI0(), tmp_path)


def test_bundle_build_encoder_preserves_checkpoint_dtype(tmp_path):
    _write_inference_bundle(tmp_path)
    state = tabero_rlt_policy.TaberoRLTBundle.load(tmp_path).load_encoder_state()
    save_file(
        {f"encoder.{key}": value.to(torch.bfloat16) for key, value in state.items()},
        tmp_path / "encoder.safetensors",
    )

    encoder = tabero_rlt_policy.TaberoRLTBundle.load(tmp_path).build_encoder()

    assert next(encoder.parameters()).dtype == torch.bfloat16


class _FakeLoadedPI0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = SimpleNamespace(to_bfloat16_for_selected_params=lambda precision: None)

    def sample_actions(self, *args, **kwargs):
        del args, kwargs
        return torch.zeros(1, 1, 1)


@pytest.mark.parametrize("action_mode", [None, "absolute", "residual"])
def test_create_trained_policy_wraps_pytorch_model_when_rlt_bundle_is_set(tmp_path, monkeypatch, action_mode):
    (tmp_path / "model.safetensors").touch()
    loaded_model = _FakeLoadedPI0()
    wrapped_model = _FakeLoadedPI0()
    model_config = SimpleNamespace(load_pytorch=lambda train_config, path: loaded_model)
    data_config = SimpleNamespace(
        asset_id=None,
        data_transforms=transforms.Group(),
        model_transforms=transforms.Group(),
        use_quantile_norm=False,
    )
    train_config = SimpleNamespace(
        name="pi0_test",
        model=model_config,
        data=SimpleNamespace(create=lambda assets_dirs, model: data_config),
        assets_dirs=tmp_path,
        policy_metadata={},
    )
    calls = []

    def wrap(model, bundle_path, **kwargs):
        calls.append((model, bundle_path, kwargs))
        return wrapped_model

    monkeypatch.setattr(tabero_rlt_policy.TaberoRLTPolicyModel, "from_bundle", wrap)

    policy = policy_config.create_trained_policy(
        train_config,
        tmp_path,
        norm_stats={},
        pytorch_device="cpu",
        rlt_bundle_path=tmp_path / "rlt",
        rlt_action_parameterization=action_mode,
    )

    assert policy._model is wrapped_model  # noqa: SLF001
    assert calls == [
        (
            loaded_model,
            tmp_path / "rlt",
            {
                "action_parameterization": action_mode,
                "base_model_path": tmp_path,
                "base_config_name": "pi0_test",
                "norm_asset_id": None,
                "use_quantile_norm": False,
            },
        )
    ]


def test_bundle_rejects_missing_residual_semantics(tmp_path):
    manifest = _write_bundle(tmp_path)
    for key in ("action_parameterization", "action_composition", "residual_action_bound"):
        manifest.pop(key)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="explicit action semantics"):
        tabero_rlt_policy.TaberoRLTBundle.load(tmp_path)


@pytest.mark.parametrize("mode", ["absolute", "residual"])
def test_unlabelled_bundle_requires_explicit_loading_mode(tmp_path, mode):
    _write_inference_bundle(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    for key in ("action_parameterization", "action_composition", "residual_action_bound"):
        manifest.pop(key)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="explicit action semantics"):
        tabero_rlt_policy.TaberoRLTPolicyModel.from_bundle(_FakePI0(), tmp_path)
    model = tabero_rlt_policy.TaberoRLTPolicyModel.from_bundle(_FakePI0(), tmp_path, action_parameterization=mode)
    reference = torch.tensor([[[5.0], [-6.0]]])
    actual = model.sample_actions(
        "cpu", SimpleNamespace(state=torch.tensor([[0.25]])), noise=reference.clone(), num_steps=2
    )
    bounded = 4 * torch.tanh(torch.tensor([[[1.0], [-1.0]]]))
    torch.testing.assert_close(actual, reference + bounded if mode == "residual" else bounded)
    assert "format_version" not in model.bundle.manifest.to_dict()


def test_explicit_load_cannot_override_declared_bundle_mode(tmp_path):
    _write_bundle(tmp_path)
    with pytest.raises(ValueError, match="conflicts"):
        tabero_rlt_policy.TaberoRLTBundle.load(tmp_path, action_parameterization="absolute")
