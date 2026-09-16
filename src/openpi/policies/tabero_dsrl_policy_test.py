"""Behavioral checks for configuration-driven DSRL deployment."""

import dataclasses
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.torch import save_file
import torch

from openpi.policies import policy as policy_module
from openpi.policies import policy_config
from openpi.policies.tabero_dsrl_policy import ActorContract
from openpi.policies.tabero_dsrl_policy import TaberoDSRLActor
from openpi.policies.tabero_dsrl_policy import TaberoDSRLBundle
from openpi.policies.tabero_dsrl_policy import TaberoDSRLPolicy


@pytest.fixture
def contract():
    return ActorContract(
        use_state=True,
        image_keys=["image", "wrist"],
        image_shapes=[[37, 53, 3], [31, 43, 3]],
        state_key="state",
        state_dim=7,
        tactile_key="tactile",
        tactile_shape=[9, 11, 2],
        image_latent_dim=8,
        state_latent_dim=12,
        tactile_latent_dim=16,
        hidden_dims=[19, 17],
        noise_dim=13,
        horizon=6,
        num_steps=4,
        dtype="bfloat16",
        image_preprocessing="uint8_bilinear64_align_false_minus_one_one",
        tactile_processing="reference_plus_history8_no_difference_causal_tcn2_kernel3",
        feature_order="state_ordered_images_tactile",
    )


@pytest.fixture
def observation(contract):
    torch.set_num_threads(2)
    obs = {
        k: torch.randint(256, s, dtype=torch.uint8)
        for k, s in zip(contract.image_keys, contract.image_shapes, strict=True)
    }
    return {**obs, "state": torch.randn(7), "tactile": torch.randn(9, 11, 2)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bundle(tmp_path, contract):
    base, root = tmp_path / "base", tmp_path / "bundle"
    base.mkdir()
    root.mkdir()
    (base / "model.safetensors").write_bytes(b"synthetic-base-identity")
    norm = base / "assets" / "norm" / "norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text("{}")
    state = TaberoDSRLActor(contract).state_dict()
    save_file(state, root / "actor.safetensors")
    manifest = {
        "format": "tabero_dsrl_t2vla",
        "algorithm": "dsrl-sac",
        "task_id": 9,
        "global_step": 3,
        "is_final": False,
        "source": {
            "checkpoint_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "observation_sha256": "c" * 64,
            "metadata": {},
            "semantics": {},
        },
        "base": {
            "model_sha256": digest(base / "model.safetensors"),
            "norm_sha256": digest(norm),
            "norm_asset_id": "norm",
            "config_name": "synthetic",
            "model": {"action_dim": 13, "action_horizon": 6, "discrete_state_input": False},
            "use_quantile_norm": True,
        },
        "actor_contract": dataclasses.asdict(contract),
        "actor_weights": "actor.safetensors",
        "actor_weights_sha256": digest(root / "actor.safetensors"),
        "actor_shapes": {k: list(v.shape) for k, v in state.items()},
    }
    (base / "config.json").write_text(json.dumps({"config_name": "synthetic", **manifest["base"]["model"]}))
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, base, manifest


def test_bundle_and_actor_noise(bundle, observation):
    root, base, _ = bundle
    loaded = TaberoDSRLBundle.load(root, base_checkpoint_dir=base)
    noise = loaded.actor.noise(observation)
    assert noise.shape == (1, 6, 13)
    assert noise.dtype == torch.bfloat16
    assert torch.equal(noise[:, 0], noise[:, -1])
    assert torch.isfinite(noise).all()
    assert (noise.abs() <= 1).all()


@pytest.mark.parametrize("field", ["model", "norm", "actor"])
def test_bundle_rejects_changed_artifact(bundle, field):
    root, base, _ = bundle
    path = {
        "model": base / "model.safetensors",
        "norm": base / "assets/norm/norm_stats.json",
        "actor": root / "actor.safetensors",
    }[field]
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="SHA-256"):
        TaberoDSRLBundle.load(root, base_checkpoint_dir=base)


@pytest.mark.parametrize("mutation", ["width", "dtype", "nan", "missing"])
def test_rehashed_inconsistent_weights_are_rejected(bundle, mutation):
    root, base, m = bundle
    state = TaberoDSRLActor(ActorContract(**m["actor_contract"])).state_dict()
    key = next(iter(state))
    if mutation == "width":
        state[key] = state[key].flatten()
    if mutation == "dtype":
        state[key] = state[key].float()
    if mutation == "nan":
        state[key].flatten()[0] = float("nan")
    if mutation == "missing":
        state.pop(key)
    save_file(state, root / "actor.safetensors")
    m["actor_weights_sha256"] = digest(root / "actor.safetensors")
    (root / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="DSRL actor"):
        TaberoDSRLBundle.load(root, base_checkpoint_dir=base)


@pytest.mark.parametrize("kind", ["missing", "shape", "dtype", "nonfinite"])
def test_bad_observations_fail_before_inference(contract, observation, kind):
    if kind == "missing":
        observation.pop("tactile")
    if kind == "shape":
        observation["tactile"] = torch.zeros(9, 10, 2)
    if kind == "dtype":
        observation["state"] = observation["state"].double()
    if kind == "nonfinite":
        observation["state"][0] = float("inf")
    with pytest.raises((ValueError, KeyError)):
        TaberoDSRLActor(contract).noise(observation)


def test_no_state_base_does_not_disable_actor_state(bundle, observation):
    root, base, _ = bundle
    loaded = TaberoDSRLBundle.load(root, base_checkpoint_dir=base)

    @dataclasses.dataclass
    class Model:
        action_dim: int = 13
        action_horizon: int = 6
        discrete_state_input: bool = True

    @dataclasses.dataclass
    class Assets:
        asset_id: str = "wrong"

    @dataclasses.dataclass
    class Data:
        assets: Assets

    @dataclasses.dataclass
    class Config:
        name: str
        model: Model
        data: Data

    cfg = loaded.configure_base(Config("synthetic", Model(), Data(Assets())))
    assert cfg.model.discrete_state_input is False
    assert cfg.data.assets.asset_id == "norm"
    assert loaded.actor.contract.state_dim == 7
    original = loaded.actor.mean(observation)
    changed = {**observation, "state": observation["state"] + torch.arange(7)}
    assert not torch.equal(original, loaded.actor.mean(changed))


def test_policy_keeps_original_observation_and_noise(contract, observation):
    actor = TaberoDSRLActor(contract)
    calls = []
    base = SimpleNamespace(metadata={}, infer=lambda obs, noise: calls.append((obs, noise)))
    TaberoDSRLPolicy(base, actor).infer(observation)
    assert calls[0][0] is observation
    assert torch.equal(calls[0][1], actor.noise(observation))


def test_stable_base_rejects_replacement(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"original")
    with pytest.raises(ValueError, match="changed"), policy_config._stable_dsrl_checkpoint(path):  # noqa: PT012, SLF001
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"new")
        replacement.replace(path)


class _NoiseCapturingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.received_noise = None

    def sample_actions(self, device, observation, **kwargs):
        del device, observation
        self.received_noise = kwargs["noise"]
        return torch.zeros(1, 1, 1)


def test_policy_preserves_torch_noise_dtype_without_numpy_roundtrip(monkeypatch):
    model = _NoiseCapturingModel()
    monkeypatch.setattr(policy_module._model.Observation, "from_dict", staticmethod(lambda inputs: inputs))  # noqa: SLF001
    base_policy = policy_module.Policy(model, is_pytorch=True, pytorch_device="cpu")
    noise = torch.zeros(1, 50, 32, dtype=torch.bfloat16)

    base_policy.infer({"state": np.zeros(7, dtype=np.float32)}, noise=noise)

    assert model.received_noise is noise
    assert model.received_noise.dtype == torch.bfloat16


def test_policy_keeps_existing_numpy_noise_conversion_and_2d_batching(monkeypatch):
    model = _NoiseCapturingModel()
    monkeypatch.setattr(policy_module._model.Observation, "from_dict", staticmethod(lambda inputs: inputs))  # noqa: SLF001
    base_policy = policy_module.Policy(model, is_pytorch=True, pytorch_device="cpu")
    noise = np.zeros((50, 32), dtype=np.float32)

    base_policy.infer({"state": np.zeros(7, dtype=np.float32)}, noise=noise)

    assert isinstance(model.received_noise, torch.Tensor)
    assert model.received_noise.shape == (1, 50, 32)
    assert model.received_noise.dtype == torch.float32


def test_policy_rejects_torch_noise_for_jax_model_with_clear_error(monkeypatch):
    class FakeJaxModel:
        def sample_actions(self, rng, observation, **kwargs):
            del rng, observation, kwargs
            return np.zeros((1, 1, 1), dtype=np.float32)

    monkeypatch.setattr(policy_module.nnx_utils, "module_jit", lambda function: function)
    base_policy = policy_module.Policy(FakeJaxModel())

    with pytest.raises(TypeError, match=r"torch.*JAX"):
        base_policy.infer(
            {"state": np.zeros(7, dtype=np.float32)},
            noise=torch.zeros(1, 50, 32),
        )


def test_loader_uses_bundle_steps_and_preserves_no_state(bundle, monkeypatch):
    root, base, _ = bundle
    loaded = TaberoDSRLBundle.load(root, base_checkpoint_dir=base)
    group = SimpleNamespace(inputs=(), outputs=())
    data = SimpleNamespace(asset_id="norm", use_quantile_norm=True, data_transforms=group, model_transforms=group)
    model = SimpleNamespace(paligemma_with_expert=SimpleNamespace(to_bfloat16_for_selected_params=lambda _: None))
    settings = SimpleNamespace(load_pytorch=lambda *_: model)
    config = SimpleNamespace(
        model=settings, data=SimpleNamespace(create=lambda *_: data), assets_dirs=(), policy_metadata={}
    )
    monkeypatch.setattr(loaded, "configure_base", lambda _: config)
    monkeypatch.setattr(TaberoDSRLBundle, "load", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(policy_config._checkpoints, "load_norm_stats", lambda *_args: {})  # noqa: SLF001
    calls = []
    policy_factory = policy_config._policy  # noqa: SLF001
    monkeypatch.setattr(
        policy_factory, "Policy", lambda *_args, **kwargs: calls.append(kwargs) or SimpleNamespace(metadata={})
    )
    result = policy_config.create_trained_policy(config, base, dsrl_bundle_path=root, pytorch_device="cpu")
    assert isinstance(result, TaberoDSRLPolicy)
    assert calls[0]["sample_kwargs"]["num_steps"] == 4
    with pytest.raises(ValueError, match="requires num_steps=4"):
        policy_config.create_trained_policy(config, base, dsrl_bundle_path=root, sample_kwargs={"num_steps": 10})


def test_bundle_rejects_forged_no_state_setting(bundle):
    root, base, manifest = bundle
    manifest["base"]["model"]["discrete_state_input"] = True
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"config\.json"):
        TaberoDSRLBundle.load(root, base_checkpoint_dir=base)


def test_disabled_actor_ignores_missing_or_changed_state(contract, observation):
    actor = TaberoDSRLActor(
        dataclasses.replace(contract, use_state=False, feature_order="ordered_images_tactile")
    ).eval()
    assert not hasattr(actor, "actor_state_encoder")
    original = actor.noise(observation)
    changed = {**observation, "state": torch.full((7,), float("nan"))}
    missing = {k: v for k, v in observation.items() if k != "state"}
    assert torch.equal(original, actor.noise(changed))
    assert torch.equal(original, actor.noise(missing))
    assert actor.dsrl_action_noise_net.shared_net[0].in_features == 32


def test_opposite_state_mode_checkpoint_is_rejected(bundle):
    root, base, manifest = bundle
    manifest["actor_contract"]["use_state"] = False
    manifest["actor_contract"]["feature_order"] = "ordered_images_tactile"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest mismatch"):
        TaberoDSRLBundle.load(root, base_checkpoint_dir=base)
