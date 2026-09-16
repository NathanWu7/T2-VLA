"""Configuration-driven DSRL actor bundles for PyTorch OpenPI serving."""

from collections.abc import Mapping
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors.torch import load
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _bundle_filename(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Bundle artifacts must be local filenames.")
    return root / name


@dataclasses.dataclass(frozen=True)
class ActorContract:
    """Inference architecture and raw observation contract exported from training."""

    use_state: bool
    image_keys: list[str]
    image_shapes: list[list[int]]
    state_key: str
    state_dim: int
    tactile_key: str
    tactile_shape: list[int]
    image_latent_dim: int
    state_latent_dim: int
    tactile_latent_dim: int
    hidden_dims: list[int]
    noise_dim: int
    horizon: int
    num_steps: int
    dtype: str
    image_preprocessing: str
    tactile_processing: str
    feature_order: str

    def __post_init__(self):
        if type(self.use_state) is not bool:
            raise ValueError("Actor use_state must be boolean.")
        for field in (
            "state_dim",
            "image_latent_dim",
            "state_latent_dim",
            "tactile_latent_dim",
            "noise_dim",
            "horizon",
            "num_steps",
        ):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer.")
        if not self.hidden_dims or any(type(x) is not int or x <= 0 for x in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers.")
        if not 1 <= len(self.image_keys) <= 3 or len(self.image_keys) != len(self.image_shapes):
            raise ValueError("image keys and shapes must describe 1 to 3 ordered views.")
        keys = [*self.image_keys, self.state_key, self.tactile_key]
        if any(not isinstance(k, str) or not k for k in keys) or len(set(keys)) != len(keys):
            raise ValueError("Observation keys must be nonempty and distinct.")
        for shape in self.image_shapes:
            if len(shape) != 3 or shape[-1] != 3 or any(type(x) is not int or x <= 0 for x in shape):
                raise ValueError("Image shapes must be positive HWC RGB shapes.")
        if (
            len(self.tactile_shape) != 3
            or self.tactile_shape[0] != 9
            or self.tactile_shape[2] != 2
            or any(type(x) is not int or x <= 0 for x in self.tactile_shape)
        ):
            raise ValueError("This training TCN requires [9, markers, 2] tactile input.")
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("Actor dtype must be bfloat16 or float32.")
        # These identify implemented training semantics, not freely selectable transforms.
        if self.image_preprocessing != "uint8_bilinear64_align_false_minus_one_one":
            raise ValueError("Unsupported image preprocessing.")
        if self.tactile_processing != "reference_plus_history8_no_difference_causal_tcn2_kernel3":
            raise ValueError("Unsupported tactile processing.")
        expected_order = "state_ordered_images_tactile" if self.use_state else "ordered_images_tactile"
        if self.feature_order != expected_order:
            raise ValueError("Unsupported feature order.")

    @property
    def tactile_input_dim(self):
        return self.tactile_shape[1] * self.tactile_shape[2]

    @property
    def feature_dim(self):
        return (
            (self.state_latent_dim if self.use_state else 0)
            + len(self.image_keys) * self.image_latent_dim
            + self.tactile_latent_dim
        )


class _GaussianPolicy(nn.Module):
    def __init__(self, contract: ActorContract):
        super().__init__()
        layers = []
        width = contract.feature_dim
        for hidden in contract.hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.ReLU()])
            width = hidden
        self.shared_net = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(width, contract.noise_dim)
        self.log_std_layer = nn.Linear(width, contract.noise_dim)

    def forward(self, features):
        hidden = self.shared_net(features)
        return self.mean_layer(hidden), self.log_std_layer(hidden)


class _ImageEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        layers = []
        for i in range(4):
            layers.extend([nn.Conv2d(3 if i == 0 else 32, 32, 3, stride=2 if i == 0 else 1, padding=1), nn.ReLU()])
        self.encoder = nn.Sequential(*layers)
        self.bottleneck = nn.Sequential(nn.Flatten(), nn.Linear(32768, latent_dim), nn.LayerNorm(latent_dim), nn.Tanh())

    def forward(self, images):
        batch, views, channels, height, width = images.shape
        features = self.bottleneck(self.encoder(images.reshape(batch * views, channels, height, width)))
        return features.reshape(batch, -1)


class _StateEncoder(nn.Module):
    def __init__(self, state_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim, latent_dim), nn.LayerNorm(latent_dim), nn.Tanh())

    def forward(self, state):
        return self.encoder(state)


class _TactileTCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.kernels = nn.ModuleList(nn.Linear(input_dim, output_dim) for _ in range(3))
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, tactile):
        batch, steps, dim = tactile.shape
        padded = torch.cat([tactile.new_zeros(batch, 2, dim), tactile], dim=1)
        result = None
        for i, kernel in enumerate(self.kernels):
            projected = kernel(padded[:, 2 - i : 2 - i + steps, :])
            result = projected if result is None else result + projected
        return F.silu(result + (tactile if self.residual_proj is None else self.residual_proj(tactile)))


class _TactileEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.blocks = nn.ModuleList([_TactileTCNBlock(input_dim, latent_dim), _TactileTCNBlock(latent_dim, latent_dim)])
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, tactile):
        for block in self.blocks:
            tactile = block(tactile)
        return self.out_proj(tactile[:, -1, :])


class TaberoDSRLActor(nn.Module):
    def __init__(self, contract: ActorContract):
        super().__init__()
        self.contract = contract
        self.dsrl_action_noise_net = _GaussianPolicy(contract)
        self.actor_image_encoder = _ImageEncoder(contract.image_latent_dim)
        if contract.use_state:
            self.actor_state_encoder = _StateEncoder(contract.state_dim, contract.state_latent_dim)
        self.actor_tactile_encoder = _TactileEncoder(contract.tactile_input_dim, contract.tactile_latent_dim)
        self.to(dtype=getattr(torch, contract.dtype))

    @property
    def device(self):
        return next(self.parameters()).device

    def preprocess(self, observation: Mapping[str, Any]):
        c = self.contract

        def tensor(key, shape, dtype):
            if key not in observation:
                raise KeyError(f"DSRL observation missing {key}.")
            value = torch.as_tensor(observation[key])
            if tuple(value.shape) != tuple(shape) or value.dtype != dtype:
                raise ValueError(f"DSRL {key}: expected {shape}/{dtype}, got {tuple(value.shape)}/{value.dtype}.")
            if not torch.isfinite(value).all():
                raise ValueError(f"DSRL {key} contains nonfinite values.")
            return value.to(self.device)

        images = []
        for key, shape in zip(c.image_keys, c.image_shapes, strict=True):
            image = tensor(key, shape, torch.uint8).float().permute(2, 0, 1)[None] / 255.0
            images.append(F.interpolate(image, size=(64, 64), mode="bilinear", align_corners=False) * 2 - 1)
        dtype = getattr(torch, c.dtype)
        state = tensor(c.state_key, [c.state_dim], torch.float32)[None].to(dtype) if c.use_state else None
        tactile = tensor(c.tactile_key, c.tactile_shape, torch.float32).reshape(1, 9, c.tactile_input_dim).to(dtype)
        return torch.stack(images, dim=1).to(dtype), state, tactile

    @torch.no_grad()
    def features(self, observation):
        images, state, tactile = self.preprocess(observation)
        features = [self.actor_image_encoder(images), self.actor_tactile_encoder(tactile)]
        if self.contract.use_state:
            features.insert(0, self.actor_state_encoder(state))
        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def mean(self, observation):
        return self.dsrl_action_noise_net(self.features(observation))[0]

    @torch.no_grad()
    def noise(self, observation):
        return self.mean(observation).tanh()[:, None, :].expand(-1, self.contract.horizon, -1)

    def forward(self, observation):
        return self.noise(observation)


@dataclasses.dataclass
class TaberoDSRLBundle:
    manifest: dict
    actor: TaberoDSRLActor

    @classmethod
    def load(cls, path, *, base_checkpoint_dir, base_model_sha256=None):
        root, base = Path(path), Path(base_checkpoint_dir)
        content = (root / "manifest.json").read_bytes()
        manifest = json.loads(content)
        required = {
            "format",
            "algorithm",
            "task_id",
            "global_step",
            "is_final",
            "source",
            "base",
            "actor_contract",
            "actor_weights",
            "actor_weights_sha256",
            "actor_shapes",
        }
        if (
            set(manifest) != required
            or manifest["format"] != "tabero_dsrl_t2vla"
            or manifest["algorithm"] != "dsrl-sac"
        ):
            raise ValueError("Unsupported DSRL manifest; re-export using the current exporter.")
        for key in ("task_id", "global_step"):
            if type(manifest[key]) is not int or manifest[key] < 0:
                raise ValueError(f"{key} must be a nonnegative integer.")
        if type(manifest["is_final"]) is not bool:
            raise ValueError("is_final must be boolean.")
        source = manifest["source"]
        if set(source) != {"checkpoint_sha256", "config_sha256", "observation_sha256", "metadata", "semantics"}:
            raise ValueError("Source provenance is incomplete.")
        for key in ("checkpoint_sha256", "config_sha256", "observation_sha256"):
            _require_sha256(source[key], key)
        base_contract = manifest["base"]
        expected_base_keys = {
            "model_sha256",
            "norm_sha256",
            "norm_asset_id",
            "config_name",
            "model",
            "use_quantile_norm",
        }
        if set(base_contract) != expected_base_keys:
            raise ValueError("Base contract is incomplete.")
        for key in ("model_sha256", "norm_sha256"):
            _require_sha256(base_contract[key], key)
        actual = base_model_sha256 or _sha256(base / "model.safetensors")
        if actual != base_contract["model_sha256"]:
            raise ValueError("DSRL base model SHA-256 mismatch.")
        asset = Path(base_contract["norm_asset_id"])
        if asset.is_absolute() or ".." in asset.parts:
            raise ValueError("Normalization asset must remain inside checkpoint assets.")
        base_settings = json.loads((base / "config.json").read_bytes())
        if base_settings.get("config_name") != base_contract["config_name"] or any(
            type(base_settings.get(k)) is not type(v) or base_settings.get(k) != v
            for k, v in base_contract["model"].items()
        ):
            raise ValueError("DSRL manifest settings disagree with the base config.json.")
        if type(base_contract["use_quantile_norm"]) is not bool:
            raise ValueError("Normalization mode must be boolean.")
        norm = base / "assets" / asset / "norm_stats.json"
        if _sha256(norm) != base_contract["norm_sha256"]:
            raise ValueError("DSRL normalization SHA-256 mismatch.")
        c = ActorContract(**manifest["actor_contract"])
        if c.horizon != base_contract["model"]["action_horizon"] or c.noise_dim != base_contract["model"]["action_dim"]:
            raise ValueError("Noise dimensions do not match the base model.")
        actor_path = _bundle_filename(root, manifest["actor_weights"])
        actor_bytes = actor_path.read_bytes()
        if hashlib.sha256(actor_bytes).hexdigest() != _require_sha256(manifest["actor_weights_sha256"], "actor hash"):
            raise ValueError("DSRL actor SHA-256 mismatch.")
        state = load(actor_bytes)
        actor = TaberoDSRLActor(c)
        expected = actor.state_dict()
        if set(state) != set(expected) or manifest["actor_shapes"] != {k: list(v.shape) for k, v in expected.items()}:
            raise ValueError("DSRL actor tensor manifest mismatch.")
        for key, value in state.items():
            if (
                value.shape != expected[key].shape
                or value.dtype != expected[key].dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"DSRL actor tensor mismatch or nonfinite value: {key}.")
        actor.load_state_dict(state, strict=True)
        actor.eval()
        if content != (root / "manifest.json").read_bytes() or _sha256(actor_path) != manifest["actor_weights_sha256"]:
            raise ValueError("DSRL bundle changed during loading.")
        return cls(manifest, actor)

    def configure_base(self, train_config):
        """Honor exported VLA settings independently of the DSRL state encoder."""
        base = self.manifest["base"]
        if train_config.name != base["config_name"]:
            raise ValueError("DSRL base config name mismatch.")
        fields = {f.name for f in dataclasses.fields(train_config.model)}
        if not set(base["model"]) <= fields:
            raise ValueError("Unknown exported VLA model setting.")
        return dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, **base["model"]),
            data=dataclasses.replace(
                train_config.data, assets=dataclasses.replace(train_config.data.assets, asset_id=base["norm_asset_id"])
            ),
        )


class TaberoDSRLPolicy:
    def __init__(self, base_policy, actor):
        self._base_policy, self._actor = base_policy, actor

    def infer(self, obs):
        return self._base_policy.infer(obs, noise=self._actor.noise(obs))

    @property
    def metadata(self):
        return {**self._base_policy.metadata, "dsrl_actor_contract": dataclasses.asdict(self._actor.contract)}
