"""Tabero RLT Stage 1/2 inference components for PyTorch serving."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from safetensors.torch import load_file
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


def checkpoint_sha256(path: str | Path) -> str:
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "model.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Base model checkpoint file not found: {checkpoint_path}")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class TaberoRLTManifest:
    format: str
    action_space: str
    normalized_action_bound: float
    z_dim: int
    proprio_dim: int
    action_dim: int
    num_action_chunks: int
    ref_num_action_chunks: int
    rlt_input_dim: int
    rlt_embed_dim: int
    rlt_num_rl_tokens: int
    rlt_prefix_seq_len: int
    rlt_num_layers: int
    rlt_num_heads: int
    rlt_mlp_ratio: float
    rlt_image_only: bool
    rlt_use_mask: bool
    encoder_weights: str
    actor_weights: str
    action_parameterization: str
    action_composition: str
    residual_action_bound: float | None = None
    actor_hidden_dim: int = 256
    base_model: str | None = None
    base_config_name: str | None = None
    base_action_horizon: int | None = None
    base_model_action_dim: int | None = None
    base_effective_action_dim: int | None = None
    base_prefix_hidden_dim: int | None = None
    base_norm_asset_id: str | None = None
    base_use_quantile_norm: bool | None = None
    rlt_use_normalized_proprio: bool = True
    state_indices: list[int] | None = None
    reference_num_steps: int = 10
    reference_sampling_method: str = "flow_ode"
    base_model_sha256: str | None = None

    @classmethod
    def from_dict(cls, values: dict[str, Any], *, action_parameterization: str | None = None) -> TaberoRLTManifest:
        values = dict(values)
        declared = values.get("action_parameterization")
        if action_parameterization is not None and declared is not None and declared != action_parameterization:
            raise ValueError("RLT bundle action_parameterization conflicts with explicit loading argument.")
        mode = declared if declared is not None else action_parameterization
        if mode not in ("absolute", "residual"):
            raise ValueError("RLT bundle requires explicit action semantics: absolute or residual.")
        if declared is None:
            # A caller must declare the semantics of unlabelled bundles. Never
            # override contradictory composition/bound metadata.
            values["action_parameterization"] = mode
            values.setdefault("action_composition", "direct" if mode == "absolute" else "reference_plus_bounded_delta")
            if mode == "residual":
                values.setdefault("residual_action_bound", values.get("normalized_action_bound"))
        elif "action_composition" not in values or (mode == "residual" and "residual_action_bound" not in values):
            raise ValueError("RLT bundle has incomplete action semantics.")
        known = {field.name for field in dataclasses.fields(cls)}
        filtered = {key: value for key, value in values.items() if key in known}
        manifest = cls(**filtered)
        if manifest.format != "tabero_rlt_t2vla":
            raise ValueError(f"Unsupported Tabero RLT bundle format: {manifest.format!r}.")
        if manifest.action_space != "model_normalized":
            raise ValueError(
                f"Tabero RLT T2-VLA inference requires action_space='model_normalized'; got {manifest.action_space!r}."
            )
        if not math.isfinite(manifest.normalized_action_bound) or manifest.normalized_action_bound <= 0:
            raise ValueError("normalized_action_bound must be finite and positive.")
        if mode == "residual":
            if (
                manifest.action_composition != "reference_plus_bounded_delta"
                or manifest.residual_action_bound != manifest.normalized_action_bound
            ):
                raise ValueError("RLT bundle residual composition or bound mismatch.")
        elif manifest.action_composition != "direct" or manifest.residual_action_bound is not None:
            raise ValueError("RLT bundle absolute actions require direct composition and no residual bound.")
        positive_dimensions = {
            "z_dim": manifest.z_dim,
            "proprio_dim": manifest.proprio_dim,
            "action_dim": manifest.action_dim,
            "num_action_chunks": manifest.num_action_chunks,
            "ref_num_action_chunks": manifest.ref_num_action_chunks,
            "actor_hidden_dim": manifest.actor_hidden_dim,
            "rlt_input_dim": manifest.rlt_input_dim,
            "rlt_embed_dim": manifest.rlt_embed_dim,
            "rlt_num_rl_tokens": manifest.rlt_num_rl_tokens,
            "rlt_prefix_seq_len": manifest.rlt_prefix_seq_len,
            "rlt_num_layers": manifest.rlt_num_layers,
            "rlt_num_heads": manifest.rlt_num_heads,
            "reference_num_steps": manifest.reference_num_steps,
        }
        invalid = [name for name, value in positive_dimensions.items() if value <= 0]
        if invalid:
            raise ValueError(f"RLT manifest dimensions must be positive; invalid={invalid}.")
        if manifest.ref_num_action_chunks < manifest.num_action_chunks:
            raise ValueError("ref_num_action_chunks must be >= num_action_chunks.")
        if manifest.z_dim != manifest.rlt_embed_dim * manifest.rlt_num_rl_tokens:
            raise ValueError("z_dim must equal rlt_embed_dim * rlt_num_rl_tokens.")
        if manifest.rlt_embed_dim % manifest.rlt_num_heads != 0:
            raise ValueError("rlt_embed_dim must be divisible by rlt_num_heads.")
        if not math.isfinite(manifest.rlt_mlp_ratio) or manifest.rlt_mlp_ratio <= 0:
            raise ValueError("rlt_mlp_ratio must be finite and positive.")
        if not manifest.rlt_use_normalized_proprio:
            raise ValueError("Tabero RLT T2-VLA inference requires normalized proprio.")
        if manifest.state_indices is not None:
            if len(manifest.state_indices) != manifest.proprio_dim:
                raise ValueError("state_indices length must equal proprio_dim.")
            if any(index < 0 for index in manifest.state_indices) or len(set(manifest.state_indices)) != len(
                manifest.state_indices
            ):
                raise ValueError("state_indices must be unique non-negative integers.")
        if manifest.reference_sampling_method != "flow_ode":
            raise ValueError("Tabero RLT T2-VLA inference requires reference sampling method 'flow_ode'.")
        if manifest.base_model_sha256 is not None and (
            len(manifest.base_model_sha256) != 64
            or any(character not in "0123456789abcdef" for character in manifest.base_model_sha256.lower())
        ):
            raise ValueError("base_model_sha256 must be a SHA-256 hex digest.")
        return manifest

    def to_dict(self) -> dict[str, Any]:
        values = dataclasses.asdict(self)
        if self.actor_hidden_dim == 256:
            # Omit the default architecture width from serialized metadata.
            values.pop("actor_hidden_dim")
        for key in tuple(values):
            if values[key] is None and key != "state_indices":
                values.pop(key)
        return values


@dataclasses.dataclass(frozen=True)
class TaberoRLTBundle:
    root: Path
    manifest: TaberoRLTManifest

    @classmethod
    def load(cls, path: str | Path, *, action_parameterization: str | None = None) -> TaberoRLTBundle:
        root = Path(path)
        manifest_path = root / "manifest.json"
        manifest = TaberoRLTManifest.from_dict(
            json.loads(manifest_path.read_text()), action_parameterization=action_parameterization
        )
        for filename in (manifest.encoder_weights, manifest.actor_weights):
            weight_path = root / filename
            if not weight_path.is_file():
                raise FileNotFoundError(f"Tabero RLT bundle weight file not found: {weight_path}")
        return cls(root=root, manifest=manifest)

    def load_encoder_state(self, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        state = load_file(self.root / self.manifest.encoder_weights, device=str(device))
        prefix = "encoder."
        if not state or any(not key.startswith(prefix) for key in state):
            raise ValueError("RLT encoder weights must contain only encoder.* tensors.")
        return {key.removeprefix(prefix): value for key, value in state.items()}

    def load_actor_state(self, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        return load_file(self.root / self.manifest.actor_weights, device=str(device))

    def validate_base_policy(
        self,
        base_model: nn.Module,
        *,
        base_model_path: str | Path | None = None,
        base_config_name: str | None = None,
        norm_asset_id: str | None = None,
        use_quantile_norm: bool | None = None,
    ) -> None:
        manifest = self.manifest
        if manifest.base_model_sha256 is not None:
            if base_model_path is None:
                raise ValueError("RLT bundle requires the base checkpoint path for checksum validation.")
            actual_digest = checkpoint_sha256(base_model_path)
            if actual_digest != manifest.base_model_sha256.lower():
                raise ValueError(
                    "RLT bundle base checkpoint SHA-256 mismatch: "
                    f"expected {manifest.base_model_sha256}, got {actual_digest}."
                )
        elif manifest.base_model is not None:
            if base_model_path is None:
                raise ValueError("RLT bundle requires the base checkpoint path for compatibility validation.")
            expected = Path(manifest.base_model).expanduser().resolve()
            actual = Path(base_model_path).expanduser().resolve()
            if actual != expected:
                raise ValueError(f"RLT bundle base checkpoint mismatch: expected {expected}, got {actual}.")
        context_checks = (
            ("base config", manifest.base_config_name, base_config_name),
            ("norm asset", manifest.base_norm_asset_id, norm_asset_id),
            ("quantile normalization", manifest.base_use_quantile_norm, use_quantile_norm),
        )
        for label, expected, actual in context_checks:
            if expected is not None and actual != expected:
                raise ValueError(f"RLT bundle {label} mismatch: expected {expected!r}, got {actual!r}.")

        config = getattr(base_model, "config", None)
        model_checks = (
            ("action horizon", manifest.base_action_horizon, getattr(config, "action_horizon", None)),
            ("model action dimension", manifest.base_model_action_dim, getattr(config, "action_dim", None)),
            (
                "effective action dimension",
                manifest.base_effective_action_dim,
                getattr(config, "effective_action_dim", None),
            ),
        )
        for label, expected, actual in model_checks:
            if expected is not None and actual != expected:
                raise ValueError(f"RLT bundle {label} mismatch: expected {expected}, got {actual}.")

        if manifest.base_prefix_hidden_dim is not None:
            language_model = getattr(
                getattr(getattr(base_model, "paligemma_with_expert", None), "paligemma", None),
                "language_model",
                None,
            )
            hidden_size = getattr(getattr(language_model, "config", None), "hidden_size", None)
            if hidden_size != manifest.base_prefix_hidden_dim:
                raise ValueError(
                    "RLT bundle prefix hidden dimension mismatch: "
                    f"expected {manifest.base_prefix_hidden_dim}, got {hidden_size}."
                )

    def build_encoder(self) -> RLTTokenEncoder:
        manifest = self.manifest
        encoder = RLTTokenEncoder(
            input_dim=manifest.rlt_input_dim,
            embed_dim=manifest.rlt_embed_dim,
            num_rl_tokens=manifest.rlt_num_rl_tokens,
            prefix_seq_len=manifest.rlt_prefix_seq_len,
            num_layers=manifest.rlt_num_layers,
            num_heads=manifest.rlt_num_heads,
            mlp_ratio=manifest.rlt_mlp_ratio,
        )
        encoder.load_state_dict(self.load_encoder_state(), strict=True, assign=True)
        return encoder

    def build_actor(self) -> RLTActorMLP:
        manifest = self.manifest
        input_dim = manifest.num_action_chunks * manifest.action_dim + manifest.z_dim + manifest.proprio_dim
        actor = RLTActorMLP(
            input_dim=input_dim,
            output_dim=manifest.num_action_chunks * manifest.action_dim,
            hidden_dim=manifest.actor_hidden_dim,
            normalized_action_bound=manifest.normalized_action_bound,
        )
        actor.load_state_dict(self.load_actor_state(), strict=True)
        return actor


def sinusoidal_pe_init(seq_len: int, embed_dim: int) -> torch.Tensor:
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / embed_dim))
    pe = torch.zeros(seq_len, embed_dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class GeGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(inputs).chunk(2, dim=-1)
        return x * F.gelu(gate)


class RLTSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        mlp_ratio: float,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        mlp_dim = int(embed_dim * mlp_ratio)
        self.self_norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.Dropout(dropout_rate),
            GeGLU(mlp_dim),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        key_padding_mask = None if mask is None else ~mask.to(device=inputs.device, dtype=torch.bool)
        residual = inputs
        normalized = self.self_norm(inputs)
        attended = self.self_attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        hidden = residual + attended
        return hidden + self.mlp(self.mlp_norm(hidden))


class RLTTokenEncoder(nn.Module):
    """Exact encoder architecture used by RLinf RLT Stage 1."""

    def __init__(
        self,
        *,
        input_dim: int,
        embed_dim: int,
        num_rl_tokens: int,
        prefix_seq_len: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_rl_tokens = int(num_rl_tokens)
        self.prefix_seq_len = int(prefix_seq_len)
        self.input_proj = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()
        self.rl_token_embed = nn.Parameter(sinusoidal_pe_init(self.num_rl_tokens, embed_dim))
        self.prefix_pos_enc = nn.Parameter(sinusoidal_pe_init(self.prefix_seq_len, embed_dim))
        self.rl_token_pos_enc = nn.Parameter(sinusoidal_pe_init(self.num_rl_tokens, embed_dim))
        self.layers = nn.ModuleList(
            [
                RLTSelfAttentionLayer(
                    embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, prefix_embs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        prefix_embs = self.input_proj(prefix_embs)
        seq_len = prefix_embs.shape[-2]
        if seq_len > self.prefix_seq_len:
            raise ValueError(
                f"prefix sequence length {seq_len} exceeds configured prefix_seq_len {self.prefix_seq_len}."
            )
        prefix_pos = self.prefix_pos_enc[:seq_len].to(prefix_embs)
        batch_size = prefix_embs.shape[0]
        rl_tokens = self.rl_token_embed.to(prefix_embs).unsqueeze(0).expand(batch_size, -1, -1)
        rl_tokens = rl_tokens + self.rl_token_pos_enc.to(prefix_embs)
        hidden = torch.cat([prefix_embs + prefix_pos, rl_tokens], dim=1)
        if mask is not None:
            mask = mask.to(device=prefix_embs.device, dtype=torch.bool)
            rl_mask = torch.ones(
                batch_size,
                self.num_rl_tokens,
                device=prefix_embs.device,
                dtype=torch.bool,
            )
            mask = torch.cat([mask, rl_mask], dim=1)
        for layer in self.layers:
            hidden = layer(hidden, mask)
        return hidden[:, -self.num_rl_tokens :]

    def encode_flat(self, prefix_embs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self(prefix_embs, mask).reshape(prefix_embs.shape[0], -1)


class RLTActorMLP(nn.Module):
    """Bounded actor output interpreted according to explicit bundle action semantics."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        normalized_action_bound: float = 4.0,
    ):
        super().__init__()
        self.normalized_action_bound = float(normalized_action_bound)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raw_action = self.actor_mean(self.backbone(inputs))
        bound = self.normalized_action_bound
        return bound * torch.tanh(raw_action / bound)


def make_actor_input(
    *,
    ref_chunk: torch.Tensor,
    z_rl: torch.Tensor,
    proprio: torch.Tensor,
    num_action_chunks: int,
    action_dim: int,
) -> torch.Tensor:
    expected_shape = (num_action_chunks, action_dim)
    if tuple(ref_chunk.shape[-2:]) != expected_shape:
        raise ValueError(f"ref_chunk must end in {expected_shape}, got {tuple(ref_chunk.shape[-2:])}.")
    return torch.cat([ref_chunk.flatten(start_dim=1), z_rl, proprio], dim=-1)


def rlinf_flow_ode_step(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    timestep: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Match the floating-point operation order used for Stage 2 references."""
    if timestep.ndim == 0:
        timestep = timestep.expand(x_t.shape[0])
    if delta.ndim == 0:
        delta = delta.expand(x_t.shape[0])
    timestep = timestep[:, None, None].expand_as(x_t)
    delta = delta[:, None, None].expand_as(x_t)
    x0_pred = x_t - velocity * timestep
    x1_pred = x_t + velocity * (1 - timestep)
    x0_weight = 1 - (timestep - delta)
    x1_weight = timestep - delta
    return x0_pred * x0_weight + x1_pred * x1_weight


def _make_attention_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    cumulative = torch.cumsum(att_masks, dim=1)
    attention = cumulative[:, None, :] <= cumulative[:, :, None]
    valid = pad_masks[:, None, :] * pad_masks[:, :, None]
    return attention & valid


class TaberoRLTPolicyModel(nn.Module):
    """Serve a frozen PI0 reference policy through a Stage 1/2 RLT actor."""

    def __init__(
        self,
        base_model: nn.Module,
        bundle: TaberoRLTBundle,
        encoder: RLTTokenEncoder,
        actor: RLTActorMLP,
    ):
        super().__init__()
        self.base_model = base_model
        self.bundle = bundle
        self.encoder = encoder
        self.actor = actor

    @classmethod
    def from_bundle(
        cls,
        base_model: nn.Module,
        bundle_path: str | Path,
        *,
        action_parameterization: str | None = None,
        base_model_path: str | Path | None = None,
        base_config_name: str | None = None,
        norm_asset_id: str | None = None,
        use_quantile_norm: bool | None = None,
    ) -> TaberoRLTPolicyModel:
        bundle = TaberoRLTBundle.load(bundle_path, action_parameterization=action_parameterization)
        bundle.validate_base_policy(
            base_model,
            base_model_path=base_model_path,
            base_config_name=base_config_name,
            norm_asset_id=norm_asset_id,
            use_quantile_norm=use_quantile_norm,
        )
        return cls(
            base_model=base_model,
            bundle=bundle,
            encoder=bundle.build_encoder(),
            actor=bundle.build_actor(),
        )

    def _select_rlt_prefix(
        self,
        prefix_output: torch.Tensor,
        prefix_mask: torch.Tensor,
        lang_tokens: torch.Tensor | None,
        tactile_token_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.bundle.manifest.rlt_image_only or lang_tokens is None:
            return prefix_output, prefix_mask

        num_image_tokens = prefix_output.shape[1] - lang_tokens.shape[1] - tactile_token_count
        selected_output = prefix_output[:, :num_image_tokens]
        selected_mask = prefix_mask[:, :num_image_tokens]
        if tactile_token_count:
            selected_output = torch.cat([selected_output, prefix_output[:, -tactile_token_count:]], dim=1)
            selected_mask = torch.cat([selected_mask, prefix_mask[:, -tactile_token_count:]], dim=1)
        return selected_output, selected_mask

    @torch.no_grad()
    def sample_actions(
        self,
        device: str | torch.device,
        observation: Any,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        base = self.base_model
        manifest = self.bundle.manifest
        if num_steps is None:
            num_steps = manifest.reference_num_steps
        elif num_steps != manifest.reference_num_steps:
            raise ValueError(
                "num_steps must match the bundle reference_num_steps; "
                f"expected {manifest.reference_num_steps}, got {num_steps}."
            )
        batch_size = observation.state.shape[0]
        device = torch.device(device)
        if noise is None:
            noise = base.sample_noise(
                (batch_size, base.config.action_horizon, base.config.action_dim),
                device,
            )

        images, img_masks, lang_tokens, lang_masks, state, tactile_prefix = base._preprocess_observation(  # noqa: SLF001
            observation,
            train=False,
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = base.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            tactile_prefix,
        )
        prefix_attention = _make_attention_masks(prefix_pad_masks, prefix_att_masks)
        prefix_attention = torch.where(prefix_attention[:, None, :, :], 0.0, -2.3819763e38)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        base.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (prefix_output, _), past_key_values = base.paligemma_with_expert.forward(
            attention_mask=prefix_attention,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        tactile_token_count = int(
            getattr(base, "tactile_prefix_encoder", None) is not None and tactile_prefix is not None
        )
        rlt_prefix, rlt_mask = self._select_rlt_prefix(
            prefix_output,
            prefix_pad_masks,
            lang_tokens,
            tactile_token_count,
        )
        encoder_param = next(self.encoder.parameters())
        rlt_prefix = rlt_prefix.to(device=encoder_param.device, dtype=encoder_param.dtype)
        z_rl = self.encoder.encode_flat(
            rlt_prefix,
            rlt_mask if manifest.rlt_use_mask else None,
        ).to(dtype=torch.float32)

        timesteps = torch.linspace(1, 1 / num_steps, num_steps, device=device)
        timesteps = torch.cat([timesteps, torch.zeros(1, device=device)])
        ref_actions = noise
        for index in range(num_steps):
            timestep = timesteps[index].expand(batch_size)
            velocity = base.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                ref_actions,
                timestep,
            )
            ref_actions = rlinf_flow_ode_step(
                ref_actions,
                velocity,
                timestep,
                timesteps[index] - timesteps[index + 1],
            )

        ref_chunk = ref_actions[:, : manifest.num_action_chunks, : manifest.action_dim].to(
            device=z_rl.device, dtype=torch.float32
        )
        if manifest.state_indices is None:
            proprio = state[:, : manifest.proprio_dim]
        else:
            state_index = torch.as_tensor(manifest.state_indices, device=state.device)
            if int(state_index.max().item()) >= state.shape[-1]:
                raise ValueError(
                    f"state_indices={manifest.state_indices} exceed normalized state width {state.shape[-1]}."
                )
            proprio = state.index_select(-1, state_index)
        proprio = proprio.to(device=z_rl.device, dtype=torch.float32)
        actor_input = make_actor_input(
            ref_chunk=ref_chunk,
            z_rl=z_rl,
            proprio=proprio,
            num_action_chunks=manifest.num_action_chunks,
            action_dim=manifest.action_dim,
        )
        actor_param = next(self.actor.parameters())
        actor_input = actor_input.to(device=actor_param.device, dtype=actor_param.dtype)
        actor_output = self.actor(actor_input).reshape(
            batch_size,
            manifest.num_action_chunks,
            manifest.action_dim,
        )
        # Add once in normalized model space before the output transforms.
        # Do not squash the sum: zero residual must preserve the reference.
        if manifest.action_parameterization == "residual":
            return ref_chunk.to(device=actor_output.device) + actor_output
        return actor_output
