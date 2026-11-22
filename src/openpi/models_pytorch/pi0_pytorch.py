import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.tactile_encoder_pytorch import (
    MarkerTokenEncoder,
    TactilePrefixCrossAttention,
    TactileTCNEncoder,
)


_CONTACT_UNSET = object()


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        self.marker_token_encoder = None
        self.tactile_cross_attn = None
        self._init_tactile_prefix_encoder()

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def _init_tactile_prefix_encoder(self):
        if getattr(self.config, "tactile_fusion", "none") == "taccorl":
            self._init_taccorl_modules()
            # TacCoRL has its own multi-token encoder; never instantiate the
            # legacy single-token tactile_prefix_encoder alongside it.
            self.tactile_prefix_encoder = None
            return

        tactile_streams = tuple(getattr(self.config, "tactile_streams", ()) or ())
        tactile_prefix_dim_in = getattr(self.config, "tactile_prefix_dim_in", None)
        if "tactile_prefix" not in tactile_streams or not tactile_prefix_dim_in or tactile_prefix_dim_in <= 0:
            self.tactile_prefix_encoder = None
            return

        if getattr(self.config, "tactile_prefix_encoder_type", None) != "tcn":
            raise ValueError("Only tactile_prefix_encoder_type='tcn' is supported for Tabero PyTorch serving.")

        history_len = getattr(self.config, "tactile_prefix_history", None)
        has_reference_frame = getattr(self.config, "tactile_prefix_use_reference_frame", None)
        diff_from_reference = getattr(self.config, "tactile_prefix_diff_from_reference", None)
        if history_len is None:
            raise ValueError("tactile_prefix_history is required for TCN tactile.")
        if has_reference_frame is None:
            raise ValueError("tactile_prefix_use_reference_frame is required for TCN tactile.")
        if diff_from_reference is None:
            raise ValueError("tactile_prefix_diff_from_reference is required for TCN tactile.")

        steps = history_len + 1 if has_reference_frame else history_len
        if tactile_prefix_dim_in % steps != 0:
            raise ValueError(
                "tactile_prefix_dim_in must be divisible by the effective history length; "
                f"got dim={tactile_prefix_dim_in}, steps={steps}."
            )

        prefix_width = _gemma.get_config(self.config.paligemma_variant).width
        self.tactile_prefix_encoder = TactileTCNEncoder(
            input_dim=tactile_prefix_dim_in // steps,
            hidden_dim=2 * prefix_width,
            output_dim=prefix_width,
            history_len=history_len,
            has_reference_frame=has_reference_frame,
            diff_from_reference=diff_from_reference,
        )

    def _init_taccorl_modules(self):
        """Build the PyTorch modules corresponding to JAX TacCoRL modules."""
        tactile_type = getattr(self.config, "tactile_type", None)
        # Avoid importing the enum here: training configs may use a compatible
        # enum object, and the required marker fields below give clearer errors.
        if tactile_type is None or (
            getattr(tactile_type, "name", None) != "EXPERT_HIS_C_FUT"
            and str(tactile_type).split(".")[-1] != "EXPERT_HIS_C_FUT"
        ):
            raise ValueError("taccorl requires tactile_type=EXPERT_HIS_C_FUT.")

        streams = set(tuple(getattr(self.config, "tactile_streams", ()) or ()))
        if streams != {"tactile_prefix", "tactile_suffix"}:
            raise ValueError("taccorl requires both tactile_prefix and tactile_suffix in tactile_streams.")

        required = (
            "tactile_prefix_dim_in",
            "tactile_prefix_history",
            "tactile_prefix_use_reference_frame",
            "tactile_prefix_diff_from_reference",
            "tactile_prefix_num_tokens",
            "tactile_suffix_num_tokens",
            "tactile_prefix_cross_attn",
            "tactile_cross_attn_heads",
        )
        missing = [name for name in required if not hasattr(self.config, name) or getattr(self.config, name) is None]
        if missing:
            raise ValueError(f"taccorl requires {', '.join(missing)}.")
        if getattr(self.config, "tactile_suffix_placement", "suffix") != "suffix":
            raise ValueError("taccorl requires tactile_suffix_placement='suffix'.")

        prefix_dim_in = self.config.tactile_prefix_dim_in
        history_len = self.config.tactile_prefix_history
        has_reference = self.config.tactile_prefix_use_reference_frame
        diff_from_reference = self.config.tactile_prefix_diff_from_reference
        if history_len <= 0:
            raise ValueError("taccorl requires tactile_prefix_history > 0.")
        steps = history_len + int(has_reference)
        if prefix_dim_in <= 0 or prefix_dim_in % steps:
            raise ValueError("taccorl requires tactile_prefix_dim_in divisible by history plus reference frames.")
        input_dim = prefix_dim_in // steps
        if input_dim % 2:
            raise ValueError("taccorl requires tactile_prefix_dim_in = frames * markers * 2.")
        if min(self.config.tactile_prefix_num_tokens, self.config.tactile_suffix_num_tokens) <= 0:
            raise ValueError("taccorl requires positive tactile token counts.")
        if self.config.tactile_cross_attn_heads <= 0:
            raise ValueError("taccorl requires tactile_cross_attn_heads > 0.")

        prefix_width = _gemma.get_config(self.config.paligemma_variant).width
        action_width = _gemma.get_config(self.config.action_expert_variant).width
        self.marker_token_encoder = MarkerTokenEncoder(
            in_dim=input_dim,
            history_len=history_len,
            has_reference_frame=has_reference,
            diff_from_reference=diff_from_reference,
            prefix_width=prefix_width,
            suffix_width=action_width,
            prefix_num_tokens=self.config.tactile_prefix_num_tokens,
            suffix_num_tokens=self.config.tactile_suffix_num_tokens,
        )
        if self.config.tactile_prefix_cross_attn:
            if prefix_width % self.config.tactile_cross_attn_heads:
                raise ValueError("tactile_cross_attn_heads must divide the PaliGemma width.")
            self.tactile_cross_attn = TactilePrefixCrossAttention(
                prefix_width, self.config.tactile_cross_attn_heads
            )

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(
            observation,
            train=train,
            image_noise_std=float(getattr(self.config, "image_noise_std", 0.0)),
        )
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
            getattr(observation, "tactile_prefix", None),
        )

    @staticmethod
    def _get_tactile_contact(observation):
        """Return the per-example contact gate; ``None`` means all gates open."""
        return getattr(observation, "tactile_contact", None)

    def _marker_token_mask(self, tactile_tokens, tactile_contact):
        if tactile_contact is None:
            contact = torch.ones(
                tactile_tokens.shape[0], dtype=torch.bool, device=tactile_tokens.device
            )
        else:
            contact = torch.as_tensor(tactile_contact, device=tactile_tokens.device, dtype=torch.bool)
            if contact.shape != (tactile_tokens.shape[0],):
                raise ValueError(
                    f"tactile_contact must have shape {(tactile_tokens.shape[0],)}, got {tuple(contact.shape)}."
                )
        return contact[:, None].expand(-1, tactile_tokens.shape[1])

    def _encode_marker_tokens(self, tactile_prefix):
        marker_encoder = getattr(self, "marker_token_encoder", None)
        if marker_encoder is None:
            return None
        if tactile_prefix is None:
            raise ValueError("taccorl requires observation.tactile_prefix containing per-frame marker vectors.")
        tactile_prefix = tactile_prefix.to(
            device=marker_encoder.prefix_proj.weight.device,
            dtype=marker_encoder.prefix_proj.weight.dtype,
        )
        return self._apply_checkpoint(marker_encoder, tactile_prefix)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def _align_inference_noise(self, noise, device):
        return noise.to(device=device, dtype=self.action_in_proj.weight.dtype)

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        tactile_prefix=None,
        *,
        marker_tokens=None,
        tactile_contact=_CONTACT_UNSET,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed image/language tokens and optional tactile prefix tokens.

        ``marker_tokens`` and ``tactile_contact`` are keyword-only additions;
        the legacy positional call shape remains unchanged for RLinf clients.
        """
        if getattr(self.config, "tactile_fusion", "none") == "taccorl" and tactile_contact is _CONTACT_UNSET:
            raise ValueError("embed_prefix requires explicit tactile_contact in taccorl mode; pass None for open gates.")
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        if getattr(self, "marker_token_encoder", None) is not None:
            if marker_tokens is None:
                marker_tokens = self._encode_marker_tokens(tactile_prefix)
            tactile_emb = marker_tokens[0].to(device=lang_emb.device, dtype=lang_emb.dtype)
            tactile_mask = self._marker_token_mask(tactile_emb, tactile_contact)
            base_embs = torch.cat(embs, dim=1)
            if self.tactile_cross_attn is not None:
                base_embs = self._apply_checkpoint(
                    self.tactile_cross_attn, base_embs, tactile_emb, tactile_mask
                )
            embs = [base_embs, tactile_emb]
            pad_masks.append(tactile_mask)
            att_masks += [0] * tactile_emb.shape[1]
        elif self.tactile_prefix_encoder is not None and tactile_prefix is not None:
            tactile_prefix = tactile_prefix.to(
                device=lang_emb.device,
                dtype=self.tactile_prefix_encoder.out_proj.weight.dtype,
            )

            def tactile_embed_func(tactile):
                return self.tactile_prefix_encoder(tactile)

            tactile_emb = self._apply_checkpoint(tactile_embed_func, tactile_prefix)
            tactile_emb = tactile_emb.to(dtype=lang_emb.dtype)[:, None, :]
            embs.append(tactile_emb)
            pad_masks.append(torch.ones(tactile_emb.shape[:2], dtype=torch.bool, device=tactile_emb.device))
            att_masks += [0] * tactile_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(
        self,
        state,
        noisy_actions,
        timestep,
        *,
        tactile_prefix=None,
        marker_tokens=None,
        tactile_contact=_CONTACT_UNSET,
    ):
        """Embed tactile markers, state, noisy actions, and timestep.

        TacCoRL marker tokens are prepended before the state/action suffix. The
        extra arguments are keyword-only so legacy callers keep their signature.
        """
        if getattr(self.config, "tactile_fusion", "none") == "taccorl" and tactile_contact is _CONTACT_UNSET:
            raise ValueError("embed_suffix requires explicit tactile_contact in taccorl mode; pass None for open gates.")
        embs = []
        pad_masks = []
        att_masks = []

        if getattr(self, "marker_token_encoder", None) is not None:
            if marker_tokens is None:
                marker_tokens = self._encode_marker_tokens(tactile_prefix)
            tactile_emb = marker_tokens[1].to(device=noisy_actions.device, dtype=noisy_actions.dtype)
            tactile_mask = self._marker_token_mask(tactile_emb, tactile_contact)
            embs.append(tactile_emb)
            pad_masks.append(tactile_mask)
            # The first tactile token starts the suffix causal block; the
            # remaining marker tokens share that block, matching JAX.
            att_masks += [1] + ([0] * (tactile_emb.shape[1] - 1))

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_dtype = torch.bool if getattr(self, "marker_token_encoder", None) is not None else embs.dtype
        att_masks = torch.tensor(att_masks, dtype=att_dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state, tactile_prefix = self._preprocess_observation(
            observation, train=True
        )

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        if getattr(self.config, "tactile_fusion", "none") == "taccorl":
            tactile_contact = self._get_tactile_contact(observation)
            marker_tokens = self._encode_marker_tokens(tactile_prefix)
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                tactile_prefix,
                marker_tokens=marker_tokens,
                tactile_contact=tactile_contact,
            )
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                state,
                x_t,
                time,
                tactile_prefix=tactile_prefix,
                marker_tokens=marker_tokens,
                tactile_contact=tactile_contact,
            )
        else:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, tactile_prefix
            )
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        else:
            noise = self._align_inference_noise(noise, device)

        images, img_masks, lang_tokens, lang_masks, state, tactile_prefix = self._preprocess_observation(
            observation, train=False
        )

        taccorl = getattr(self.config, "tactile_fusion", "none") == "taccorl"
        if taccorl:
            tactile_contact = self._get_tactile_contact(observation)
            marker_tokens = self._encode_marker_tokens(tactile_prefix)
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                tactile_prefix,
                marker_tokens=marker_tokens,
                tactile_contact=tactile_contact,
            )
        else:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, tactile_prefix
            )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            if not taccorl:
                v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
            else:
                v_t = self.denoise_step(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    expanded_time,
                    marker_tokens=marker_tokens,
                    tactile_prefix=tactile_prefix,
                    tactile_contact=tactile_contact,
                )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        *,
        marker_tokens=None,
        tactile_prefix=None,
        tactile_contact=_CONTACT_UNSET,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        if getattr(self.config, "tactile_fusion", "none") == "taccorl":
            if tactile_contact is _CONTACT_UNSET:
                raise ValueError(
                    "denoise_step requires explicit tactile_contact in taccorl mode; pass None for open gates."
                )
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                state,
                x_t,
                timestep,
                marker_tokens=marker_tokens,
                tactile_prefix=tactile_prefix,
                tactile_contact=tactile_contact,
            )
        else:
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        if getattr(self, "marker_token_encoder", None) is not None:
            # Match JAX's query-side gate: closed tactile suffix rows cannot
            # attend any prefix keys, while valid state/action rows retain the
            # regular prefix padding mask.
            prefix_pad_2d_masks = prefix_pad_2d_masks & suffix_pad_masks[:, :, None]

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
