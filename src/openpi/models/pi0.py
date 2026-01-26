import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.models import tactile_encoder as _tactile_encoder
from openpi.shared import array_typing as at
from openpi.shared.tactile_type import TactileType

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.tactile_type = config.tactile_type
        self.tactile_dim = config.tactile_dim
        self.tactile_history = config.tactile_history
        self.tactile_encoder_type = config.tactile_encoder_type
        # 触觉通道选择（"tactile_suffix" / "tactile_prefix" / 两者）。
        self.tactile_streams = tuple(config.tactile_streams)
        # tacforce（tactile_suffix）token 放置位置：默认 suffix（向后兼容），可选 prefix。
        self.tactile_suffix_placement = getattr(config, "tactile_suffix_placement", "suffix")
        # 有效 action 维度：用于 loss 中的 [动作, 力矩] 切分。
        # 允许数据只在前 K 维有意义，其余为 padding。
        self.effective_action_dim = config.effective_action_dim
        # 力 / 触觉 loss 的权重（total_loss = action_loss + tactile_loss_weight * tactile_loss）。
        self.tactile_loss_weight = config.tactile_loss_weight
        # 对 effective_action_dim 之后 padding 维度的 loss 权重（默认 0：忽略 padding；>0：也监督 padding）。
        self.padding_loss_weight = getattr(config, "padding_loss_weight", 0.0)
        # EXPERT_HIS_C_FUT 的 loss 计算模式（默认 split，向后兼容）。
        self.expert_his_c_fut_loss_mode = getattr(config, "expert_his_c_fut_loss_mode", "split")

        # 仅禁止尚未实现的 TactileType 组合；允许 pi05 + EXPERT_HIS_C_FUT。
        if self.pi05 and self.tactile_type not in (TactileType.NO, TactileType.EXPERT_HIS_C_FUT):
            raise ValueError(
                f"TactileType {self.tactile_type} is not supported for Pi05; "
                "only TactileType.NO and TactileType.EXPERT_HIS_C_FUT are allowed."
            )
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # Tactile 编码器（历史 encoder）——仅在需要 tactile token 时创建权重。
        # 对于只想做 loss 拆分、不需要 tactile token 的配置，可以把 tactile_dim_in/tactile_prefix_dim_in 设为 0 或 None，
        # 这样既不会创建新的编码器权重，也不会影响已有 checkpoint 加载。
        #
        self.tactile_prefix_encoder = None
        self.tactile_suffix_encoder = None
        # 当把 tactile_suffix token 放到 prefix 序列时，可能需要做宽度对齐（action_expert_width -> paligemma_width）。
        self.tactile_suffix_to_prefix_proj = None
        if self.tactile_type is TactileType.EXPERT_HIS_C_FUT:
            use_prefix = "tactile_prefix" in self.tactile_streams
            use_suffix = "tactile_suffix" in self.tactile_streams

            # Encoder-prefix 触觉（例如 Tabero marker motion，经 TCN 编码）。
            if use_prefix and config.tactile_prefix_dim_in is not None and config.tactile_prefix_dim_in > 0:
                prefix_width = paligemma_config.width
                if config.tactile_prefix_encoder_type is None:
                    raise ValueError(
                        "Prefix tactile stream is enabled but Pi0Config.tactile_prefix_encoder_type is None. "
                        "Please set tactile_prefix_encoder_type explicitly (e.g., 'tcn' or 'mlp')."
                    )
                if config.tactile_prefix_use_reference_frame is None:
                    raise ValueError(
                        "Prefix tactile stream is enabled but Pi0Config.tactile_prefix_use_reference_frame is None. "
                        "Please set tactile_prefix_use_reference_frame explicitly (True/False)."
                    )
                if config.tactile_prefix_diff_from_reference is None:
                    raise ValueError(
                        "Prefix tactile stream is enabled but Pi0Config.tactile_prefix_diff_from_reference is None. "
                        "Please set tactile_prefix_diff_from_reference explicitly (True/False)."
                    )
                self.tactile_prefix_encoder = _tactile_encoder.create_tactile_encoder(
                    encoder_type=config.tactile_prefix_encoder_type,
                    tactile_dim_in=config.tactile_prefix_dim_in,
                    tactile_history=config.tactile_prefix_history,
                    has_reference_frame=config.tactile_prefix_use_reference_frame,
                    diff_from_reference=config.tactile_prefix_diff_from_reference,
                    expert_width=prefix_width,
                    rngs=rngs,
                )
            # Decoder-suffix 触觉（例如 8×6 指力，经 MLP 编码）。
            if use_suffix and config.tactile_dim_in is not None and config.tactile_dim_in > 0:
                suffix_width = action_expert_config.width
                self.tactile_suffix_encoder = _tactile_encoder.create_tactile_encoder(
                    encoder_type=self.tactile_encoder_type,
                    tactile_dim_in=config.tactile_dim_in,
                    tactile_history=self.tactile_history,
                    has_reference_frame=config.tactile_use_reference_frame,
                    diff_from_reference=config.tactile_diff_from_reference,
                    expert_width=suffix_width,
                    rngs=rngs,
                )
                # 若选择把 suffix token 放到 prefix 序列，则需把 embedding 宽度对齐到 paligemma 的 width。
                if self.tactile_suffix_placement == "prefix":
                    prefix_width = paligemma_config.width
                    if suffix_width != prefix_width:
                        self.tactile_suffix_to_prefix_proj = nnx.Linear(
                            suffix_width, prefix_width, rngs=rngs
                        )

        # Action / time path.
        if config.pi05:
            # Pi05: same action dim as config.action_dim, no explicit state token.
            self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        else:
            # Pi0: 对 EXPERT_HIS_C_FUT，action_dim 保持不变（例如 7 动作 + 6 力，共 13 维），
            # 在 loss 内部按 [动作vs力] 维度切分加权。
            self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _encode_tactile_suffix(self, tactile: jax.Array) -> at.Float[at.Array, "b emb"]:
        """将 Observation.tactile_suffix 编码为单个 decoder-suffix embedding token。"""
        if tactile is None:
            raise ValueError("Suffix tactile encoder was called but observation.tactile_suffix is None.")
        encoder = self.tactile_suffix_encoder
        if encoder is None:
            raise ValueError("Suffix tactile encoder is not initialized but was called.")
        return encoder(tactile)

    def _encode_tactile_prefix(self, tactile: jax.Array) -> at.Float[at.Array, "b emb"]:
        """将 Observation.tactile_prefix 编码为单个 encoder-prefix embedding token。"""
        if tactile is None:
            raise ValueError("Prefix tactile encoder was called but observation.tactile_prefix is None.")
        encoder = self.tactile_prefix_encoder
        if encoder is None:
            raise ValueError("Prefix tactile encoder is not initialized but was called.")
        return encoder(tactile)

    def _process_tactile_tokens(
        self, obs: _model.Observation, mode: str
    ) -> tuple[list[jax.Array], list[jax.Array], list[bool]]:
        """Build tactile tokens for either the LLM prefix or expert suffix."""
        tokens_list: list[jax.Array] = []
        input_mask_list: list[jax.Array] = []
        ar_mask_list: list[bool] = []

        if self.tactile_type is TactileType.NO:
            return tokens_list, input_mask_list, ar_mask_list

        # suffix tokens will not be attended by postfix tokens
        ar_mask_value = mode == "suffix"

        if self.tactile_type is TactileType.EXPERT_HIS_C_FUT:
            # encoder-prefix 通道：从 Observation.tactile_prefix 读取，经 prefix encoder 编码。
            if mode == "prefix" and "tactile_prefix" in self.tactile_streams and obs.tactile_prefix is not None:
                tactile_token = self._encode_tactile_prefix(obs.tactile_prefix)[:, None, :]
                tokens_list.append(tactile_token)
                input_mask_list.append(jnp.ones(tactile_token.shape[:2], dtype=jnp.bool_))
                ar_mask_list.append(ar_mask_value)
            # tactile_suffix（原 tacforce 通道）：可选择放到 prefix 或 suffix。
            if "tactile_suffix" in self.tactile_streams and obs.tactile_suffix is not None:
                place_in_prefix = self.tactile_suffix_placement == "prefix"
                if (mode == "prefix" and place_in_prefix) or (mode == "suffix" and not place_in_prefix):
                    tactile_emb = self._encode_tactile_suffix(obs.tactile_suffix)  # [b, emb]
                    if mode == "prefix" and place_in_prefix and self.tactile_suffix_to_prefix_proj is not None:
                        tactile_emb = self.tactile_suffix_to_prefix_proj(tactile_emb)
                    tactile_token = tactile_emb[:, None, :]
                    tokens_list.append(tactile_token)
                    input_mask_list.append(jnp.ones(tactile_token.shape[:2], dtype=jnp.bool_))
                    ar_mask_list.append(ar_mask_value)

        return tokens_list, input_mask_list, ar_mask_list

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]

        # add tactile tokens to LLM prefix if requested
        tactile_tokens, tactile_input_mask, tactile_ar_mask = self._process_tactile_tokens(obs, mode="prefix")
        tokens.extend(tactile_tokens)
        input_mask.extend(tactile_input_mask)
        ar_mask.extend(tactile_ar_mask)

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []

        # 可选：在 expert suffix 前添加 tactile token（Pi0 与 Pi05 共享逻辑）。
        tactile_tokens, tactile_input_mask, tactile_ar_mask = self._process_tactile_tokens(obs, mode="suffix")
        tokens.extend(tactile_tokens)
        input_mask.extend(tactile_input_mask)
        ar_mask.extend(tactile_ar_mask)

        if not self.pi05:
            # 仅对标准 Pi0 添加显式 state token；Pi05 使用离散 state 编码，不再需要此 token。
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        return_components: bool = False,
    ) -> at.Float[at.Array, "*b ah"] | tuple[
        at.Float[at.Array, "*b ah"], dict[str, at.Float[at.Array, "*b ah"]]
    ]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            tactile_type=self.tactile_type,
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        if self.tactile_type is TactileType.EXPERT_HIS_C_FUT:
            # EXPERT_HIS_C_FUT（方案 B）：actions 本身已经包含 [动作, 触觉力]。
            # 约定：在“有效 action 维度” effective_action_dim 内，前 (effective_action_dim - tactile_dim) 维是动作，
            # 随后的 tactile_dim 维是触觉力，其后的维度（如果有）视为 padding，并且在 loss 中完全忽略。
            effective_ad = self.effective_action_dim
            ctrl_dim = effective_ad - self.tactile_dim
            if ctrl_dim <= 0:
                raise ValueError(
                    "EXPERT_HIS_C_FUT requires effective_action_dim "
                    f"({effective_ad}) > tactile_dim ({self.tactile_dim})."
                )
            # 组件分量（用于日志/对齐）：按维度段分别计算 MSE（段内取 mean）。
            # 动作损失：只在前 ctrl_dim 维计算（例如 7 个关节）。
            action_err2 = jnp.square(v_t[..., :ctrl_dim] - u_t[..., :ctrl_dim])
            action_loss = jnp.mean(action_err2, axis=-1)
            # 触觉力损失：紧接着的 tactile_dim 维（例如第 8–13 维）。
            tactile_slice = slice(ctrl_dim, ctrl_dim + self.tactile_dim)
            tactile_err2 = jnp.square(v_t[..., tactile_slice] - u_t[..., tactile_slice])
            tactile_loss = jnp.mean(tactile_err2, axis=-1)

            pad_loss = None
            if effective_ad < self.action_dim:
                pad_slice = slice(effective_ad, self.action_dim)
                pad_err2 = jnp.square(v_t[..., pad_slice] - u_t[..., pad_slice])
                pad_loss = jnp.mean(pad_err2, axis=-1)

            # 两种 loss 模式：
            # - split：与旧实现一致，按段内 MSE 相加（每段各自按段内维度 mean）。
            # - weighted_full：一次性整向量加权 MSE（按 action_dim mean，更贴近“除以 32”的直觉）。
            if self.expert_his_c_fut_loss_mode == "weighted_full":
                err2 = jnp.square(v_t - u_t)  # [..., ah, action_dim]
                w = jnp.ones((self.action_dim,), dtype=err2.dtype)
                # 力段权重（其余有效动作段权重为 1）
                w = w.at[tactile_slice].set(self.tactile_loss_weight)
                # padding 段权重（0 表示忽略；1 表示参与整体 MSE；也可设为其他值）
                if effective_ad < self.action_dim:
                    pad_w = self.padding_loss_weight if self.padding_loss_weight else 0.0
                    w = w.at[pad_slice].set(pad_w)
                total_loss = jnp.mean(err2 * w, axis=-1)
            else:
                # 默认 split：与旧实现一致
                total_loss = action_loss + self.tactile_loss_weight * tactile_loss
                if self.padding_loss_weight and pad_loss is not None:
                    total_loss = total_loss + self.padding_loss_weight * pad_loss
            if return_components:
                # 为了减小 aux 体积，仅返回 batch+time 维度上聚合后的 scalar 分量，
                # 避免在训练循环中携带完整 [*b, ah] 张量，减少内存与通信开销。
                action_loss_mean = jnp.mean(action_loss)
                tactile_loss_mean = jnp.mean(tactile_loss)
                aux = {
                    "action_loss": action_loss_mean,
                    "tactile_loss": tactile_loss_mean,
                }
                if pad_loss is not None:
                    aux["padding_loss"] = jnp.mean(pad_loss)
                return total_loss, aux
            return total_loss

        total_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if return_components:
            # 非 EXPERT_HIS_C_FUT 情况：返回占位分量，方便上层统一处理。
            zero = jnp.zeros_like(total_loss)
            return total_loss, {"action_loss": zero, "tactile_loss": zero}
        return total_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(
            None,
            observation,
            train=False,
            tactile_type=self.tactile_type,
        )
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            # EXPERT_HIS_C_FUT 中，actions 维度保持为 self.action_dim（7 动作 + 6 力等）。
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
