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
from openpi.shared import array_typing as at
from openpi.shared.effort_type import EffortType

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
        self.effort_type = config.effort_type
        self.effort_dim = config.effort_dim
        # 有效 action 维度：用于 loss 中的 [动作, 力矩] 切分。
        # 允许数据只在前 K 维有意义，其余为 padding。
        self.effective_action_dim = config.effective_action_dim

        if self.pi05 and self.effort_type is not EffortType.NO:
            raise ValueError("Effort inputs / predictions are currently only supported for standard Pi0 (pi05=False).")
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

        # Effort projection (history encoder) — 当前仅支持 EXPERT_HIS_C_FUT 一种模式。
        if self.effort_type is EffortType.EXPERT_HIS_C_FUT:
            self.effort_proj_in = nnx.Linear(config.effort_dim_in, 2 * action_expert_config.width, rngs=rngs)
            self.effort_proj_out = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.effort_proj_in = None
            self.effort_proj_out = None

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

    def _project_effort(self, effort: at.Float[at.Array, "b *d"]) -> at.Float[at.Array, "b emb"]:
        if self.effort_proj_in is None or self.effort_proj_out is None:
            raise ValueError("Effort projection is not initialized but was called.")
        effort_hidden = self.effort_proj_in(effort)
        effort_hidden = nnx.swish(effort_hidden)
        return self.effort_proj_out(effort_hidden)

    def _process_effort_tokens(
        self, obs: _model.Observation, mode: str
    ) -> tuple[list[jax.Array], list[jax.Array], list[bool]]:
        """Build effort tokens for either the LLM prefix or expert suffix."""
        tokens_list: list[jax.Array] = []
        input_mask_list: list[jax.Array] = []
        ar_mask_list: list[bool] = []

        if obs.effort is None or self.effort_type is EffortType.NO:
            return tokens_list, input_mask_list, ar_mask_list

        # suffix tokens will not be attended by postfix tokens
        ar_mask_value = mode == "suffix"

        if mode == "suffix" and self.effort_type is EffortType.EXPERT_HIS_C_FUT:
            # 将多帧历史 effort concat 成一个向量，作为单个 expert token。
            if obs.effort.ndim == obs.state.ndim + 1:
                batch_size, _, _ = obs.effort.shape
                effort_flat = obs.effort.reshape(batch_size, -1)
            else:
                effort_flat = obs.effort
            effort_token = self._project_effort(effort_flat)[:, None, :]
            tokens_list.append(effort_token)
            input_mask_list.append(jnp.ones(effort_token.shape[:2], dtype=jnp.bool_))
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

        # add effort tokens to LLM prefix if requested
        effort_tokens, effort_input_mask, effort_ar_mask = self._process_effort_tokens(obs, mode="prefix")
        tokens.extend(effort_tokens)
        input_mask.extend(effort_input_mask)
        ar_mask.extend(effort_ar_mask)

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

        if not self.pi05:
            # For Pi0, optionally add effort tokens to the expert suffix (EXPERT_HIS_C_FUT, etc.).
            effort_tokens, effort_input_mask, effort_ar_mask = self._process_effort_tokens(obs, mode="suffix")
            tokens.extend(effort_tokens)
            input_mask.extend(effort_input_mask)
            ar_mask.extend(effort_ar_mask)

            # add a single state token
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
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            effort_type=self.effort_type,
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

        if self.effort_type is EffortType.EXPERT_HIS_C_FUT:
            # EXPERT_HIS_C_FUT（方案 B）：actions 本身已经包含 [动作, 力]。
            # 约定：在“有效 action 维度” effective_action_dim 内，前 (effective_action_dim - effort_dim) 维是动作，
            # 随后的 effort_dim 维是力矩，其后的维度（如果有）视为 padding，并且在 loss 中完全忽略。
            effective_ad = self.effective_action_dim
            ctrl_dim = effective_ad - self.effort_dim
            if ctrl_dim <= 0:
                raise ValueError(
                    "EXPERT_HIS_C_FUT requires effective_action_dim "
                    f"({effective_ad}) > effort_dim ({self.effort_dim})."
                )
            # 动作损失：只在前 ctrl_dim 维计算（例如 7 个关节）。
            action_loss = jnp.mean(jnp.square(v_t[..., :ctrl_dim] - u_t[..., :ctrl_dim]), axis=-1)
            # 力矩损失：紧接着的 effort_dim 维（例如第 8–13 维），其余 padding 维度忽略。
            effort_slice = slice(ctrl_dim, ctrl_dim + self.effort_dim)
            effort_loss = jnp.mean(
                jnp.square(v_t[..., effort_slice] - u_t[..., effort_slice]),
                axis=-1,
            )
            return action_loss + 0.1 * effort_loss

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

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
            effort_type=self.effort_type,
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
