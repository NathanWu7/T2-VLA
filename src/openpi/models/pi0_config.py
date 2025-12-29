import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
from openpi.shared.tactile_type import TactileType
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Tactile / torque configuration (JAX-only; PyTorch side handles its own config).
    # By default, models ignore tactile.
    tactile_type: TactileType = TactileType.NO
    # Per-timestep tactile dimension (e.g., 14 for 7 joints * 2 arms), before history concatenation.
    tactile_dim: int = 14
    # Effective input dim for the tactile MLP projector. Usually tactile_dim * len(tactile_history).
    # When not set explicitly (e.g., from training config + data.tactile_history), we fall back to tactile_dim.
    tactile_dim_in: int | None = None
    # Tactile 历史长度（时间步数），主要用于 TCN 编码器：
    # - 对于 Tabero marker motion（9 帧：1 基准 + 8 接触），可以设为 8（只数历史帧）。
    # - 对于 gripper_force（8×6），可以设为 8。
    tactile_history: int | None = None
    # 有效的“语义 action 维度”。当数据中的 action 先 padding 到较大的 action_dim（例如 32），
    # 但真实只有前 K 维有意义时，可以把 effective_action_dim 设为 K，用于 loss 中的动作/力矩切分。
    # 默认为 action_dim，保持向后兼容。
    effective_action_dim: int | None = None
    # 触觉 / 力损失的权重：total_loss = action_loss + tactile_loss_weight * tactile_loss。
    # 仅在 tactile_type 为 EXPERT_HIS_C_FUT 时生效。
    tactile_loss_weight: float = 0.1
    # Padding 维度（effective_action_dim 之后）loss 权重：
    # - 0.0：忽略 padding 维度（当前 Tabero-force 方案的默认行为）
    # - 1.0：对 padding 维度也计算 loss（等价于“强制把 padding 维度回归到 0”，适合对齐官方 checkpoint 的默认训练范式）
    # 仅在 tactile_type 为 EXPERT_HIS_C_FUT 且 effective_action_dim < action_dim 时生效。
    padding_loss_weight: float = 1.0
    # EXPERT_HIS_C_FUT 的 loss 计算模式：
    # - "split"：当前默认实现。按 [动作段, 力段, padding 段] 分别计算 MSE（各自按段内维度取 mean）
    #            再按权重相加：action_loss + w*tactile_loss (+ pad_w*pad_loss)。
    # - "weighted_full"：一次性对整条 action 向量做加权 MSE：
    #            mean( w[d] * (v_t - u_t)^2 )，其中 w[d] 在不同维度段取不同常数。
    #            注意：该模式下“除数”固定为 action_dim（例如 32），更贴近“整体 MSE”的直觉。
    # 仅在 tactile_type 为 EXPERT_HIS_C_FUT 时生效。
    expert_his_c_fut_loss_mode: str = "weighted_full"
    # Tactile 编码器类型：
    # - "mlp"：使用旧版 flatten+MLP（向后兼容，默认）
    # - "tcn"：使用时序卷积网络（TCN）作为 tactile tokenizer（例如 Tabero tacfield marker motion）
    tactile_encoder_type: str = "mlp"
    # 是否存在“基准帧”（仅影响 TCN 编码器）：
    # - False：输入视为不含显式基准帧（如 8×6 gripper_force）；
    # - True：约定输入为 [1 + H, D]，第 0 帧为基准，其余 H 帧为历史（如 Tabero 9 帧 marker）。
    tactile_use_reference_frame: bool = False
    # 是否对历史帧做“减基准帧”差分（仅影响 TCN 编码器，且在 tactile_use_reference_frame=True 时生效）：
    # - True：只对后 H 帧做 (frame - baseline)，长度为 H；
    # - False：对 [baseline + H 帧] 全部做 TCN，例如直接用 9 帧 marker 序列。
    tactile_diff_from_reference: bool = True

    # Encoder-prefix 触觉通道配置（仅在显式启用 prefix 触觉流时生效）。
    tactile_prefix_dim_in: int | None = None
    tactile_prefix_history: int | None = None
    tactile_prefix_encoder_type: str | None = None
    tactile_prefix_use_reference_frame: bool | None = None
    tactile_prefix_diff_from_reference: bool | None = None

    # 触觉通道选择（新接口，用于替代 dual_tactile 开关）：
    # - () / []：不启用任何触觉 token，仅使用动作/力的 loss 拆分逻辑；
    # - ("tactile_suffix",)：只启用 decoder-suffix 通道（例如原版 tacforce 的 8×6 指力）；
    # - ("tactile_prefix",)：只启用 encoder-prefix 通道（例如 tacfield 的 marker motion）；
    # - ("tactile_suffix", "tactile_prefix")：同时启用前缀+后缀双通道。
    tactile_streams: tuple[str, ...] = ()

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.tactile_dim_in is None:
            object.__setattr__(self, "tactile_dim_in", self.tactile_dim)
        if self.effective_action_dim is None:
            object.__setattr__(self, "effective_action_dim", self.action_dim)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
