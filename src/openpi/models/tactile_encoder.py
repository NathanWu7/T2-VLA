import logging
from typing import Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


logger = logging.getLogger("openpi")


class MLPTactileEncoder(nnx.Module):
    """简单的 flatten+MLP 触觉编码器（向后兼容原有实现）。

    输入约定：
        tactile: [b, *d]，在 batch 维之后的所有维度会整体 flatten 成一维，
        其总长度必须等于 `in_dim`。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        emb_dim: int,
        rngs: Optional[nnx.Rngs] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.proj_in = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.proj_out = nnx.Linear(hidden_dim, emb_dim, rngs=rngs)

    def __call__(self, tactile: jax.Array) -> at.Float[at.Array, "b emb"]:
        if tactile.ndim < 2:
            raise ValueError(f"MLPTactileEncoder expects input with rank >= 2, got shape {tactile.shape}.")
        batch_size = tactile.shape[0]
        tactile_flat = tactile.reshape(batch_size, -1)
        if tactile_flat.shape[-1] != self.in_dim:
            raise ValueError(
                f"MLPTactileEncoder: expected flattened dim={self.in_dim}, "
                f"got {tactile_flat.shape[-1]} (input shape={tactile.shape})."
            )
        hidden = self.proj_in(tactile_flat)
        hidden = nnx.swish(hidden)
        return self.proj_out(hidden)


class TactileTCNBlock(nnx.Module):
    """简单的 1D TCN block：显式实现因果卷积 + 残差（仅依赖 nnx.Linear）。

    输入形状：[b, n, in_dim]，输出形状：[b, n, out_dim]。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        rngs: Optional[nnx.Rngs] = None,
    ):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1 for TactileTCNBlock.")
        self.kernel_size = kernel_size
        kernels: dict[str, nnx.Linear] = {}
        for k in range(kernel_size):
            kernels[f"kernel_{k}"] = nnx.Linear(in_dim, out_dim, rngs=rngs)
        self.kernels = nnx.Dict(**kernels)
        self.residual_proj = None
        if in_dim != out_dim:
            self.residual_proj = nnx.Linear(in_dim, out_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        if x.ndim != 3:
            raise ValueError(f"TactileTCNBlock expects input of rank 3, got shape {x.shape}.")
        b, n, d = x.shape
        k = self.kernel_size

        # 因果 padding：在时间维前面补 (k-1) 个 0。
        pad = jnp.zeros((b, k - 1, d), dtype=x.dtype)
        x_pad = jnp.concatenate([pad, x], axis=1)  # [b, n + k - 1, d]

        # 聚合 K 个时间偏移的线性变换（完全向量化，便于 XLA 并行优化）。
        y = 0.0
        for idx, linear in self.kernels.items():
            # idx: "kernel_0" 表示当前时刻，"kernel_1" 表示前 1 步，依此类推。
            offset = int(idx.split("_")[1])
            start = (k - 1 - offset)
            end = start + n
            x_slice = x_pad[:, start:end, :]  # [b, n, d]
            y = y + linear(x_slice)

        residual = x if self.residual_proj is None else self.residual_proj(x)
        return nnx.swish(y + residual)


class TactileTCNEncoder(nnx.Module):
    """多层 TCN encoder，用于将多帧 tactile 序列编码为单个 token embedding。

    约定：
    - has_reference_frame=True：
        输入为 [B, 1 + H, E]（第 0 帧为基准，其余 H 帧为历史）：
        - 若 diff_from_reference=True：只对后 H 帧做 (frame - baseline)，长度为 H；
        - 若 diff_from_reference=False：直接使用全部 1+H 帧做 TCN（例如 Tabero 9 帧 marker）。
    - has_reference_frame=False：
        输入视为不含显式基准帧，约定为 [B, H, E] 或更多帧，多出来的帧只保留最近 H 帧。
    """

    def __init__(
        self,
        in_dim: int,
        history_len: int,
        hidden_dim: int,
        emb_dim: int,
        has_reference_frame: bool,
        diff_from_reference: bool,
        num_layers: int = 1,
        kernel_size: int = 3,
        rngs: Optional[nnx.Rngs] = None,
    ):
        super().__init__()
        if history_len <= 0:
            raise ValueError("TactileTCNEncoder.history_len must be > 0.")
        if num_layers < 1:
            raise ValueError("TactileTCNEncoder.num_layers must be >= 1.")
        self.history_len = history_len  # 这里的 history_len 统一指 config 里的 H（例如 8）。
        self.has_reference_frame = has_reference_frame
        self.diff_from_reference = diff_from_reference

        blocks: dict[str, TactileTCNBlock] = {}
        for i in range(num_layers):
            block_in = in_dim if i == 0 else hidden_dim
            blocks[f"block_{i}"] = TactileTCNBlock(
                in_dim=block_in,
                out_dim=hidden_dim,
                kernel_size=kernel_size,
                rngs=rngs,
            )
        self.blocks = nnx.Dict(**blocks)
        # 最后对时间维做「取最后一帧」池化，再线性映射到 embedding 维度。
        self.out_proj = nnx.Linear(hidden_dim, emb_dim, rngs=rngs)

    def __call__(self, tactile: jax.Array) -> at.Float[at.Array, "b emb"]:
        # 允许输入为 [b, n, in_dim] 或 [b, in_dim]（后者视作单步序列）。
        if tactile.ndim == 2:
            tactile_seq = tactile[:, None, :]
        elif tactile.ndim == 3:
            tactile_seq = tactile
        else:
            raise ValueError(
                f"TactileTCNEncoder expects input of rank 2 or 3, got shape {tactile.shape}."
            )

        b, n, d = tactile_seq.shape
        H = self.history_len

        if self.has_reference_frame:
            # 至少需要 2 帧（1 基准 + 至少 1 帧历史）。
            if n < 2:
                raise ValueError(
                    "TactileTCNEncoder with has_reference_frame=True expects at least 2 frames, "
                    f"got {n} (shape={tactile_seq.shape})."
                )
            if self.diff_from_reference:
                # 旧版差分逻辑：只用 H 帧历史，每帧减基准帧。
                max_hist = min(H, n - 1)
                baseline = tactile_seq[:, 0:1, :]           # [B, 1, d]
                history = tactile_seq[:, 1 : 1 + max_hist]  # [B, max_hist, d]
                history = history - baseline                 # [B, max_hist, d]
                if max_hist != H:
                    logger.warning(
                        "TactileTCNEncoder(diff): expected history_len=%d (excluding baseline), "
                        "but only %d history frames available; using %d frames.",
                        H,
                        n - 1,
                        max_hist,
                    )
                seq_for_tcn = history
            else:
                # 新版“9 帧直接 TCN”逻辑：直接把 [baseline + H 帧历史] 一起送入 TCN。
                max_steps = min(H + 1, n)
                seq_for_tcn = tactile_seq[:, :max_steps, :]  # [B, 1+H, d] 或更短
                if max_steps != H + 1:
                    logger.warning(
                        "TactileTCNEncoder(full-seq): expected 1+history_len=%d frames, "
                        "but only %d available; using %d frames.",
                        H + 1,
                        n,
                        max_steps,
                    )
        else:
            # 不使用基准帧：只保留最近 H 帧。
            if n >= H:
                seq_for_tcn = tactile_seq[:, -H:, :]
            else:
                seq_for_tcn = tactile_seq
                logger.warning(
                    "TactileTCNEncoder: expected history_len=%d, but only %d frames available; "
                    "using all frames.",
                    H,
                    n,
                )

        h = seq_for_tcn
        for block in self.blocks.values():
            h = block(h)
        # 使用最后一个时间步的 hidden 作为整体 tactile 序列的表示。
        h_last = h[:, -1, :]  # [b, hidden_dim]
        return self.out_proj(h_last)  # [b, emb_dim]


def create_tactile_encoder(
    *,
    encoder_type: str,
    tactile_dim_in: int,
    tactile_history: Optional[int],
    has_reference_frame: bool,
    diff_from_reference: bool,
    expert_width: int,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """根据配置创建 tactile 编码器（MLP 或 TCN）。

    - encoder_type == "mlp"：沿用 flatten+MLP 路径（与旧实现完全一致）。
    - encoder_type == "tcn"：使用 TCN，对 Tabero tacfield 提供基准帧差分 + 历史长度裁剪。
    """
    if tactile_dim_in <= 0:
        raise ValueError("create_tactile_encoder: tactile_dim_in must be > 0 when encoder is enabled.")

    if encoder_type == "tcn":
        if tactile_history is None:
            raise ValueError(
                "Pi0Config.tactile_history must be set when tactile_encoder_type='tcn'. "
                "例如：tacforce/tacfield 均可设为 8。"
            )
        # 根据是否存在基准帧来推断“用于维度计算的时间步数”：
        # - has_reference_frame=True（如 tacfield）：总步数 = 1(基准) + H(历史)
        # - has_reference_frame=False（如 tacforce）：总步数 = H
        steps_for_dim = tactile_history + 1 if has_reference_frame else tactile_history
        if steps_for_dim <= 0:
            raise ValueError("tactile_history must be > 0 for TCN encoder.")
        if tactile_dim_in % steps_for_dim != 0:
            raise ValueError(
                "Pi0Config.tactile_dim_in ({tactile_dim_in}) must be divisible by "
                f"effective_steps={steps_for_dim} when using TCN encoder; "
                f"got tactile_dim_in={tactile_dim_in}, history={tactile_history}, "
                f"has_reference_frame={has_reference_frame}."
            )
        per_step_dim = tactile_dim_in // steps_for_dim
        return TactileTCNEncoder(
            in_dim=per_step_dim,
            history_len=tactile_history,
            # 轻量版 TCN：hidden_dim 减半为 expert_width，并只保留 1 层。
            hidden_dim=expert_width,
            emb_dim=expert_width,
            has_reference_frame=has_reference_frame,
            diff_from_reference=diff_from_reference,
            rngs=rngs,
        )

    # 默认：MLP 编码器。
    return MLPTactileEncoder(
        in_dim=tactile_dim_in,
        hidden_dim=2 * expert_width,
        emb_dim=expert_width,
        rngs=rngs,
    )


