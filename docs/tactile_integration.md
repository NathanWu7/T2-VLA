# 触觉/力觉在本工程中的引入方式（Tabero / PI0 / PI05）

本仓库的“触觉/力觉”并不是只做一个额外的输入张量，而是同时支持两类路径：

- **作为额外 token 融入 Transformer**：支持 **encoder-prefix**（进 LLM 前缀）与 **decoder-suffix**（进 action expert 后缀）两条触觉流。
- **作为动作向量的一部分参与监督**：将动作向量的后若干维视为“力/触觉力矩槽位”，在 loss 中与关节动作拆分并加权。

下文以 Tabero 数据为例说明，并在后文给出与 **OpenPI 官方 PI0（pi0_base）** 的逐项对比（可以直接当作“本工程相对官方 PI0 的改动说明”）。

---

## 数据侧：Tabero 的三种触觉形态

Tabero v2.1（本仓库约定）可能包含：

- **`tactile_image`**：第三路图像（在策略输入中映射到 `right_wrist_0_rgb`），本质是视觉模态。
- **`tactile_marker_motion`**：触觉力场/marker motion，典型形状 **`[9, 198, 2]`**。
- **`tactile_gripper_force`**：触觉指力历史，典型形状 **`[8, 6]`**。

在 `src/openpi/policies/libero_policy.py` 中：

- `TaberoTacFieldInputs`：读取 `tactile_marker_motion`，reshape 成 **`[9, 198*2]`**，写入 `inputs["tactile_prefix"]`（prefix 通道）。
- `TaberoTacForceInputs`：读取 `tactile_gripper_force`，写入 `inputs["tactile_suffix"]`（suffix 通道）。
- `TaberoTacAllInputs`：三路图像 + 同时写入 `tactile_prefix` 与 `tactile_suffix`（双通道）。

> 注意（归一化）：训练管线里会执行 `transforms.Normalize(norm_stats, ...)`。如果你用更新后的 `scripts/compute_norm_stats.py` 计算统计量，
> 则会**同时**对 `state/actions/tactile_prefix/tactile_suffix` 做归一化；如果你的 norm_stats 里没有 tactile 的统计量（例如旧版本 stats），
> 则 tactile 会保持原尺度（除非你在数据转换阶段自行标准化/裁剪/量纲变换）。

---

## 模型侧：两条触觉 token 流（prefix / suffix）

核心模型在 `src/openpi/models/pi0.py`，触觉编码器在 `src/openpi/models/tactile_encoder.py`。

### 0) 先明确“触觉/力觉”在本工程里的两种角色

- **条件信息（conditioning）**：通过 `Observation.tactile_prefix` / `Observation.tactile_suffix` 编成 **单个 token**，参与 self-attention，从而影响动作生成。
- **监督目标（supervision target）**：通过 `actions` 的后若干维（典型 6 维）作为“力/触觉力矩槽位”，在 loss 中单独计算并加权。

这两条路径彼此独立：你可以只用其中一条（例如只做力槽位监督但不注入 tactile token；或注入 token 但不监督力）。

### 1) prefix（encoder 前缀触觉 token）

典型对应 **tacfield / marker motion**：

- 数据进入 `Observation.tactile_prefix`（形状近似 `[B, 9, 396]`）。
- 配置里设置 `tactile_prefix_encoder_type="tcn"`（见 `src/openpi/training/config.py` 的 tabero 系列配置）。
- `TactileTCNEncoder` 会将多帧序列编码为 **单个 embedding token**（取最后一帧 hidden，再 `Linear` 到 embedding 维）。

### 2) suffix（decoder 后缀触觉 token）

典型对应 **tacforce / gripper force 历史**：

- 数据进入 `Observation.tactile_suffix`（形状近似 `[B, 8, 6]`）。
- `MLPTactileEncoder` 会将输入在 batch 后整体 flatten 成长度 `in_dim`（例如 `8*6=48`），两层 MLP 得到 **单个 embedding token**。
- 该 token 会插入到 suffix（action expert 输入序列）里，位置在 state/action tokens 之前（见 `Pi0.embed_suffix`）。

---

## “计算方式”：触觉编码 + loss 拆分

### 1) 触觉编码（tokenizer，张量/算子级描述）

#### 1.1 MLP tactile encoder（suffix 通道，tacforce）

输入 `tactile_suffix` 形状一般为 `[B, 8, 6]`，实现上会对 batch 后维度整体 flatten 得到长度 `in_dim=48` 的向量，再经过两层线性层（中间 swish）得到单个 token：

- \(x = \mathrm{flatten}(\mathrm{tactile\_suffix}) \in \mathbb{R}^{B\times 48}\)
- \(h = \mathrm{swish}(W_1 x + b_1)\)
- \(z = W_2 h + b_2\)，输出 \(z \in \mathbb{R}^{B\times d_{\text{emb}}}\)

工程意义：极轻量的历史汇聚器，把短窗触觉历史压成 **1 个条件 token**，放在 action expert 的 suffix 序列前部。

#### 1.2 TCN tactile encoder（prefix 通道，tacfield）

输入 `tactile_prefix` 形状一般为 `[B, 9, 396]`（Tabero marker motion 由 `[9,198,2]` reshape 得到）。TCN block 是“因果卷积”的显式实现：对每个时间步 \(t\) 聚合过去 \(k\) 个偏移的线性核，并加入残差：

- 令 \(\tilde{x}\) 为因果 padding 后序列
- \(y_t=\sum_{i=0}^{k-1} W_i \tilde{x}_{t-i}\)
- \(r_t=x_t\) 或 \(r_t=W_r x_t\)（维度不一致时）
- 输出：\(\mathrm{swish}(y_t+r_t)\)

堆叠若干层后，TCN encoder 取最后一个时间步 hidden \(h_T\)，再投影为单个 token：

- \(z=W_o h_T+b_o \in \mathbb{R}^{B\times d_{\text{emb}}}\)

工程意义：强调时序动态（marker motion 演化），更适合放入 prefix，与图像/语言同序列对齐。

### 2) 动作/力矩联合预测与 loss

当 `Pi0Config.tactile_type = EXPERT_HIS_C_FUT` 时（见 `src/openpi/models/pi0.py` 的 `compute_loss`）：

- 约定 **actions 本身已经包含 `[关节动作, 力/触觉力矩]`**（例如 13 维：7 关节 + 6 力）。
- 通过 `effective_action_dim` 告诉模型“在 padding 到更大 action_dim 前，真实有效维度是多少”（例如 `effective_action_dim=13`）。
- 用 `tactile_dim` 指定“力/触觉槽位的维度”（常见为 6）。
- loss 拆分：
  - **action_loss**：只对前 `ctrl_dim = effective_action_dim - tactile_dim` 维（例如 7）做 MSE。
  - **tactile_loss**：对接下来的 `tactile_dim` 维（例如 6）做 MSE。
  - **total_loss**：`action_loss + tactile_loss_weight * tactile_loss`。

这也解释了配置中常见两种基线：

- **no-force（不监督力）**：`tactile_loss_weight=0.0`（仍可保持 13D 输出，但力维度不计入 loss）。
- **only-joint（只训练 7 维）**：在数据侧 `SliceActions(7)`，彻底丢弃力维度。

> 更专业一点的表述：PI0/PI05 的训练目标属于 flow matching/扩散式回归（实现里构造 \(x_t\) 与 \(u_t\)），模型预测 \(v_t\)，损失是在 \(v_t\) 与 \(u_t\) 上的均方误差。
> 本工程的“动作/力拆分”实质是在 **同一个预测向量的不同维度子空间上赋予不同权重**（关节维度权重 1，力槽位权重 `tactile_loss_weight`，padding 维度权重 0）。

---

## 与 OpenPI 官方 PI0（pi0_base）对比：本工程到底改了什么？

这里的“OpenPI 官方 PI0”指 **官方发布的 pi0_base 系列训练/推理语义**：以图像+语言+低维 state 为条件输出未来动作序列；不引入触觉 token，也不把力/触觉槽位纳入输出与监督（推理侧常只回传前 7 维关节动作）。

> 说明：你当前仓库本身是 openpi 的派生/集成实现。下面对比使用“官方语义”描述，并用本仓库代码位置来锚定差异点（例如 `LiberoOutputs` 只回传 7 维，而 `LiberoForceOutputs` 回传 13 维）。

### 1) 输入/输出接口对比（数据字典 → Observation/Actions）

| 维度 | OpenPI 官方 PI0（pi0_base 语义） | 本工程（T2-VLA） |
|---|---|---|
| 图像输入 | 1～2 路为主（缺失视角用 padding + mask） | 明确支持 3 路：`image/wrist_image/tactile_image` → `base/left/right` |
| 触觉输入 | 无触觉字段 | 新增 `Observation.tactile_prefix` / `Observation.tactile_suffix`（可选） |
| state 输入 | PI0：连续 state，经 `state_proj` 形成显式 state token | PI0：一致；PI05：不再加显式 state token（state 更偏离散/语言侧配置） |
| actions 输出语义 | 通常只关心/回传前 7 维关节动作 | 支持 **13 维（7+6）**：后 6 维作为“力/触觉槽位”可监督也可回传 |

本仓库推理输出裁剪在 `src/openpi/policies/libero_policy.py`：

- 只回传关节 7D（对齐官方常用语义）：

```369:374:src/openpi/policies/libero_policy.py
return {"actions": np.asarray(data["actions"][:, :7])}
```

- 回传 13D（7+6 force，力矩方案）：

```377:385:src/openpi/policies/libero_policy.py
return {"actions": np.asarray(data["actions"][:, :13])}
```

### 2) token 序列结构对比（融合方式核心）

| 维度 | OpenPI 官方 PI0（pi0_base 语义） | 本工程（T2-VLA） |
|---|---|---|
| Prefix 序列 | `ImgTokens + TextTokens` | `ImgTokens + TextTokens + (tactile_prefix_token 可选)` |
| Suffix 序列（PI0） | `state_token + ActionTokens` | `(tactile_suffix_token 可选) + state_token + ActionTokens` |
| Suffix 序列（PI05） | `ActionTokens`（无显式 state token） | `(tactile_suffix_token 可选) + ActionTokens` |
| 触觉影响范围 | 无 | 触觉 token 通过 self-attention 影响后续动作生成（prefix/suffix 两条路径） |

本仓库插入点在 `src/openpi/models/pi0.py`：

- prefix 侧插入（图像/语言之后）：

```231:270:src/openpi/models/pi0.py
tactile_tokens, tactile_input_mask, tactile_ar_mask = self._process_tactile_tokens(obs, mode="prefix")
tokens.extend(tactile_tokens)
```

- suffix 侧插入（位于 state/action 之前）：

```285:297:src/openpi/models/pi0.py
tactile_tokens, tactile_input_mask, tactile_ar_mask = self._process_tactile_tokens(obs, mode="suffix")
tokens.extend(tactile_tokens)
...
state_token = self.state_proj(obs.state)[:, None, :]
```

### 3) 训练目标/损失函数对比（为什么“力槽位”会改变训练）

两者都属于“在扩散/flow matching 框架下回归 \(v_t\)”（本仓库实现见 `Pi0.compute_loss`）。差异在于：

- OpenPI 官方 PI0（pi0_base 语义）：动作向量主要承载关节动作语义（推理侧常只用/回传 7 维），loss 通常只对关节维度或对动作空间统一计算。
- 本工程：将动作空间拆成 \([a, f]\) 两个子空间（关节 vs 力槽位），并赋予不同权重；同时通过 `effective_action_dim` 把 padding 维度从 loss 中完全剔除。

对应实现：

```368:388:src/openpi/models/pi0.py
action_loss = jnp.mean(jnp.square(v_t[..., :ctrl_dim] - u_t[..., :ctrl_dim]), axis=-1)
tactile_loss = jnp.mean(jnp.square(v_t[..., ctrl_dim:ctrl_dim + self.tactile_dim] - u_t[..., ctrl_dim:ctrl_dim + self.tactile_dim]), axis=-1)
total_loss = action_loss + self.tactile_loss_weight * tactile_loss
```

> 工程/优化视角：这等价于在动作空间上施加一个“分块加权度量”，会改变梯度规模与多任务权衡；当 `tactile_loss_weight>0` 时，模型会被显式驱动去预测力槽位，因此推理时也可以输出力的未来轨迹。

### 3.1 padding 维度：忽略 vs “强制回归到 0”（本工程新增可选项）

在固定 `action_dim`（例如 32）但数据真实动作维度更小（例如 7/13/14）的场景里，`PadStatesAndActions` 会把多出来的维度用 0 补齐。
此时 padding 维度在 loss 里通常有两种策略：

- **忽略 padding（masked loss）**：只对有效维度计算 loss，padding 维完全不参与优化；
- **强制回归到 0（unmasked padding loss）**：padding 维也参与 loss，由于 target 是 padding 后的 0，等价于把这些维度压到“静默”状态。

本工程在 `TactileType.EXPERT_HIS_C_FUT` 分支里默认是“忽略 padding”，但你如果在**官方 checkpoint（pi0_base/pi05_base）上微调**希望更贴近官方训练范式，
可以把 `Pi0Config.padding_loss_weight` 设置为非 0（推荐 1.0 或更小），从而对 `effective_action_dim` 之后的维度也计算一个 `padding_loss`：

- `padding_loss_weight=0.0`：忽略 padding（默认，保持原行为）
- `padding_loss_weight>0.0`：加入 `padding_loss_weight * padding_loss`（相当于“强制 padding 回归到 0”）

### 4) 本工程特有：只做“力槽位监督拆分”，不引入触觉 token 参数

本工程允许只启用拆分监督而不创建触觉 token 编码器权重：

- `tactile_type=EXPERT_HIS_C_FUT`：启用动作/力拆分 loss
- `tactile_dim_in=0`：不创建 suffix tactile encoder（避免引入新权重，方便兼容官方 checkpoint）
- `tactile_streams=()`：不注入任何 tactile token（prefix/suffix 都关）

这种配置在 `src/openpi/training/config.py` 的 Tabero 系列里大量出现（例如 `pi0_lora_tacimg_tabero` / `pi0_lora_notac_tabero` / `pi05_notac_tabero`）。

---

## 与“传统 PI0”相比有什么不同？（面向快速阅读的总结）

### 传统 PI0（本仓库语境：不启用触觉）

- 输入：图像 +（连续）state token + language prompt。
- 输出：动作向量（常被 padding 到固定维度），loss 通常对所有维度统一计算。
- 不存在 `tactile_prefix/suffix` token，也没有“动作 vs 力”拆分加权。

### 本工程的“触觉 PI0 / PI05”

- **新增两条可选触觉 token 流**：
  - prefix：tacfield（marker motion → TCN → 1 token）；
  - suffix：tacforce（8×6 指力 → MLP → 1 token）。
- **动作向量内显式包含力/触觉槽位**（例如 13 维），并通过 `effective_action_dim / tactile_dim / tactile_loss_weight` 在 loss 中拆分加权。
- **归一化统计差异**：默认只统计/归一化 `state/actions`，触觉张量通常保持原尺度（除非你在数据转换阶段自行标准化）。

### PI05 相对 PI0 的额外差异（与触觉无关但常一起出现）

- **state 输入方式**：PI05 走“离散 state（拼进语言 token）”，而 PI0 有显式连续 `state token`（见 `Pi0.embed_suffix`）。
- **时间步注入**：PI05 的 action expert 使用 adaRMS 注入时间条件；PI0 用 MLP 混合 time+action（见 `Pi0.embed_suffix`）。
- 本仓库对 PI05 的触觉类型限制更严格（当前仅允许 `NO` 或 `EXPERT_HIS_C_FUT`）。

---

## Mermaid 图：一图看懂触觉如何进入模型

```mermaid
flowchart LR
  subgraph Dataset[Tabero sample（扁平字段）]
    IMG1[image]
    IMG2[wrist_image]
    IMG3[tactile_image]
    MM[tactile_marker_motion<br/>[9,198,2]]
    GF[tactile_gripper_force<br/>[8,6]]
    ST[state]
    AC[actions<br/>[13]=7 joint + 6 force]
  end

  subgraph Inputs[policy 输入映射（libero_policy）]
    IIMG[images: base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb]
    ITAC_P[tactile_prefix<br/>reshape: [9,198*2]=[9,396]]
    ITAC_S[tactile_suffix<br/>[8,6]]
    IST[state]
    IAC[actions]
  end

  subgraph Enc[触觉编码器（tactile_encoder）]
    TCN[TCN encoder<br/>[B,9,396] → 1 token]
    MLP[MLP encoder<br/>flatten 48 → 1 token]
  end

  subgraph Model[Pi0 / Pi05（pi0.py）]
    PFX[Prefix tokens<br/>image + text (+ tactile_prefix token)]
    SFX[Suffix tokens<br/>(+ tactile_suffix token) (+ state token for Pi0) + action tokens]
    LLM[Gemma Transformer]
    OUT[action_out_proj → v_t]
    LOSS[Loss split (EXPERT_HIS_C_FUT)<br/>action_loss + w * tactile_loss]
  end

  IMG1 --> IIMG
  IMG2 --> IIMG
  IMG3 --> IIMG
  MM --> ITAC_P --> TCN --> PFX
  GF --> ITAC_S --> MLP --> SFX
  ST --> IST --> SFX
  AC --> IAC --> LOSS

  IIMG --> PFX --> LLM --> OUT --> LOSS
  SFX --> LLM
```


