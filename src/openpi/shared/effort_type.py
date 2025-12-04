from enum import Enum, auto


class EffortType(Enum):
    """
    Effort (joint torque) usage modes.

    当前实现的 token 排布（只保留 expert 历史 + 未来联合预测这一条路径）：

        |<------- prefix (LLM) ---->|<----------- suffix (expert) --------------------->|
        |<- images ->|<- language ->|<- effort(hist) ->|<- state ->|<- actions+effort ->|

    - prefix：只包含视觉 + 语言，不再有任何 effort 相关 token。
    - suffix：在 state 之前插入一个由多帧历史 effort concat+MLP 得到的 expert token，
      然后是 state token，最后是一整段 [action_dim] 维的动作序列（其中后 effort_dim 维是力矩）。
    """

    NO = auto()
    """不使用 effort，但可以在数据中保留它用于统计等；模型完全忽略 effort。"""

    EXPERT_HIS_C_FUT = auto()
    """当前唯一保留的“力矩模式”：

    - 输入端：将多帧历史 effort concat 成一条向量，经 MLP 投成一个 token，作为 expert 的条件（HIS_C）。
    - 输出端：decoder 在 action 通道上同时学习 [动作 + 力]，loss 内部分别对动作/力做加权监督（FUT）。
    """

