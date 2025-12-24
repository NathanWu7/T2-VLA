import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        #
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Optional: third camera (e.g., tactile image) mapped to the "right wrist" slot.
        # If not provided, we fall back to zeros like the original LiberoInputs.
        if "observation/tactile_image" in data:
            right_image = _parse_image(data["observation/tactile_image"])
            right_mask = np.True_
        else:
            right_image = np.zeros_like(base_image)
            # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
            right_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Either real third view (e.g. tactile image) or zero padding.
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_mask,
            },
        }

        # Optional: multi-frame gripper / tactile force as tactile / torque input.
        # For your own dataset, if you store a window of gripper forces under a different key,
        # map it to one of the keys below in the repack transform, and it will appear here.
        if "observation/gripper_force" in data:
            inputs["tactile_suffix"] = data["observation/gripper_force"]
        elif "observation/tactile_gripper_force" in data:
            inputs["tactile_suffix"] = data["observation/tactile_gripper_force"]

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TaberoTacImgInputs(transforms.DataTransformFn):
    """
    Tabero 输入（3 路图像 + 13 维动作，不使用触觉力场 / tactile）。

    - 图像：
        - observation/image          -> base_0_rgb
        - observation/wrist_image    -> left_wrist_0_rgb
        - observation/tactile_image  -> right_wrist_0_rgb
    - 动作：
        - actions                    -> 13 维（7 关节 + 6 力），在后续 PadStatesAndActions 中 padding 到 32 维。
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Tabero v2.1 数据集的单条样本是一个“扁平”的 dict，keys 直接是
        #   "image", "wrist_image", "tactile_image", "state", "actions", ...
        # 而不是嵌套在 "observation/..." 之下。
        base_image = _parse_image(data["image"])
        wrist_image = _parse_image(data["wrist_image"])
        tactile_image = _parse_image(data["tactile_image"])

        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": tactile_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # 明确：不读取任何 gripper_force / tactile_gripper_force，tactile 为空。
        return inputs


@dataclasses.dataclass(frozen=True)
class TaberoTacFieldInputs(transforms.DataTransformFn):
    """
    Tabero 输入（2 路图像 + 触觉力场 tactile + 13 维动作）。

    - 图像：
        - Tabero v2.1 扁平格式：
            - image                  -> base_0_rgb
            - wrist_image            -> left_wrist_0_rgb
          第三路视觉留空，用零图 + mask=False（与原始 LiberoInputs 一致）
        - 兼容旧格式：observation/image, observation/wrist_image
    - 触觉力场：
        - 推荐：tactile_marker_motion，形状约为 [9, 198, 2]
          在这里先 reshape 成 [9, 198*2]，再作为 Observation.tactile（[*b, n, e]）进入模型，
          模型内部再整体 flatten 成一个 tactile token。
        - 兼容：tactile_gripper_force / observation/tactile_gripper_force（例如 [8, 6]），
          直接作为 Observation.tactile 使用。
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 支持 Tabero v2.1 扁平格式和旧的 observation/... 格式。
        if "image" in data:
            base_image = _parse_image(data["image"])
            wrist_image = _parse_image(data["wrist_image"])
            state = data["state"]
        else:
            base_image = _parse_image(data["observation/image"])
            wrist_image = _parse_image(data["observation/wrist_image"])
            state = data["observation/state"]

        right_image = np.zeros_like(base_image)
        right_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_mask,
            },
        }

        # 触觉力场作为 decoder-suffix tactile（兼容旧 tacfield 单通道配置）。
        # 优先使用 tactile_marker_motion（例如 [9, 198, 2]），reshape 成 [9, 198*2] 后作为 [n, e]。
        if "tactile_marker_motion" in data:
            motion = np.asarray(data["tactile_marker_motion"])
            if motion.ndim != 3:
                raise ValueError(f"tactile_marker_motion must be 3D, got shape {motion.shape}")
            n, m, d = motion.shape
            tactile = motion.reshape(n, m * d)
            inputs["tactile_suffix"] = tactile
        elif "tactile_gripper_force" in data:
            inputs["tactile_suffix"] = data["tactile_gripper_force"]
        elif "observation/tactile_gripper_force" in data:
            inputs["tactile_suffix"] = data["observation/tactile_gripper_force"]
        elif "observation/gripper_force" in data:
            inputs["tactile_suffix"] = data["observation/gripper_force"]

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TaberoTacForceInputs(transforms.DataTransformFn):
    """
    Tabero 输入（2 路图像 + 8×6 触觉指力历史 + 13 维动作）。

    - 图像：
        - Tabero v2.1 扁平格式：
            - image                  -> base_0_rgb
            - wrist_image            -> left_wrist_0_rgb
          第三路视觉留空，用零图 + mask=False（与原始 LiberoInputs 一致）
        - 兼容旧格式：observation/image, observation/wrist_image
    - 指力历史：
        - tactile_gripper_force（优先），形状约为 [8, 6]
        - 兼容：observation/tactile_gripper_force / observation/gripper_force
      直接作为 Observation.tactile（[*b, n, e]）喂给模型，用于 EXPERT_HIS_C_FUT 的“历史指力 token”。
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 支持 Tabero v2.1 扁平格式和旧的 observation/... 格式。
        if "image" in data:
            base_image = _parse_image(data["image"])
            wrist_image = _parse_image(data["wrist_image"])
            state = data["state"]
        else:
            base_image = _parse_image(data["observation/image"])
            wrist_image = _parse_image(data["observation/wrist_image"])
            state = data["observation/state"]

        right_image = np.zeros_like(base_image)
        right_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_mask,
            },
        }

        # 8×6 左右指力历史作为 decoder-suffix tactile。
        if "tactile_gripper_force" in data:
            inputs["tactile_suffix"] = data["tactile_gripper_force"]
        elif "observation/tactile_gripper_force" in data:
            inputs["tactile_suffix"] = data["observation/tactile_gripper_force"]
        elif "observation/gripper_force" in data:
            inputs["tactile_suffix"] = data["observation/gripper_force"]

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TaberoTacAllInputs(transforms.DataTransformFn):
    """
    Tabero 输入（3 路图像 + tacfield+tacforce 双触觉 + 13 维动作）。

    - tacimg：第三路 `tactile_image` 作为额外摄像头像素输入；
    - tacfield（encoder 前缀通道）：
        - `tactile_marker_motion` 形状 [9, 198, 2]，reshape 成 [9, 198*2] → `tactile_prefix`；
    - tacforce（decoder 后缀通道）：
        - `tactile_gripper_force` 形状 [8, 6]，直接 → `tactile`；
    - 动作：13 维（7 关节 + 6 力），在 loss 中仍按 [动作, 力] 拆分并对力做加权监督。
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 假设使用 Tabero v2.1 扁平格式，不再做额外兼容分支：
        #   image / wrist_image / tactile_image / state / tactile_marker_motion / tactile_gripper_force / actions / prompt
        base_image = _parse_image(data["image"])
        wrist_image = _parse_image(data["wrist_image"])
        tactile_image = _parse_image(data["tactile_image"])
        state = data["state"]

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": tactile_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # tacfield：marker motion → encoder 前缀触觉通道（tactile_prefix）。
        motion = np.asarray(data["tactile_marker_motion"])
        if motion.ndim != 3:
            raise ValueError(f"tactile_marker_motion must be 3D, got shape {motion.shape}")
        n, m, d = motion.shape
        tactile_prefix = motion.reshape(n, m * d)
        inputs["tactile_prefix"] = tactile_prefix

        # tacforce：8×6 指力历史 → decoder 后缀触觉通道（tactile_suffix）。
        inputs["tactile_suffix"] = data["tactile_gripper_force"]

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TaberoNoTactInputs(transforms.DataTransformFn):
    """
    Tabero 输入（只用 2 路图像 + state + 动作，不使用任何 tactile / 力）。

    - 图像：
        - Tabero v2.1 扁平格式：
            - image                  -> base_0_rgb
            - wrist_image            -> left_wrist_0_rgb
          第三路视觉留空，用零图 + mask=False（与原始 LiberoInputs 一致）
        - 兼容旧格式：observation/image, observation/wrist_image
    - 不读取任何 tactile_gripper_force / tactile_marker_motion / gripper_force。
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 支持 Tabero v2.1 扁平格式和旧的 observation/... 格式。
        if "image" in data:
            base_image = _parse_image(data["image"])
            wrist_image = _parse_image(data["wrist_image"])
            state = data["state"]
        else:
            base_image = _parse_image(data["observation/image"])
            wrist_image = _parse_image(data["observation/wrist_image"])
            state = data["observation/state"]

        right_image = np.zeros_like(base_image)
        right_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_mask,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # 对于官方 Libero，我们只返回前 7 维关节动作（其余为 padding）。
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class LiberoForceOutputs(transforms.DataTransformFn):
    """
    Libero 输出变换（带力矩）：返回前 13 维（7 动作 + 6 力），用于你当前的力矩方案。
    """

    def __call__(self, data: dict) -> dict:
        # data["actions"] 形状为 [B, H, D]，这里裁剪到前 13 维（7 动作 + 6 力）。
        return {"actions": np.asarray(data["actions"][:, :13])}
