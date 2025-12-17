"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
from openpi.shared.tactile_type import TactileType
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType

# 全局触觉 / 力 loss 权重（用于 EXPRT_HIS_C_FUT：total_loss = action_loss + w * tactile_loss）
TACTILE_LOSS_WEIGHT: float = 0.1
# Tabero / 力矩相关实验使用的统一 tactile 历史长度（单位：帧数）。
TABERO_TACTILE_HISTORY: int = 8
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoTactileDataConfig(DataConfigFactory):
    """
    Libero 数据配置（带 gripper_force 作为 tactile）。

    在标准 `LeRobotLiberoDataConfig` 的基础上，将多帧 `observation/gripper_force`
    透传到后续 policy transform，并最终映射为 `Observation.tactile`。
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        # 额外：把多帧 gripper_force 转发出来
                        "observation/gripper_force": "gripper_force",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoForceOutputs()]
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TaberoTacImgDataConfig(DataConfigFactory):
    """
    Tabero（三路图像 + 13D 动作，无 tactile）数据配置。

    - 图像：observation/image, observation/wrist_image, observation/tactile_image
      通过 TaberoTacImgInputs 映射到 3 路视觉 token。
    - 动作：13 维（7 关节 + 6 力），经 PadStatesAndActions padding 到 32 维。
    """

    extra_delta_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 直接使用 Tabero 的 LeRobot 格式（observation/...），不做 repack。
        data_transforms = _transforms.Group(
            inputs=[libero_policy.TaberoTacImgInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoForceOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TaberoTacFieldDataConfig(DataConfigFactory):
    """
    Tabero（两路图像 + 触觉力场 + 13D 动作）数据配置。

    - 图像：observation/image, observation/wrist_image
    - 触觉力场：observation/tactile_gripper_force（优先）或 observation/gripper_force
      直接映射为 Observation.tactile，后续在模型内部 flatten+MLP 得到 tactile token。
    """

    extra_delta_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[libero_policy.TaberoTacFieldInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoForceOutputs()],
        )


@dataclasses.dataclass(frozen=True)
class TaberoTacForceDataConfig(DataConfigFactory):
    """
    Tabero（两路图像 + 8×6 指力历史 + 13D 动作）数据配置。

    - 图像：image, wrist_image
    - 指力历史：tactile_gripper_force（或 observation/tactile_gripper_force / observation/gripper_force）
      直接作为 Observation.tactile（[*b, n, e]）输入，用于 EXPERT_HIS_C_FUT 的“历史指力 token”。
    """

    extra_delta_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[libero_policy.TaberoTacForceInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoForceOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TaberoNoTactNoForceDataConfig(DataConfigFactory):
    """
    Tabero（多路图像 + 只用 7D 关节动作，不使用任何 tactile / 指力）的数据配置。

    - 图像：image, wrist_image, tactile_image（第三路当作普通视觉使用）
    - 动作：原始 13D（7 关节 + 6 力）在这里通过 SliceActions(7) 截断为 7D，只训练关节，
      后 6 维力完全从 loss 中“屏蔽掉”，行为与原始 pi0_libero_wo_force 类似。
    """

    extra_delta_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[
                # 只用两路图像（image / wrist_image）+ state + 13D actions，从 Tabero v2.1 扁平格式读入，
                # 不读取 tactile_image / tactile_force 等任何触觉模态。
                libero_policy.TaberoNoTactInputs(model_type=model_config.model_type),
                # 将动作截断为前 7 维（关节），彻底丢弃后 6 维指力。
                _transforms.SliceActions(7),
            ],
            outputs=[libero_policy.LiberoOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoNoTactileDataConfig(DataConfigFactory):
    """Libero 数据配置（不使用 gripper_force / tactile，只保留前 7 维动作）。"""

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 与 LeRobotLiberoDataConfig 相同的 repack，只是不再转发 gripper_force。
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # 数据流：LiberoInputs 负责 key 适配与 padding，SliceActions(7) 将动作截断到前 7 维。
        data_transforms = _transforms.Group(
            inputs=[
                libero_policy.LiberoInputs(model_type=model_config.model_type),
                _transforms.SliceActions(7),
            ],
            outputs=[libero_policy.LiberoOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_lora_tacimg_tabero",
        # 三路图像（image / wrist_image / tactile_image），13 维动作（7 关节 + 6 力），
        # 不使用历史 tactile token，但在 loss 里仍对 [关节 vs 力] 做加权（0.1 * tactile_loss）。
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # 数据中真实有效动作维度为 13，其余通过 PadStatesAndActions padding。
            effective_action_dim=13,
            # 启用 EXPERT_HIS_C_FUT 的 loss 拆分逻辑，但 Observation.tactile 为空，
            # 所以只做 [前 7 维动作 + 后 6 维力] 的加权监督，不注入 tactile token。
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            # tactile_dim_in=0 表示不需要 tactile token 的 Linear 投影，只启用 loss 拆分逻辑，
            # 既避免引入新的权重，又保持和原有 checkpoint 的结构兼容。
            tactile_dim_in=0,
            tactile_loss_weight=TACTILE_LOSS_WEIGHT,
        ),
        data=TaberoTacImgDataConfig(
            # 你的原始 Tabero 数据集（含 tactile_image / tactile_gripper_force 等字段）。
            repo_id="NathanWu7/tabero",
            base_config=DataConfig(
                # 如果在 LeRobot meta 里有 tasks 信息，可以启用从 task 里自动生成 prompt。
                prompt_from_task=True,
            ),
            # 与当前 pi0_libero_force 配置保持一致，额外做一次 delta transform。
            extra_delta_transform=True,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=1.25e-5,
            decay_lr=1.25e-6,
        ),
        # 使用官方 pi0 base checkpoint 初始化，再做 LoRA 微调。
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params",
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_lora_notac_tabero",
        # Tabero 基线：使用三路图像（image / wrist_image / tactile_image）作为纯视觉输入，
        # 只训练前 7 维关节动作，不使用任何 tactile token，也不对后 6 维指力做监督。
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # 使用与 force 版相同的 EXPERT_HIS_C_FUT loss 拆分逻辑，但关闭力的监督：
            # - effective_action_dim=13：前 7 维是真实关节动作，后 6 维作为“力槽位”；
            # - tactile_type=EXPERT_HIS_C_FUT：在 compute_loss 中按 [7 动作 + 6 力] 拆分；
            # - tactile_dim_in=0：不创建 tactile token 相关 Linear，只启用 loss 拆分逻辑；
            # - tactile_loss_weight=0.0：力的 loss 权重为 0，只剩 7 维动作 loss。
            effective_action_dim=13,
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            tactile_dim_in=0,
            tactile_loss_weight=0.0,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=1.25e-5,
            decay_lr=1.25e-6,
        ),
        data=TaberoNoTactNoForceDataConfig(
            repo_id="NathanWu7/tabero",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params",
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_lora_tacfield_tabero",
        # 两路图像（image / wrist_image）+ 触觉力场 tactile（8×6）+ 13 维未来动作/力联合预测。
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # 13 维 = 7 关节 + 6 力，loss 内部按 [动作, 力] 维度切分。
            effective_action_dim=13,
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            # Tabero 力场：tactile_marker_motion 形状为 [9, 198, 2]，
            # 在 TaberoTacFieldInputs 中先 reshape 成 [9, 198*2]，然后在模型内部使用 TCN：
            # - 第 1 帧作为“基准帧”（无接触）；
            # - 后 8 帧每一帧都减去基准帧，形成 8 帧历史；
            # - 因此 tactile_dim_in 仍然是 9 * 198 * 2（1+8 帧），但 history=TABERO_TACTILE_HISTORY。
            tactile_dim_in=9 * 198 * 2,
            # 历史长度 H：TABERO_TACTILE_HISTORY 帧接触历史（例如 8），
            # 但在 full-seq 模式下实际 TCN 序列长度为 1+H（基准 + 历史），仍使用 H 作为超参。
            tactile_history=TABERO_TACTILE_HISTORY,
            # 使用 TCN（时序卷积网络）作为 marker motion 的 tactile tokenizer，
            # 且直接使用 [基准 + 8 帧接触] 共 9 帧做 TCN，不做差分；
            # 通过 tactile_diff_from_reference 可切换回“差分版”。
            tactile_encoder_type="tcn",
            tactile_use_reference_frame=True,
            tactile_diff_from_reference=False,
            tactile_in_prefix_only=True,
            tactile_loss_weight=TACTILE_LOSS_WEIGHT,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=1.25e-5,
            decay_lr=1.25e-6,
        ),
        data=TaberoTacFieldDataConfig(
            repo_id="NathanWu7/tabero",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params",
            # 新增的 tactile_proj_* 参数在 base checkpoint 里不存在，允许缺失。
            missing_regex=".*",
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_lora_tacforce_tabero",
        # 两路图像（image / wrist_image）+ 8×6 指力历史 tactile + 13 维未来动作/力联合预测。
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # 13 维 = 7 关节 + 6 力，loss 内部按 [动作, 力] 维度切分并对力做 0.1 加权。
            effective_action_dim=13,
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            # 指力历史：tactile_gripper_force 形状约为 [8, 6]，在模型内部 flatten 成 8*6。
            tactile_dim_in=8 * 6,
            # 显式设定历史长度为 TABERO_TACTILE_HISTORY（无显式基准帧，仅时间窗长度）。
            tactile_history=TABERO_TACTILE_HISTORY,
            # tacforce：默认只在 decoder/suffix 侧注入 tactile token（历史指力），
            # prefix 侧只包含视觉 + 语言。
            tactile_in_prefix_only=False,
            tactile_loss_weight=TACTILE_LOSS_WEIGHT,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=1.25e-5,
            decay_lr=1.25e-6,
        ),
        data=TaberoTacForceDataConfig(
            repo_id="NathanWu7/tabero",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params",
            missing_regex=".*",
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_lora_noforce_taforce",
        # 与 pi0_libero_low_mem_finetune 使用相同的 LoRA 配置，但在 Tabero 力矩数据上
        # 只使用前 7 维关节动作，不使用 tactile。
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # 使用与 force 版相同的 EXPERT_HIS_C_FUT loss 拆分逻辑，但关闭力的监督：
            # - effective_action_dim=13：前 7 维是真实关节动作，后 6 维作为“力槽位”；
            # - tactile_type=EXPERT_HIS_C_FUT：在 compute_loss 中按 [7 动作 + 6 力] 拆分；
            # - tactile_dim_in=0：不创建 tactile token 相关 Linear，只启用 loss 拆分逻辑；
            # - tactile_loss_weight=0.0：力的 loss 权重为 0，只剩 7 维动作 loss。
            effective_action_dim=13,
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            tactile_dim_in=0,
            tactile_loss_weight=0.0,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=1.25e-5,
            decay_lr=1.25e-6,
        ),
        data=LeRobotLiberoNoTactileDataConfig(
            repo_id="NathanWu7/tabero_force",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_lora_force_taforce",
        # 带力矩历史 + 未来动作/力联合预测的 LoRA 微调配置。
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # 你当前数据里 action = 7 动作 + 6 力，一共 13 维。
            # 模型内部 action_dim 仍然保持 32（与官方 checkpoint 对齐），
            # 但通过 effective_action_dim=13 告诉模型：前 13 维才是“有语义的”，
            # 其中前 7 维是动作，第 8–13 维是力矩，其余视为 padding。
            effective_action_dim=13,
            # 与 pi0_libero_low_mem_finetune 保持一致，沿用基线的 action_horizon=50。
            # 启用我们实现的 EXPERT_HIS_C_FUT 模式：
            # - 历史力来自 gripper_force[8,6]，flatten 后经 MLP 做 expert token；
            # - 未来动作/力直接从 actions 的 13 维中学习（前 7 维动作，后 6 维力）。
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            tactile_dim_in=8 * 6,  # gripper_force 的 8 帧历史 * 6 维
            tactile_history=TABERO_TACTILE_HISTORY,
            # 力 / 触觉 loss 的权重，可以在这里直接修改（默认 0.1）。
            tactile_loss_weight=TACTILE_LOSS_WEIGHT,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=1.25e-5,
            decay_lr=1.25e-6,
        ),
        data=LeRobotLiberoTactileDataConfig(
            # 使用你在 Hugging Face 上的 LeRobot 数据集仓库作为 repo_id。
            # 这里直接填 HF Hub 的 dataset id，DataLoader 会通过 fsspec 拉取数据。
            repo_id="NathanWu7/tabero_force",
            # 这里不再强制指定本地 assets_dir，norm_stats 训练后会先保存在本地
            #   ./assets/pi0_libero_force_low_mem_finetune/NathanWu7_tabero_force/...
            # 然后你可以把这整个目录和 checkpoint 一起上传到同一个 HF 仓库。
            base_config=DataConfig(
                # 你的数据里如果没有 task->prompt，这里可以保持 False；有的话可以改成 True。
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # 注意：这里使用 base pi0 checkpoint 来初始化绝大部分权重，
        # 对于新增的 tactile_proj_in/tactile_proj_out 这类在 checkpoint 中不存在的参数，
        # 会通过 missing_regex=".*" 让它们保持模型随机初始化的值。
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params",
            missing_regex=".*",
        ),
        # 初始权重：使用官方 pi0 base checkpoint，然后做 LoRA 微调。
        # 其余设置与上方 pi0_libero_low_mem_finetune 保持一致。
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_noforce_taforce",
        # Pi05 tabero_force 数据，只使用 7 维关节动作，不读触觉力（tactile）。
        # 结构上仿照 pi0_libero_low_mem_finetune_wo_force，但使用 Pi05 模型与相同 tabero_force 数据。
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            # 使用与 force 版相同的 EXPERT_HIS_C_FUT loss 拆分逻辑，但关闭力的监督：
            # - effective_action_dim=13：前 7 维是真实关节动作，后 6 维作为“力槽位”；
            # - tactile_type=EXPERT_HIS_C_FUT：在 compute_loss 中按 [7 动作 + 6 力] 拆分；
            # - tactile_dim_in=0：不创建 tactile token 相关 Linear，只启用 loss 拆分逻辑；
            # - tactile_loss_weight=0.0：力的 loss 权重为 0，只剩 7 维动作 loss。
            effective_action_dim=13,
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            tactile_dim_in=0,
            tactile_loss_weight=0.0,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=1.25e-5,
            decay_lr=1.25e-6,
        ),
        data=LeRobotLiberoNoTactileDataConfig(
            repo_id="NathanWu7/tabero_force",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=".*",
        ),
        num_train_steps=30_000,
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
    ),
    TrainConfig(
        name="pi05_force_taforce_dec",
        # Pi05 力矩（decoder 版）：8×6 tactile 先 flatten→MLP→变成单个 tactile token，
        # 只在 suffix（expert decoder）前拼接，和 state token + action tokens 串联。
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            effective_action_dim=13,
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            tactile_dim_in=8 * 6,
            tactile_history=TABERO_TACTILE_HISTORY,
            tactile_loss_weight=TACTILE_LOSS_WEIGHT,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=1.25e-5,
            decay_lr=1.25e-6,
        ),
        data=LeRobotLiberoTactileDataConfig(
            repo_id="NathanWu7/tabero_force",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=".*",
        ),
        num_train_steps=30_000,
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
    ),
    TrainConfig(
        name="pi05_force_tacforce_enc",
        # Pi05 力矩（encoder 版）：8×6 tactile 先 flatten→MLP→变成单个 tactile token，
        # 只在 prefix 侧与视觉 token、语言 token、state token 串联，suffix 仅看动作 + 时间。
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            effective_action_dim=13,
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            tactile_dim_in=8 * 6,
            tactile_history=TABERO_TACTILE_HISTORY,
            tactile_in_prefix_only=True,
            tactile_loss_weight=TACTILE_LOSS_WEIGHT,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=1.25e-5,
            decay_lr=1.25e-6,
        ),
        data=LeRobotLiberoTactileDataConfig(
            repo_id="NathanWu7/tabero_force",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=".*",
        ),
        num_train_steps=30_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
    ),
    #

    #
    # RoboArena configs.
    #
    *roboarena_config.get_roboarena_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
