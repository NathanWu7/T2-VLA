import pathlib

import numpy as np

from openpi.models import model as _model
from openpi.policies import libero_policy
from openpi.training import config as _config


def test_pi05_tacfield_tabero_matches_xense_replay_contract():
    config = _config.get_config("pi05_lora_tacfield_tabero")

    assert config.model.pi05 is True
    assert config.model.action_horizon == 10
    assert config.model.effective_action_dim == 13
    assert config.model.tactile_prefix_dim_in == 9 * 440 * 2
    assert config.model.tactile_prefix_history == 8
    assert config.model.tactile_prefix_use_reference_frame is True
    assert config.model.tactile_prefix_diff_from_reference is False
    assert config.data.repo_id == "replay_firm_tabero"
    assert config.data.assets.asset_id == "replay_firm_tabero"

    transformed = libero_policy.TaberoTacFieldInputs(
        model_type=_model.ModelType.PI05
    )(
        {
            "image": np.zeros((32, 32, 3), dtype=np.uint8),
            "wrist_image": np.zeros((32, 32, 3), dtype=np.uint8),
            "state": np.zeros(7, dtype=np.float32),
            "actions": np.zeros((10, 13), dtype=np.float32),
            "tactile_marker_motion": np.zeros((9, 440, 2), dtype=np.float32),
            "prompt": "test",
        }
    )

    assert transformed["state"].shape == (7,)
    assert transformed["actions"].shape == (10, 13)
    assert transformed["tactile_prefix"].shape == (9, 880)
    assert set(transformed["image"]) == {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }


def test_pi05_tacfield_xarm_gripper_config_uses_dedicated_asset():
    config = _config.get_config("pi05_lora_tacfield_tabero_xarm_gripper")

    assert config.model.model_type == _model.ModelType.PI05
    assert config.model.action_horizon == 10
    assert config.model.discrete_state_input is True
    assert config.model.effective_action_dim == 13
    assert config.model.tactile_prefix_dim_in == 9 * 440 * 2
    assert config.data.repo_id == "replay_firm_tabero_xarm_gripper"
    assert config.data.assets.asset_id == "replay_firm_tabero_xarm_gripper"
    assert config.num_train_steps == 5000


def test_pi05_libero_tacfield_xarm_gripper_training_config():
    config = _config.get_config("pi05_libero_lora_tacfield_tabero_xarm_gripper")

    assert config.model.model_type == _model.ModelType.PI05
    assert config.model.pi05 is True
    assert config.model.action_horizon == 10
    assert config.model.discrete_state_input is True
    assert config.model.effective_action_dim == 13
    assert config.model.tactile_prefix_dim_in == 9 * 440 * 2
    assert config.model.tactile_prefix_history == 8
    assert config.model.tactile_loss_weight == 0.01
    assert config.data.repo_id == "replay_firm_tabero_xarm_gripper"
    assert config.data.assets.asset_id == "replay_firm_tabero_xarm_gripper"
    assert config.weight_loader.params_path == "/data/home/sim6g/code/tabero/models/pi05_libero/params"
    assert config.weight_loader.missing_regex == r".*(?:lora|tactile_prefix_encoder).*"
    assert config.batch_size == 8
    assert config.num_train_steps == 80_000
    assert config.seed == 0
    assert config.fsdp_devices == 2


def test_pi05_libero_tacfield_xarm_gripper_repaired_v1_training_config():
    config = _config.get_config("pi05_libero_lora_tacfield_tabero_xarm_gripper_repaired_v1")

    assert config.model.model_type == _model.ModelType.PI05
    assert config.model.pi05 is True
    assert config.model.action_horizon == 10
    assert config.model.discrete_state_input is True
    assert config.model.effective_action_dim == 13
    assert config.model.tactile_prefix_dim_in == 9 * 440 * 2
    assert config.model.tactile_prefix_history == 8
    assert config.model.tactile_loss_weight == 0.01
    assert config.model.padding_loss_weight == 1.0
    assert config.model.expert_his_c_fut_loss_mode == "weighted_full"
    assert config.data.repo_id == "replay_firm_tabero_xarm_gripper_repaired_v1"
    assert config.data.assets.asset_id == "replay_firm_tabero_xarm_gripper_repaired_v1"
    workspace_root = pathlib.Path(_config.__file__).resolve().parents[4]
    assert config.weight_loader.params_path == str((workspace_root / "models/pi05_libero/params").resolve())
    assert config.weight_loader.missing_regex == r".*(?:lora|tactile_prefix_encoder).*"
    assert config.batch_size == 8
    assert config.num_train_steps == 20_000
    assert config.seed == 0
    assert config.fsdp_devices == 1
    assert config.tensorboard_enabled is True


def test_pi0_tacfield_xarm_gripper_config_uses_50_step_continuous_state():
    config = _config.get_config("pi0_lora_tacfield_tabero_xarm_gripper")

    assert config.model.model_type == _model.ModelType.PI0
    assert config.model.pi05 is False
    assert config.model.action_horizon == 50
    assert config.model.discrete_state_input is False
    assert config.model.max_token_len == 48
    assert config.model.effective_action_dim == 13
    assert config.model.tactile_prefix_dim_in == 9 * 440 * 2
    assert config.model.tactile_prefix_history == 8
    assert config.data.repo_id == "replay_firm_tabero_xarm_gripper"
    assert config.data.assets.asset_id == "replay_firm_tabero_xarm_gripper"
    assert config.num_train_steps == 5000

    transformed = libero_policy.TaberoTacFieldInputs(
        model_type=_model.ModelType.PI0
    )(
        {
            "image": np.zeros((32, 32, 3), dtype=np.uint8),
            "wrist_image": np.zeros((32, 32, 3), dtype=np.uint8),
            "state": np.zeros(7, dtype=np.float32),
            "actions": np.zeros((50, 13), dtype=np.float32),
            "tactile_marker_motion": np.zeros((9, 440, 2), dtype=np.float32),
            "prompt": "test",
        }
    )
    assert transformed["state"].shape == (7,)
    assert transformed["actions"].shape == (50, 13)
    assert transformed["tactile_prefix"].shape == (9, 880)
