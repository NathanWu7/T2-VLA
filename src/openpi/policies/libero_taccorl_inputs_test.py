import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import libero_policy


def _sample(motion):
    return {
        "image": np.zeros((12, 16, 3), dtype=np.uint8),
        "wrist_image": np.ones((3, 12, 16), dtype=np.float32),
        "state": np.arange(7, dtype=np.float32),
        "actions": np.zeros((2, 13), dtype=np.float32),
        "prompt": "Pick up the object.",
        "tactile_marker_motion": motion,
    }


@pytest.mark.parametrize(
    ("offset", "active", "frame", "threshold", "minimum", "expected"),
    [
        ((0.0, 0.0), 6, 1, 1.0, 5, False),
        ((0.5, 0.5), 5, 1, 1.0, 5, False),
        ((1.0, 0.0), 5, 1, 1.0, 5, False),
        ((1.001, 0.0), 4, 1, 1.0, 5, False),
        ((1.001, 0.0), 5, 1, 1.0, 5, True),
        ((0.75, 0.75), 5, 8, 1.0, 5, True),
        ((-1.001, 0.0), 5, 4, 1.0, 5, True),
        ((2.0, 0.0), 3, 8, 2.0, 3, False),
        ((2.001, 0.0), 3, 8, 2.0, 3, True),
    ],
)
def test_contact_gate_boundaries(offset, active, frame, threshold, minimum, expected):
    motion = np.full((9, 6, 2), 100.0, dtype=np.float32)
    motion[frame, :active] += offset
    result = libero_policy.TaberoTacCoRLInputs(
        model_type=_model.ModelType.PI05,
        contact_threshold=threshold,
        contact_min_active=minimum,
    )(_sample(motion))
    assert isinstance(result["tactile_contact"], np.bool_)
    assert result["tactile_contact"].shape == ()
    assert bool(result["tactile_contact"]) == expected


def test_contact_requires_minimum_markers_in_one_frame():
    motion = np.full((9, 6, 2), 100.0, dtype=np.float32)
    for frame in range(1, 9):
        motion[frame, frame % 6, 0] += 20.0
    result = libero_policy.TaberoTacCoRLInputs(model_type=_model.ModelType.PI05)(_sample(motion))
    assert not result["tactile_contact"]


@pytest.mark.parametrize("model_type", [_model.ModelType.PI0, _model.ModelType.PI05, _model.ModelType.PI0_FAST])
@pytest.mark.parametrize("include_optional", [False, True])
def test_shared_inputs_match_tacfield(model_type, include_optional):
    motion = np.arange(9 * 440 * 2, dtype=np.float32).reshape(9, 440, 2)
    original = motion.copy()
    sample = _sample(motion)
    if not include_optional:
        sample.pop("actions")
        sample.pop("prompt")
    expected = libero_policy.TaberoTacFieldInputs(model_type=model_type)(sample)
    result = libero_policy.TaberoTacCoRLInputs(model_type=model_type)(sample)
    assert result.keys() == expected.keys() | {"tactile_contact"}
    assert result["tactile_prefix"].shape == (9, 440 * 2)
    for key, value in expected.items():
        if isinstance(value, dict):
            assert result[key].keys() == value.keys()
            for subkey, subvalue in value.items():
                np.testing.assert_array_equal(result[key][subkey], subvalue)
        else:
            np.testing.assert_array_equal(result[key], value)
    np.testing.assert_array_equal(motion, original)
