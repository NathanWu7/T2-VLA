from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.models import model as model_lib
from openpi.models_pytorch import preprocessing_pytorch


def _observation(image):
    return SimpleNamespace(
        images={"left_wrist_0_rgb": image},
        image_masks={"left_wrist_0_rgb": torch.ones(image.shape[0], dtype=torch.bool)},
        state=torch.zeros(image.shape[0], 1),
        tokenized_prompt=torch.zeros(image.shape[0], 2, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(image.shape[0], 2, dtype=torch.bool),
        token_ar_mask=None,
        token_loss_mask=None,
    )


def test_torch_image_noise_has_requested_scale_and_eval_is_unchanged():
    image = torch.zeros(8, 3, 32, 32)
    obs = _observation(image)
    clean = preprocessing_pytorch.preprocess_observation_pytorch(
        obs, train=False, image_noise_std=0.03, image_keys=("left_wrist_0_rgb",), image_resolution=(32, 32)
    ).images["left_wrist_0_rgb"]
    assert torch.equal(clean, image)

    torch.manual_seed(0)
    noisy = preprocessing_pytorch.preprocess_observation_pytorch(
        obs, train=True, image_noise_std=0.03, image_keys=("left_wrist_0_rgb",), image_resolution=(32, 32)
    ).images["left_wrist_0_rgb"]
    assert float(noisy.min()) >= -1.0 and float(noisy.max()) <= 1.0
    assert abs(float(noisy.std()) - 0.06) < 0.01

    torch.manual_seed(1)
    no_noise = preprocessing_pytorch.preprocess_observation_pytorch(
        obs, train=True, image_noise_std=0.0, image_keys=("left_wrist_0_rgb",), image_resolution=(32, 32)
    ).images["left_wrist_0_rgb"]
    torch.manual_seed(1)
    baseline = preprocessing_pytorch.preprocess_observation_pytorch(
        obs, train=True, image_keys=("left_wrist_0_rgb",), image_resolution=(32, 32)
    ).images["left_wrist_0_rgb"]
    assert torch.equal(no_noise, baseline)


def test_jax_image_noise_has_requested_scale_eval_is_unchanged_and_zero_is_noop(monkeypatch):
    class _IdentityChain:
        def __init__(self, *transforms):
            del transforms

        def __call__(self, rng, value):
            del rng
            return value

    monkeypatch.setattr(model_lib.augmax, "Chain", _IdentityChain)
    image = jnp.zeros((8, 32, 32, 3), dtype=jnp.float32)
    obs = model_lib.Observation(
        images={"left_wrist_0_rgb": image},
        image_masks={"left_wrist_0_rgb": jnp.ones((8,), dtype=jnp.bool_)},
        state=jnp.zeros((8, 1), dtype=jnp.float32),
    )
    kwargs = dict(image_keys=("left_wrist_0_rgb",), image_resolution=(32, 32))
    eval_out = model_lib.preprocess_observation(None, obs, train=False, image_noise_std=0.03, **kwargs)
    assert np.array_equal(np.asarray(eval_out.images["left_wrist_0_rgb"]), np.asarray(image))

    clean = model_lib.preprocess_observation(jax.random.key(0), obs, train=True, image_noise_std=0.0, **kwargs)
    clean_again = model_lib.preprocess_observation(jax.random.key(0), obs, train=True, **kwargs)
    np.testing.assert_array_equal(clean.images["left_wrist_0_rgb"], clean_again.images["left_wrist_0_rgb"])

    noisy = model_lib.preprocess_observation(jax.random.key(1), obs, train=True, image_noise_std=0.03, **kwargs)
    values = np.asarray(noisy.images["left_wrist_0_rgb"])
    assert values.min() >= -1.0 and values.max() <= 1.0
    assert abs(float(values.std()) - 0.06) < 0.01
