from types import SimpleNamespace

import pytest
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.tactile_encoder_pytorch import MarkerTokenEncoder, TactilePrefixCrossAttention


class _FakePaliGemma:
    def __init__(self):
        self.paligemma = SimpleNamespace(
            language_model=SimpleNamespace(config=SimpleNamespace(_attn_implementation=None))
        )

    def forward(self, **kwargs):
        del kwargs
        return None, "prefix-cache"


def test_sample_actions_aligns_supplied_noise_to_action_projection_dtype():
    model = object.__new__(PI0Pytorch)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(action_horizon=1, action_dim=32)
    model.action_in_proj = torch.nn.Linear(32, 4, dtype=torch.float32)
    model.paligemma_with_expert = _FakePaliGemma()
    state = torch.zeros(1, 7)
    model._preprocess_observation = lambda observation, train: (  # noqa: SLF001
        [],
        [],
        None,
        None,
        state,
        None,
    )
    model.embed_prefix = lambda *args: (
        torch.zeros(1, 1, 4),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.bool),
    )
    model._prepare_attention_masks_4d = lambda mask: mask[:, None, :, :]  # noqa: SLF001
    denoise_dtypes = []

    def denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep):
        del state, prefix_pad_masks, past_key_values, timestep
        denoise_dtypes.append(x_t.dtype)
        return torch.zeros_like(x_t)

    model.denoise_step = denoise_step
    observation = SimpleNamespace(state=state)
    noise = torch.zeros(1, 1, 32, dtype=torch.bfloat16)

    actions = PI0Pytorch.sample_actions(
        model,
        torch.device("cpu"),
        observation,
        noise=noise,
        num_steps=1,
    )

    assert denoise_dtypes == [torch.float32]
    assert actions.dtype == torch.float32
    assert noise.dtype == torch.bfloat16


class _StubBackbone:
    def embed_image(self, image):
        return image

    def embed_language_tokens(self, tokens):
        return tokens


def _stub_taccorl_model():
    model = object.__new__(PI0Pytorch)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(action_horizon=3, action_dim=4, tactile_fusion="taccorl")
    model.pi05 = True
    model.gradient_checkpointing_enabled = False
    model.paligemma_with_expert = _StubBackbone()
    model.action_in_proj = torch.nn.Linear(4, 8)
    model.time_mlp_in = torch.nn.Linear(8, 8)
    model.time_mlp_out = torch.nn.Linear(8, 8)
    model.marker_token_encoder = MarkerTokenEncoder(
        in_dim=4,
        history_len=2,
        has_reference_frame=True,
        diff_from_reference=False,
        prefix_width=8,
        suffix_width=8,
        prefix_num_tokens=2,
        suffix_num_tokens=2,
    )
    model.tactile_cross_attn = TactilePrefixCrossAttention(8, 2)
    return model


@pytest.mark.parametrize(
    "tactile_contact, expected_mask",
    [
        (torch.tensor([True, False]), [[True, True], [False, False]]),
        (None, [[True, True], [True, True]]),
    ],
)
def test_taccorl_prefix_suffix_masks_and_closed_gate_identity(tactile_contact, expected_mask):
    model = _stub_taccorl_model()
    images = [torch.zeros(2, 2, 8)]
    image_masks = [torch.ones(2, dtype=torch.bool)]
    lang_tokens = torch.zeros(2, 1, 8)
    lang_masks = torch.ones(2, 1, dtype=torch.bool)
    tactile = torch.randn(2, 3, 4)
    marker_tokens = model._encode_marker_tokens(tactile)
    prefix, prefix_mask, prefix_ar = model.embed_prefix(
        images,
        image_masks,
        lang_tokens,
        lang_masks,
        tactile,
        marker_tokens=marker_tokens,
        tactile_contact=tactile_contact,
    )
    suffix, suffix_mask, suffix_ar, _ = model.embed_suffix(
        torch.zeros(2, 4),
        torch.zeros(2, 3, 4),
        torch.ones(2),
        marker_tokens=marker_tokens,
        tactile_contact=tactile_contact,
    )
    assert prefix.shape == (2, 5, 8)
    assert suffix.shape == (2, 5, 8)
    assert torch.equal(prefix_ar, torch.zeros_like(prefix_ar))
    assert torch.equal(suffix_ar, torch.tensor([[True, False, True, False, False]] * 2))
    assert torch.equal(prefix_mask[:, -2:], torch.tensor(expected_mask))
    assert torch.equal(suffix_mask[:, :2], torch.tensor(expected_mask))

    # With alpha nonzero, closed contact still leaves the base image/language
    # embeddings unchanged and masks the inserted tokens completely.
    assert model.tactile_cross_attn.alpha.shape == (1,)
    model.tactile_cross_attn.alpha.data.fill_(0.7)
    marker_encoder, cross_attn = model.marker_token_encoder, model.tactile_cross_attn
    model.marker_token_encoder = None
    model.tactile_cross_attn = None
    model.tactile_prefix_encoder = None
    base, base_mask, _ = model.embed_prefix(images, image_masks, lang_tokens, lang_masks, tactile_contact=None)
    model.marker_token_encoder, model.tactile_cross_attn = marker_encoder, cross_attn
    nprefix, nmask, _ = model.embed_prefix(
        images,
        image_masks,
        lang_tokens,
        lang_masks,
        tactile,
        marker_tokens=marker_tokens,
        tactile_contact=torch.tensor([False, False]),
    )
    torch.testing.assert_close(nprefix[:, :-2], base)
    torch.testing.assert_close(nmask[:, :-2], base_mask)


@pytest.mark.parametrize(
    "method_name, args",
    [
        ("embed_prefix", (None, None, None, None)),
        ("embed_suffix", (None, None, None)),
        ("denoise_step", (None, None, None, None, None)),
    ],
)
def test_taccorl_requires_explicit_tactile_contact(method_name, args):
    model = _stub_taccorl_model()
    with pytest.raises(ValueError, match=rf"{method_name} requires explicit tactile_contact"):
        getattr(model, method_name)(*args)
