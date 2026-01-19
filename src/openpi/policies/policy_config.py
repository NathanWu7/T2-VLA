import logging
import os
import pathlib
from typing import Any

import flax.nnx as nnx
import jax.numpy as jnp
import jax
import dataclasses as _dc

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import weight_loaders as _weight_loaders
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    modelbase: str = "ckpt",
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        if modelbase == "ckpt":
            model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
        else:
            # Serving-time ablation:
            # - start from official base weights (pi0_base / pi05_base)
            # - only override LoRA + tactile tokenizer weights from the provided checkpoint dir
            # This keeps the backbone close to the base model while reusing your learned adapters.

            base_choice = str(modelbase).lower()
            if base_choice not in ("pi0", "pi05"):
                raise ValueError(f"Unsupported modelbase={modelbase!r}. Use one of: 'ckpt', 'pi0', 'pi05'.")

            # If using pi05 base, switch the model architecture to Pi05.
            # We keep the rest of the config (LoRA variants, tactile streams, dims) unchanged for a fair adapter test.
            derived_config = train_config
            if base_choice == "pi05" and getattr(train_config.model, "pi05", False) is False:
                derived_model = _dc.replace(train_config.model, pi05=True, discrete_state_input=True)
                derived_config = _dc.replace(train_config, model=derived_model)

            # Build a reference param tree by initializing the derived model once.
            init_model = derived_config.model.create(jax.random.key(0))
            ref_params = nnx.state(init_model).to_pure_dict()

            base_params_path = (
                "gs://openpi-assets/checkpoints/pi05_base/params"
                if base_choice == "pi05"
                else "gs://openpi-assets/checkpoints/pi0_base/params"
            )
            loaded_base = _model.restore_params(download.maybe_download(base_params_path), dtype=jnp.bfloat16)
            base_params = _weight_loaders.merge_loaded_params(loaded_base, ref_params, missing_regex=".*")

            # Overlay: only allow LoRA + tactile weights from the finetuned checkpoint.
            overlay = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
            merged = _weight_loaders.override_params_by_regex(
                base_params,
                overlay,
                allow_regex=r".*(lora|tactile).*",
            )

            model = derived_config.model.load(merged)
            train_config = derived_config
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # NOTE:
    # Policy.infer() only returns {"state", "actions"} (plus timing). Some training setups also
    # include additional normalized inputs like tactile_prefix/tactile_suffix in norm_stats.
    # Output-side Unnormalize is strict by design, so we must avoid passing stats for keys that
    # will never appear in the policy output tree.
    output_norm_stats = None
    if norm_stats is not None:
        output_norm_stats = {k: v for k, v in norm_stats.items() if k in ("state", "actions")}

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(output_norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
