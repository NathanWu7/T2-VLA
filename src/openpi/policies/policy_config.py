import contextlib
import ctypes
import hashlib
import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.policies.tabero_dsrl_policy as _tabero_dsrl_policy
import openpi.policies.tabero_lora_bundle as _tabero_lora_bundle
import openpi.policies.tabero_rlt_policy as _tabero_rlt_policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


class _InotifyCheckpointWatch:
    _CHANGE_MASK = 0x00000002 | 0x00000004 | 0x00000008 | 0x00000400 | 0x00000800

    def __init__(self, stable_path: str):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            inotify_init1 = libc.inotify_init1
            inotify_add_watch = libc.inotify_add_watch
        except AttributeError as error:
            raise ValueError("Tabero DSRL stable checkpoint loading requires Linux inotify.") from error
        inotify_init1.argtypes = [ctypes.c_int]
        inotify_init1.restype = ctypes.c_int
        inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        inotify_add_watch.restype = ctypes.c_int

        self._fd = inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self._fd < 0:
            error_number = ctypes.get_errno()
            raise ValueError(
                f"Tabero DSRL stable checkpoint inotify initialization failed: {os.strerror(error_number)}."
            )
        watch_descriptor = inotify_add_watch(self._fd, os.fsencode(stable_path), self._CHANGE_MASK)
        if watch_descriptor < 0:
            error_number = ctypes.get_errno()
            os.close(self._fd)
            self._fd = -1
            raise ValueError(f"Tabero DSRL stable checkpoint inotify watch failed: {os.strerror(error_number)}.")

    def drain_changed(self) -> bool:
        changed = False
        while True:
            try:
                events = os.read(self._fd, 64 * 1024)
            except BlockingIOError:
                return changed
            except OSError as error:
                raise ValueError("Tabero DSRL stable checkpoint inotify read failed.") from error
            if not events:
                return changed
            changed = True

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


def _checkpoint_stat_fingerprint(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_ctime_ns,
        stat.st_mtime_ns,
    )


def _sha256_fd(fd: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(8 * 1024 * 1024, size - offset), offset)
        if not chunk:
            raise ValueError("Tabero DSRL base checkpoint changed while it was being hashed.")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def _stable_dsrl_checkpoint(path: str | pathlib.Path):
    try:
        import fcntl
    except ImportError as error:
        raise ValueError("Tabero DSRL stable checkpoint loading requires fcntl.") from error

    checkpoint_path = pathlib.Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(checkpoint_path, flags)
    except OSError as error:
        raise ValueError(f"Tabero DSRL base checkpoint could not be opened: {checkpoint_path}") from error
    watch = None
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        proc_path = f"/proc/self/fd/{fd}"
        if not os.path.isfile(proc_path):
            raise ValueError(f"Tabero DSRL stable checkpoint fd is unavailable: {proc_path}")
        watch = _InotifyCheckpointWatch(proc_path)
        before = os.fstat(fd)
        before_fingerprint = _checkpoint_stat_fingerprint(before)
        before_hash = _sha256_fd(fd, before.st_size)
        if watch.drain_changed():
            raise ValueError("Tabero DSRL base checkpoint changed during load.")
        yield proc_path, before_hash

        if watch.drain_changed():
            raise ValueError("Tabero DSRL base checkpoint changed during load.")
        after = os.fstat(fd)
        after_hash = _sha256_fd(fd, after.st_size)
        if watch.drain_changed():
            raise ValueError("Tabero DSRL base checkpoint changed during load.")
        if _checkpoint_stat_fingerprint(after) != before_fingerprint or after_hash != before_hash:
            raise ValueError("Tabero DSRL base checkpoint changed during load.")

        try:
            path_fd = os.open(checkpoint_path, flags)
        except OSError as error:
            raise ValueError("Tabero DSRL base checkpoint changed during load.") from error
        try:
            path_stat = os.fstat(path_fd)
            path_hash = _sha256_fd(path_fd, path_stat.st_size)
        finally:
            os.close(path_fd)
        if watch.drain_changed():
            raise ValueError("Tabero DSRL base checkpoint changed during load.")
        if (path_stat.st_dev, path_stat.st_ino) != (before.st_dev, before.st_ino) or path_hash != before_hash:
            raise ValueError("Tabero DSRL base checkpoint changed during load.")
    finally:
        try:
            if watch is not None:
                watch.close()
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    lora_bundle_path: pathlib.Path | str | None = None,
    rlt_bundle_path: pathlib.Path | str | None = None,
    dsrl_bundle_path: pathlib.Path | str | None = None,
) -> _policy.Policy | _tabero_dsrl_policy.TaberoDSRLPolicy:
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
        lora_bundle_path: Optional versioned RLinf PEFT-LoRA bundle. The declared
            PyTorch base is loaded with a controlled partial-load contract, then
            non-LoRA trainables and adapters are applied and audited before use.
        rlt_bundle_path: Optional exported Tabero RLT bundle. PyTorch PI0 is used as the
            frozen reference model and its normalized actions are replaced by the RLT actor.
        dsrl_bundle_path: Optional audited, allowlisted Tabero DSRL-SAC actor bundle. The actor
            consumes raw image/state/tactile observations and supplies deterministic PI0 noise.

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safetensors" in the checkpoint directory.
    """
    enabled_bundles = sum(path is not None for path in (lora_bundle_path, rlt_bundle_path, dsrl_bundle_path))
    if enabled_bundles > 1:
        raise ValueError("Tabero LoRA, RLT, and DSRL bundles are mutually exclusive.")
    if dsrl_bundle_path is not None and sample_kwargs is not None and "num_steps" in sample_kwargs:
        num_steps = sample_kwargs["num_steps"]
        if type(num_steps) is not int or num_steps != 10:
            raise ValueError(f"Tabero DSRL bundle serving requires num_steps=10; got {num_steps!r}.")
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.isfile(weight_path)
    if (lora_bundle_path is not None or rlt_bundle_path is not None or dsrl_bundle_path is not None) and not is_pytorch:
        raise ValueError(
            "Tabero LoRA/RLT/DSRL bundle serving requires an explicit PyTorch checkpoint "
            "directory containing model.safetensors."
        )

    logging.info("Loading model...")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dsrl_base_model_sha256 = None
    loaded_lora_bundle = None
    if is_pytorch:
        if dsrl_bundle_path is not None:
            with _stable_dsrl_checkpoint(weight_path) as (stable_weight_path, dsrl_base_model_sha256):
                model = train_config.model.load_pytorch(train_config, stable_weight_path)
        elif lora_bundle_path is not None:
            if data_config.asset_id is None:
                raise ValueError("Asset id is required to load a Tabero LoRA bundle.")
            resolved_bundle_path = download.maybe_download(str(lora_bundle_path))
            loaded_lora_bundle = _tabero_lora_bundle.load_lora_bundle_model(
                train_config,
                weight_path,
                resolved_bundle_path,
                base_config_name=train_config.name,
                norm_asset_id=data_config.asset_id,
            )
            model = loaded_lora_bundle.model
        else:
            model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_assets_dir = (
            loaded_lora_bundle.norm_assets_dir if loaded_lora_bundle is not None else checkpoint_dir / "assets"
        )
        norm_stats = _checkpoints.load_norm_stats(norm_assets_dir, data_config.asset_id)

    if is_pytorch and rlt_bundle_path is not None:
        model = _tabero_rlt_policy.TaberoRLTPolicyModel.from_bundle(
            model,
            rlt_bundle_path,
            base_model_path=checkpoint_dir,
            base_config_name=train_config.name,
            norm_asset_id=data_config.asset_id,
            use_quantile_norm=data_config.use_quantile_norm,
        )

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

    policy_metadata = dict(train_config.policy_metadata or {})
    if loaded_lora_bundle is not None:
        policy_metadata["tabero_lora_bundle"] = loaded_lora_bundle.metadata

    base_policy = _policy.Policy(
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
        metadata=policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
    if dsrl_bundle_path is not None:
        actor = _tabero_dsrl_policy.TaberoDSRLActor.from_bundle(
            dsrl_bundle_path,
            base_checkpoint_dir=checkpoint_dir,
            base_model_sha256=dsrl_base_model_sha256,
            device=pytorch_device,
        )
        return _tabero_dsrl_policy.TaberoDSRLPolicy(base_policy, actor)
    return base_policy
