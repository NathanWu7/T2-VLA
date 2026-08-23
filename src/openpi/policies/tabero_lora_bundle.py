"""Load versioned RLinf PEFT-LoRA bundles into the OpenPI PyTorch policy.

The normal OpenPI checkpoint loader remains strict. This module owns the only
controlled partial-load path: a bundle may supply exactly the architectural
keys that are absent from its declared base checkpoint, then PEFT adapters are
merged and the reconstructed model is checked against the exporter contract.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
from importlib import metadata as importlib_metadata
import json
import pathlib
from typing import Any

from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import load_file
import torch

from openpi.models_pytorch import pi0_pytorch

BUNDLE_FORMAT = "tabero_rlinf_openpi_lora_bundle"
BUNDLE_FORMAT_VERSION = 1
RL_ONLY_PREFIXES = (
    "value_head.",
    "noise_head.",
    "dsrl_",
    "actor_image_encoder.",
    "actor_state_encoder.",
    "critic_image_encoder.",
    "critic_state_encoder.",
    "q_head.",
)
DROP_EXACT_KEYS = {
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight",
}

_SAFETENSORS_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


@dataclasses.dataclass(frozen=True)
class LoadedLoraBundle:
    model: torch.nn.Module
    norm_assets_dir: pathlib.Path
    metadata: dict[str, Any]


def _sha256_file(path: pathlib.Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"LoRA bundle file not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: pathlib.Path) -> dict:
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read LoRA bundle JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"LoRA bundle JSON must contain an object: {path}")
    return value


def _bundle_file(bundle_root: pathlib.Path, relative_path: str) -> pathlib.Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"Invalid LoRA bundle relative path: {relative_path!r}")
    candidate = (bundle_root / relative_path).resolve()
    try:
        candidate.relative_to(bundle_root)
    except ValueError as error:
        raise ValueError(f"LoRA bundle path escapes the bundle root: {relative_path!r}") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"LoRA bundle file not found: {candidate}")
    return candidate


def _verify_bundle_checksums(bundle_root: pathlib.Path) -> dict[str, str]:
    checksums = _load_json(bundle_root / "checksums.json")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in checksums.items()):
        raise ValueError("LoRA bundle checksums.json must map paths to SHA256 strings.")
    actual_files = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file() and path.name != "checksums.json"
    }
    if set(checksums) != actual_files:
        raise ValueError(
            "LoRA bundle file inventory differs from checksums.json; "
            f"missing={sorted(actual_files - set(checksums))[:10]}, "
            f"unexpected={sorted(set(checksums) - actual_files)[:10]}"
        )
    mismatches = []
    for relative_path, expected_hash in sorted(checksums.items()):
        if len(expected_hash) != 64:
            mismatches.append((relative_path, expected_hash, "invalid_sha256"))
            continue
        actual_hash = _sha256_file(_bundle_file(bundle_root, relative_path))
        if actual_hash != expected_hash:
            mismatches.append((relative_path, expected_hash, actual_hash))
    if mismatches:
        raise ValueError(f"LoRA bundle checksum mismatch: {mismatches[:5]}")
    return checksums


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"Unsupported LoRA bundle format: {manifest.get('format')!r}")
    if manifest.get("format_version") != BUNDLE_FORMAT_VERSION:
        raise ValueError(f"Unsupported LoRA bundle format_version: {manifest.get('format_version')!r}")
    target = manifest.get("lora_target")
    if target not in {"paligemma", "action_expert", "both"}:
        raise ValueError(f"Unsupported LoRA bundle target: {target!r}")
    installed_peft = importlib_metadata.version("peft")
    if manifest.get("peft_version") != installed_peft:
        raise ValueError(
            f"LoRA bundle PEFT version mismatch: bundle={manifest.get('peft_version')!r}, installed={installed_peft!r}"
        )


def _validate_base_contract(manifest: Mapping[str, Any], base_weight_path: pathlib.Path) -> set[str]:
    base = manifest.get("base_model")
    if not isinstance(base, Mapping):
        raise ValueError("LoRA bundle manifest is missing base_model.")
    expected_hash = base.get("model_sha256")
    if _sha256_file(base_weight_path) != expected_hash:
        raise ValueError("LoRA bundle base model SHA256 mismatch; the adapter cannot be applied to this checkpoint.")
    config_file = base.get("config_file")
    config_hash = base.get("config_sha256")
    if config_file is not None:
        if config_file != "config.json" or not isinstance(config_hash, str):
            raise ValueError("LoRA bundle base config contract is invalid.")
        if _sha256_file(base_weight_path.parent / config_file) != config_hash:
            raise ValueError("LoRA bundle base config SHA256 mismatch.")
    with safe_open(base_weight_path, framework="pt", device="cpu") as handle:
        base_keys = set(handle.keys())
    if base.get("model_tensor_count") != len(base_keys):
        raise ValueError("LoRA bundle base model tensor count mismatch.")
    return base_keys


def _expected_adapter_targets(lora_target: str) -> set[str]:
    if lora_target == "both":
        return {"paligemma", "action_expert"}
    return {lora_target}


def _validate_policy_contract(
    manifest: Mapping[str, Any], train_config, config_name: str, norm_asset_id: str
) -> pathlib.PurePosixPath:
    contract = manifest.get("policy_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("LoRA bundle manifest is missing policy_contract.")
    if contract.get("config_name") != config_name:
        raise ValueError(
            f"LoRA bundle config name mismatch: bundle={contract.get('config_name')!r}, requested={config_name!r}"
        )
    if contract.get("norm_asset_id") != norm_asset_id:
        raise ValueError(
            "LoRA bundle normalization asset mismatch: "
            f"bundle={contract.get('norm_asset_id')!r}, requested={norm_asset_id!r}"
        )
    model_contract_fields = (
        "action_horizon",
        "effective_action_dim",
        "tactile_prefix_dim_in",
        "tactile_prefix_history",
    )
    for field in model_contract_fields:
        expected = getattr(train_config.model, field, None)
        actual = contract.get(field)
        if expected != actual:
            raise ValueError(f"LoRA bundle policy contract {field} mismatch: bundle={actual!r}, config={expected!r}")
    expected_family = "pi05" if bool(getattr(train_config.model, "pi05", False)) else "pi0"
    if contract.get("model_family") not in {None, expected_family}:
        raise ValueError(
            f"LoRA bundle model family mismatch: bundle={contract.get('model_family')!r}, config={expected_family!r}"
        )
    norm_path = contract.get("norm_stats_path")
    if not isinstance(norm_path, str):
        raise ValueError("LoRA bundle norm_stats_path must be a string.")
    return pathlib.PurePosixPath(norm_path)


def _load_controlled_base(
    train_config,
    base_weight_path: pathlib.Path,
    *,
    expected_extra_keys: set[str],
    base_keys: set[str],
) -> torch.nn.Module:
    model = pi0_pytorch.PI0Pytorch(config=train_config.model)
    base_state = load_file(base_weight_path, device="cpu")
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    model_keys = set(model.state_dict())
    tied_alias_missing = (DROP_EXACT_KEYS & model_keys) - base_keys
    expected_missing = (expected_extra_keys - base_keys) | tied_alias_missing
    if set(missing) != expected_missing or unexpected:
        raise ValueError(
            "LoRA bundle controlled base load mismatch; "
            f"expected_missing={sorted(expected_missing)[:10]}, "
            f"actual_missing={sorted(missing)[:10]}, unexpected={unexpected[:10]}"
        )
    return model


def _validate_safetensors_contract(path: pathlib.Path, contract: Mapping[str, Any]) -> list[str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        schema = [
            {
                "key": key,
                "shape": list(handle.get_slice(key).get_shape()),
                "dtype": str(handle.get_slice(key).get_dtype()),
            }
            for key in keys
        ]
    schema_digest = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if contract.get("tensor_count") != len(keys) or contract.get("schema_sha256") != schema_digest:
        raise ValueError(f"LoRA bundle safetensors schema mismatch: {path}")
    return keys


def _load_extra_trainable(model: torch.nn.Module, bundle_root: pathlib.Path, contract: Mapping[str, Any]) -> None:
    if not isinstance(contract, Mapping):
        raise ValueError("LoRA bundle manifest is missing extra_trainable.")
    path = _bundle_file(bundle_root, contract.get("path"))
    if _sha256_file(path) != contract.get("file_sha256"):
        raise ValueError("LoRA bundle extra_trainable SHA256 mismatch.")
    keys = _validate_safetensors_contract(path, contract)
    expected_keys = contract.get("keys")
    if keys != expected_keys:
        raise ValueError("LoRA bundle extra_trainable keys differ from the manifest.")
    prefixes = contract.get("canonical_prefixes")
    if (not isinstance(prefixes, list) or not all(isinstance(prefix, str) and prefix for prefix in prefixes)) and (
        keys or prefixes != []
    ):
        raise ValueError("LoRA bundle canonical extra prefixes are invalid.")
    invalid = [
        key
        for key in keys
        if "lora_" in key
        or key.startswith(RL_ONLY_PREFIXES)
        or not any(key == prefix or key.startswith(f"{prefix}.") for prefix in prefixes)
    ]
    if invalid:
        raise ValueError(f"LoRA bundle contains invalid extra-trainable keys: {invalid[:10]}")
    extra_state = load_file(path, device="cpu")
    model_state = model.state_dict()
    unexpected = sorted(set(extra_state) - set(model_state))
    shape_mismatches = sorted(
        (key, tuple(extra_state[key].shape), tuple(model_state[key].shape))
        for key in set(extra_state) & set(model_state)
        if extra_state[key].shape != model_state[key].shape
    )
    if unexpected or shape_mismatches:
        raise ValueError(
            "LoRA bundle extra-trainable model mismatch; "
            f"unexpected={unexpected[:10]}, shape_mismatches={shape_mismatches[:10]}"
        )
    with torch.no_grad():
        for key, value in extra_state.items():
            model_state[key].copy_(value)


def _lora_module(model: torch.nn.Module, target_name: str):
    if target_name == "paligemma":
        return (
            model.paligemma_with_expert.paligemma,
            lambda module: setattr(model.paligemma_with_expert, "paligemma", module),
        )
    if target_name == "action_expert":
        return (
            model.paligemma_with_expert.gemma_expert.model,
            lambda module: setattr(model.paligemma_with_expert.gemma_expert, "model", module),
        )
    raise ValueError(f"Unsupported LoRA adapter target: {target_name!r}")


def _apply_adapters(model: torch.nn.Module, bundle_root: pathlib.Path, manifest: Mapping[str, Any]) -> torch.nn.Module:
    adapters = manifest.get("adapters")
    if not isinstance(adapters, Mapping):
        raise ValueError("LoRA bundle manifest is missing adapters.")
    expected_targets = _expected_adapter_targets(manifest["lora_target"])
    if set(adapters) != expected_targets:
        raise ValueError(
            f"LoRA bundle adapter target mismatch; expected={sorted(expected_targets)}, actual={sorted(adapters)}"
        )
    for target_name in ("paligemma", "action_expert"):
        if target_name not in adapters:
            continue
        adapter = adapters[target_name]
        if not isinstance(adapter, Mapping):
            raise ValueError(f"LoRA bundle adapter contract is invalid: {target_name}")
        adapter_dir = _bundle_file(bundle_root, f"{adapter.get('path')}/adapter_config.json").parent
        config = _load_json(adapter_dir / "adapter_config.json")
        weights_path = adapter_dir / "adapter_model.safetensors"
        if _sha256_file(adapter_dir / "adapter_config.json") != adapter.get("config_sha256"):
            raise ValueError(f"LoRA bundle {target_name} adapter config SHA256 mismatch.")
        if _sha256_file(weights_path) != adapter.get("weights_sha256"):
            raise ValueError(f"LoRA bundle {target_name} adapter weights SHA256 mismatch.")
        expected_config = {
            "rank": config.get("r"),
            "alpha": config.get("lora_alpha"),
            "target_modules": sorted(config.get("target_modules") or []),
            "exclude_modules": config.get("exclude_modules"),
        }
        for field, actual in expected_config.items():
            if adapter.get(field) != actual:
                raise ValueError(
                    f"LoRA bundle {target_name} adapter {field} mismatch: "
                    f"manifest={adapter.get(field)!r}, config={actual!r}"
                )
        if not isinstance(adapter.get("rank"), int) or adapter["rank"] <= 0:
            raise ValueError(f"LoRA bundle {target_name} adapter rank is invalid.")
        _validate_safetensors_contract(weights_path, adapter)
        target_module, assign_module = _lora_module(model, target_name)
        peft_model = PeftModel.from_pretrained(
            target_module,
            adapter_dir,
            is_trainable=False,
            # RLinf stores trainable LoRA weights in FP32 over a BF16 frozen
            # base. PEFT's autocast keeps that merge order (FP32 delta, then
            # cast to the base dtype); disabling it changes BF16 rounding.
            autocast_adapter_dtype=True,
            torch_device="cpu",
        )
        assign_module(peft_model.merge_and_unload())
    residual_lora = [key for key in model.state_dict() if "lora_" in key]
    if residual_lora:
        raise ValueError(f"LoRA keys remained after merge: {residual_lora[:10]}")
    return model


def _tensor_bytes(tensor: torch.Tensor) -> memoryview:
    value = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return memoryview(value)


def _validate_final_model(model: torch.nn.Module, contract: Mapping[str, Any]) -> None:
    if not isinstance(contract, Mapping):
        raise ValueError("LoRA bundle manifest is missing final_merged.")
    expected_schema = contract.get("schema")
    if not isinstance(expected_schema, list):
        raise ValueError("LoRA bundle final merged schema must be a list.")
    schema_hash = hashlib.sha256(
        json.dumps(expected_schema, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if schema_hash != contract.get("schema_sha256"):
        raise ValueError("LoRA bundle final merged schema hash mismatch.")
    state = {
        key: value
        for key, value in model.state_dict().items()
        if key not in DROP_EXACT_KEYS and not key.startswith(RL_ONLY_PREFIXES)
    }
    expected_keys = [entry.get("key") for entry in expected_schema]
    if expected_keys != sorted(expected_keys) or len(set(expected_keys)) != len(expected_keys):
        raise ValueError("LoRA bundle final merged schema keys are not unique and sorted.")
    missing = sorted(set(expected_keys) - set(state))
    unexpected = sorted(set(state) - set(expected_keys))
    if missing or unexpected or contract.get("tensor_count") != len(expected_keys):
        raise ValueError(f"LoRA bundle final merged key mismatch; missing={missing[:10]}, unexpected={unexpected[:10]}")
    digest = hashlib.sha256()
    shape_mismatches = []
    for entry in expected_schema:
        key = entry["key"]
        value = state[key]
        expected_shape = tuple(entry.get("shape", ()))
        if tuple(value.shape) != expected_shape:
            shape_mismatches.append((key, tuple(value.shape), expected_shape))
            continue
        dtype_name = entry.get("dtype")
        if dtype_name not in _SAFETENSORS_DTYPES:
            raise ValueError(f"Unsupported LoRA bundle tensor dtype: {dtype_name!r}")
        digest.update(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode())
        digest.update(_tensor_bytes(value.to(_SAFETENSORS_DTYPES[dtype_name])))
    if shape_mismatches:
        raise ValueError(f"LoRA bundle final merged shape mismatch: {shape_mismatches[:10]}")
    if digest.hexdigest() != contract.get("tensor_sha256"):
        raise ValueError("LoRA bundle reconstructed tensor hash differs from the audited merged model.")


def load_lora_bundle_model(
    train_config,
    base_weight_path: str | pathlib.Path,
    bundle_path: str | pathlib.Path,
    *,
    base_config_name: str,
    norm_asset_id: str,
) -> LoadedLoraBundle:
    bundle_root = pathlib.Path(bundle_path).expanduser().resolve()
    if not bundle_root.is_dir():
        raise FileNotFoundError(f"LoRA bundle directory not found: {bundle_root}")
    _verify_bundle_checksums(bundle_root)
    manifest_path = bundle_root / "manifest.json"
    manifest = _load_json(manifest_path)
    _validate_manifest(manifest)

    weight_path = pathlib.Path(base_weight_path).expanduser().resolve()
    base_keys = _validate_base_contract(manifest, weight_path)
    norm_relative_path = _validate_policy_contract(manifest, train_config, base_config_name, norm_asset_id)
    norm_path = _bundle_file(bundle_root, norm_relative_path.as_posix())
    policy_contract = manifest["policy_contract"]
    if _sha256_file(norm_path) != policy_contract.get("norm_stats_sha256"):
        raise ValueError("LoRA bundle normalization stats SHA256 mismatch.")
    expected_norm_path = pathlib.PurePosixPath("assets") / norm_asset_id / "norm_stats.json"
    if norm_relative_path != expected_norm_path:
        raise ValueError(f"LoRA bundle normalization stats must be stored at {expected_norm_path.as_posix()}.")

    extra_contract = manifest.get("extra_trainable")
    if not isinstance(extra_contract, Mapping) or not isinstance(extra_contract.get("keys"), list):
        raise ValueError("LoRA bundle extra-trainable key contract is invalid.")
    expected_extra_keys = set(extra_contract["keys"])
    model = _load_controlled_base(
        train_config,
        weight_path,
        expected_extra_keys=expected_extra_keys,
        base_keys=base_keys,
    )
    _load_extra_trainable(model, bundle_root, extra_contract)
    model = _apply_adapters(model, bundle_root, manifest)
    _validate_final_model(model, manifest.get("final_merged"))

    metadata = {
        "format": manifest["format"],
        "format_version": manifest["format_version"],
        "bundle_path": str(bundle_root),
        "manifest_sha256": _sha256_file(manifest_path),
        "lora_target": manifest["lora_target"],
        "base_model_sha256": manifest["base_model"]["model_sha256"],
        "checkpoint": manifest.get("checkpoint"),
        "policy_contract": policy_contract,
    }
    return LoadedLoraBundle(
        model=model,
        norm_assets_dir=bundle_root / "assets",
        metadata=metadata,
    )
