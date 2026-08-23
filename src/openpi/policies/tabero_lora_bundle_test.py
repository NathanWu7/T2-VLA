from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from peft import LoraConfig
from peft import PeftModel
from peft import get_peft_model
import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch import nn

from openpi.policies import tabero_lora_bundle as bundle


class _TinyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 3, bias=False)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = _TinyBranch()
        self.paligemma_with_expert.gemma_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert.model = _TinyBranch()
        self.extra = nn.Linear(3, 2, bias=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> memoryview:
    return memoryview(tensor.detach().cpu().contiguous().view(torch.uint8).numpy())


def _safetensors_contract(path: Path) -> dict:
    with safe_open(path, framework="pt", device="cpu") as handle:
        schema = [
            {
                "key": key,
                "shape": list(handle.get_slice(key).get_shape()),
                "dtype": str(handle.get_slice(key).get_dtype()),
            }
            for key in sorted(handle.keys())
        ]
    schema_sha256 = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "tensor_count": len(schema),
        "schema_sha256": schema_sha256,
    }


def _model_contract(model: nn.Module) -> dict:
    schema = []
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        entry = {
            "key": key,
            "shape": list(value.shape),
            "dtype": "F32",
        }
        schema.append(entry)
        digest.update(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode())
        digest.update(_tensor_bytes(value.float()))
    return {
        "tensor_count": len(schema),
        "schema": schema,
        "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "tensor_sha256": digest.hexdigest(),
        "reference_model_sha256": "unit-test",
    }


def _target_module(model: _TinyModel, target: str) -> nn.Module:
    if target == "paligemma":
        return model.paligemma_with_expert.paligemma
    return model.paligemma_with_expert.gemma_expert.model


def _assign_target(model: _TinyModel, target: str, module: nn.Module) -> None:
    if target == "paligemma":
        model.paligemma_with_expert.paligemma = module
    else:
        model.paligemma_with_expert.gemma_expert.model = module


def _write_checksums(root: Path) -> None:
    checksums = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }
    (root / "checksums.json").write_text(json.dumps(checksums, indent=2, sort_keys=True))


def _make_bundle(tmp_path: Path, target: str) -> tuple[Path, Path, _TinyModel]:
    torch.manual_seed(7)
    base_model = _TinyModel()
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    base_state = {
        key: value.detach().clone() for key, value in base_model.state_dict().items() if key != "extra.weight"
    }
    save_file(base_state, base_dir / "model.safetensors")
    (base_dir / "config.json").write_text('{"model":"tiny"}\n')

    root = tmp_path / "bundle"
    root.mkdir()
    norm_path = root / "assets" / "tiny_asset" / "norm_stats.json"
    norm_path.parent.mkdir(parents=True)
    norm_path.write_text("{}\n")

    extra_value = torch.full_like(base_model.extra.weight, 0.25)
    save_file({"extra.weight": extra_value}, root / "extra_trainable.safetensors")
    extra_contract = _safetensors_contract(root / "extra_trainable.safetensors")

    targets = ["paligemma", "action_expert"] if target == "both" else [target]
    adapter_contracts = {}
    expected = _TinyModel()
    expected.load_state_dict(base_state, strict=False)
    expected.extra.weight.data.copy_(extra_value)
    for index, target_name in enumerate(targets, start=1):
        source = copy.deepcopy(_target_module(expected, target_name))
        peft_model = get_peft_model(
            source,
            LoraConfig(
                r=2,
                lora_alpha=2,
                lora_dropout=0.0,
                target_modules=["proj"],
                bias="none",
            ),
        )
        with torch.no_grad():
            peft_model.base_model.model.proj.lora_A.default.weight.fill_(0.1 * index)
            peft_model.base_model.model.proj.lora_B.default.weight.fill_(0.2 * index)
        adapter_dir = root / "adapters" / target_name
        peft_model.save_pretrained(adapter_dir, safe_serialization=True)
        config = json.loads((adapter_dir / "adapter_config.json").read_text())
        weights_path = adapter_dir / "adapter_model.safetensors"
        adapter_contracts[target_name] = {
            "path": f"adapters/{target_name}",
            "config_sha256": _sha256(adapter_dir / "adapter_config.json"),
            "weights_sha256": _sha256(weights_path),
            "rank": config["r"],
            "alpha": config["lora_alpha"],
            "target_modules": sorted(config["target_modules"]),
            "exclude_modules": config.get("exclude_modules"),
            **_safetensors_contract(weights_path),
        }
        expected_peft = PeftModel.from_pretrained(
            _target_module(expected, target_name),
            adapter_dir,
            is_trainable=False,
            autocast_adapter_dtype=False,
        )
        _assign_target(expected, target_name, expected_peft.merge_and_unload())

    manifest = {
        "format": bundle.BUNDLE_FORMAT,
        "format_version": bundle.BUNDLE_FORMAT_VERSION,
        "peft_version": "0.19.1",
        "lora_target": target,
        "base_model": {
            "model_file": "model.safetensors",
            "model_sha256": _sha256(base_dir / "model.safetensors"),
            "model_tensor_count": len(base_state),
            "config_file": "config.json",
            "config_sha256": _sha256(base_dir / "config.json"),
        },
        "policy_contract": {
            "config_name": "tiny_config",
            "norm_asset_id": "tiny_asset",
            "norm_stats_path": "assets/tiny_asset/norm_stats.json",
            "norm_stats_sha256": _sha256(norm_path),
            "model_family": "pi05",
            "action_horizon": 10,
            "effective_action_dim": 13,
            "tactile_prefix_dim_in": 7920,
            "tactile_prefix_history": 8,
        },
        "checkpoint": {"global_step": 10, "is_final": True},
        "adapters": adapter_contracts,
        "extra_trainable": {
            "path": "extra_trainable.safetensors",
            "configured_modules": ["extra"],
            "canonical_prefixes": ["extra"],
            "source_parameter_keys": ["extra.weight"],
            "keys": ["extra.weight"],
            "file_sha256": _sha256(root / "extra_trainable.safetensors"),
            **extra_contract,
        },
        "final_merged": _model_contract(expected),
        "exclusions": {
            "rl_only_prefixes": list(bundle.RL_ONLY_PREFIXES),
            "drop_exact_keys": sorted(bundle.DROP_EXACT_KEYS),
            "value_head_included": False,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    _write_checksums(root)
    return root, base_dir, expected


def _train_config() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            pi05=True,
            action_horizon=10,
            effective_action_dim=13,
            tactile_prefix_dim_in=7920,
            tactile_prefix_history=8,
        )
    )


@pytest.mark.parametrize("target", ["paligemma", "action_expert", "both"])
def test_tiny_live_peft_matches_bundle_merged_on_load(tmp_path, monkeypatch, target):
    root, base_dir, expected = _make_bundle(tmp_path, target)
    monkeypatch.setattr(bundle.pi0_pytorch, "PI0Pytorch", lambda config: _TinyModel())

    loaded = bundle.load_lora_bundle_model(
        _train_config(),
        base_dir / "model.safetensors",
        root,
        base_config_name="tiny_config",
        norm_asset_id="tiny_asset",
    )

    assert loaded.metadata["lora_target"] == target
    assert loaded.norm_assets_dir == root / "assets"
    assert not any("lora_" in key for key in loaded.model.state_dict())
    for key, expected_value in expected.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], expected_value, rtol=0, atol=0)


def _rewrite_manifest(root: Path, mutate) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    _write_checksums(root)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda manifest: manifest["base_model"].update(model_sha256="0" * 64), "base model SHA256"),
        (
            lambda manifest: manifest["adapters"]["action_expert"].update(rank=99),
            "adapter rank mismatch",
        ),
        (
            lambda manifest: manifest["extra_trainable"].update(canonical_prefixes=["wrong"]),
            "invalid extra-trainable keys",
        ),
        (
            lambda manifest: manifest["final_merged"].update(tensor_sha256="0" * 64),
            "reconstructed tensor hash",
        ),
    ],
)
def test_bundle_rejects_contract_mismatches(tmp_path, monkeypatch, mutation, match):
    root, base_dir, _ = _make_bundle(tmp_path, "action_expert")
    _rewrite_manifest(root, mutation)
    monkeypatch.setattr(bundle.pi0_pytorch, "PI0Pytorch", lambda config: _TinyModel())

    with pytest.raises(ValueError, match=match):
        bundle.load_lora_bundle_model(
            _train_config(),
            base_dir / "model.safetensors",
            root,
            base_config_name="tiny_config",
            norm_asset_id="tiny_asset",
        )


def test_bundle_rejects_wrong_config_and_norm_asset(tmp_path, monkeypatch):
    root, base_dir, _ = _make_bundle(tmp_path, "action_expert")
    monkeypatch.setattr(bundle.pi0_pytorch, "PI0Pytorch", lambda config: _TinyModel())

    with pytest.raises(ValueError, match="config name mismatch"):
        bundle.load_lora_bundle_model(
            _train_config(),
            base_dir / "model.safetensors",
            root,
            base_config_name="wrong",
            norm_asset_id="tiny_asset",
        )
    with pytest.raises(ValueError, match="normalization asset mismatch"):
        bundle.load_lora_bundle_model(
            _train_config(),
            base_dir / "model.safetensors",
            root,
            base_config_name="tiny_config",
            norm_asset_id="wrong",
        )


def test_bundle_rejects_checksum_tampering(tmp_path, monkeypatch):
    root, base_dir, _ = _make_bundle(tmp_path, "action_expert")
    with (root / "extra_trainable.safetensors").open("ab") as output:
        output.write(b"tampered")
    monkeypatch.setattr(bundle.pi0_pytorch, "PI0Pytorch", lambda config: _TinyModel())

    with pytest.raises(ValueError, match="checksum mismatch"):
        bundle.load_lora_bundle_model(
            _train_config(),
            base_dir / "model.safetensors",
            root,
            base_config_name="tiny_config",
            norm_asset_id="tiny_asset",
        )


def test_bundle_rejects_missing_adapter_weights(tmp_path, monkeypatch):
    root, base_dir, _ = _make_bundle(tmp_path, "action_expert")
    (root / "adapters" / "action_expert" / "adapter_model.safetensors").unlink()
    _write_checksums(root)
    monkeypatch.setattr(bundle.pi0_pytorch, "PI0Pytorch", lambda config: _TinyModel())

    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        bundle.load_lora_bundle_model(
            _train_config(),
            base_dir / "model.safetensors",
            root,
            base_config_name="tiny_config",
            norm_asset_id="tiny_asset",
        )


def test_extra_trainable_rejects_rl_only_key(tmp_path):
    model = _TinyModel()
    model.value_head = nn.Linear(3, 1, bias=False)
    path = tmp_path / "extra.safetensors"
    save_file({"value_head.weight": model.value_head.weight.detach()}, path)
    contract = {
        "path": "extra.safetensors",
        "keys": ["value_head.weight"],
        "canonical_prefixes": ["value_head"],
        "file_sha256": _sha256(path),
        **_safetensors_contract(path),
    }

    with pytest.raises(ValueError, match="invalid extra-trainable keys"):
        bundle._load_extra_trainable(model, tmp_path, contract)  # noqa: SLF001


def test_adapter_merge_preserves_fp32_delta_over_bf16_base(tmp_path):
    model = _TinyModel()
    model.paligemma_with_expert.gemma_expert.model.bfloat16()
    source = copy.deepcopy(model.paligemma_with_expert.gemma_expert.model)
    peft_model = get_peft_model(
        source,
        LoraConfig(r=2, lora_alpha=2, target_modules=["proj"], bias="none"),
    )
    with torch.no_grad():
        peft_model.base_model.model.proj.lora_A.default.weight.data = (
            peft_model.base_model.model.proj.lora_A.default.weight.float().fill_(0.1)
        )
        peft_model.base_model.model.proj.lora_B.default.weight.data = (
            peft_model.base_model.model.proj.lora_B.default.weight.float().fill_(0.2)
        )
    adapter_dir = tmp_path / "adapters" / "action_expert"
    peft_model.save_pretrained(adapter_dir, safe_serialization=True)
    expected = peft_model.merge_and_unload().proj.weight.detach().clone()
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    weights_path = adapter_dir / "adapter_model.safetensors"
    manifest = {
        "lora_target": "action_expert",
        "adapters": {
            "action_expert": {
                "path": "adapters/action_expert",
                "config_sha256": _sha256(adapter_dir / "adapter_config.json"),
                "weights_sha256": _sha256(weights_path),
                "rank": config["r"],
                "alpha": config["lora_alpha"],
                "target_modules": sorted(config["target_modules"]),
                "exclude_modules": config.get("exclude_modules"),
                **_safetensors_contract(weights_path),
            }
        },
    }

    loaded = bundle._apply_adapters(model, tmp_path, manifest)  # noqa: SLF001

    actual = loaded.paligemma_with_expert.gemma_expert.model.proj.weight
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
