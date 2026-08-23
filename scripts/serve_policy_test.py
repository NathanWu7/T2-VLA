import pytest
import tyro

from scripts import serve_policy


def test_checkpoint_policy_forwards_rlt_bundle(monkeypatch):
    sentinel = object()
    calls = []

    monkeypatch.setattr(serve_policy._config, "get_config", lambda name: f"config:{name}")  # noqa: SLF001

    def create_trained_policy(config, checkpoint_dir, **kwargs):
        calls.append((config, checkpoint_dir, kwargs))
        return sentinel

    monkeypatch.setattr(
        serve_policy._policy_config,  # noqa: SLF001
        "create_trained_policy",
        create_trained_policy,
    )
    args = serve_policy.Args(
        default_prompt="test prompt",
        rlt_bundle="/tmp/rlt-bundle",
        policy=serve_policy.Checkpoint(config="pi0_lora_tacfield_tabero", dir="/tmp/base"),
    )

    result = serve_policy.create_policy(args)

    assert result is sentinel
    assert calls == [
        (
            "config:pi0_lora_tacfield_tabero",
            "/tmp/base",
            {
                "default_prompt": "test prompt",
                "lora_bundle_path": None,
                "rlt_bundle_path": "/tmp/rlt-bundle",
                "dsrl_bundle_path": None,
            },
        )
    ]


def test_checkpoint_policy_forwards_dsrl_bundle(monkeypatch):
    sentinel = object()
    calls = []
    monkeypatch.setattr(serve_policy._config, "get_config", lambda name: f"config:{name}")  # noqa: SLF001

    def create_trained_policy(config, checkpoint_dir, **kwargs):
        calls.append((config, checkpoint_dir, kwargs))
        return sentinel

    monkeypatch.setattr(serve_policy._policy_config, "create_trained_policy", create_trained_policy)  # noqa: SLF001
    args = serve_policy.Args(
        dsrl_bundle="/tmp/dsrl-bundle",
        policy=serve_policy.Checkpoint(config="pi0_lora_tacfield_tabero", dir="/tmp/base"),
    )

    result = serve_policy.create_policy(args)

    assert result is sentinel
    assert calls == [
        (
            "config:pi0_lora_tacfield_tabero",
            "/tmp/base",
            {
                "default_prompt": None,
                "lora_bundle_path": None,
                "rlt_bundle_path": None,
                "dsrl_bundle_path": "/tmp/dsrl-bundle",
            },
        )
    ]


def test_cli_parses_dsrl_bundle_flag():
    args = tyro.cli(serve_policy.Args, args=["--dsrl-bundle", "/tmp/dsrl-bundle"])

    assert args.dsrl_bundle == "/tmp/dsrl-bundle"


def test_checkpoint_policy_forwards_lora_bundle(monkeypatch):
    sentinel = object()
    calls = []
    monkeypatch.setattr(serve_policy._config, "get_config", lambda name: f"config:{name}")  # noqa: SLF001

    def create_trained_policy(config, checkpoint_dir, **kwargs):
        calls.append((config, checkpoint_dir, kwargs))
        return sentinel

    monkeypatch.setattr(serve_policy._policy_config, "create_trained_policy", create_trained_policy)  # noqa: SLF001
    args = serve_policy.Args(
        lora_bundle="/tmp/lora-bundle",
        policy=serve_policy.Checkpoint(config="pi05_lora_tacfield_tabero_xarm_gripper", dir="/tmp/base"),
    )

    result = serve_policy.create_policy(args)

    assert result is sentinel
    assert calls == [
        (
            "config:pi05_lora_tacfield_tabero_xarm_gripper",
            "/tmp/base",
            {
                "default_prompt": None,
                "lora_bundle_path": "/tmp/lora-bundle",
                "rlt_bundle_path": None,
                "dsrl_bundle_path": None,
            },
        )
    ]


def test_cli_parses_lora_bundle_flag():
    args = tyro.cli(serve_policy.Args, args=["--lora-bundle", "/tmp/lora-bundle"])

    assert args.lora_bundle == "/tmp/lora-bundle"


def test_serve_policy_rejects_rlt_and_dsrl_bundles_together(monkeypatch):
    monkeypatch.setattr(
        serve_policy._config,  # noqa: SLF001
        "get_config",
        lambda name: (_ for _ in ()).throw(AssertionError("must reject before config loading")),
    )
    args = serve_policy.Args(
        rlt_bundle="/tmp/rlt",
        dsrl_bundle="/tmp/dsrl",
        policy=serve_policy.Checkpoint(config="pi0_test", dir="/tmp/base"),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        serve_policy.create_policy(args)


@pytest.mark.parametrize(
    "bundle_kwargs",
    [
        {"lora_bundle": "/tmp/lora", "rlt_bundle": "/tmp/rlt"},
        {"lora_bundle": "/tmp/lora", "dsrl_bundle": "/tmp/dsrl"},
    ],
)
def test_serve_policy_rejects_lora_with_rl_bundles(monkeypatch, bundle_kwargs):
    monkeypatch.setattr(
        serve_policy._config,  # noqa: SLF001
        "get_config",
        lambda name: (_ for _ in ()).throw(AssertionError("must reject before config loading")),
    )
    args = serve_policy.Args(
        **bundle_kwargs,
        policy=serve_policy.Checkpoint(config="pi05_test", dir="/tmp/base"),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        serve_policy.create_policy(args)


@pytest.mark.parametrize(
    "bundle_kwargs",
    [
        {"lora_bundle": "/tmp/lora"},
        {"rlt_bundle": "/tmp/rlt"},
        {"dsrl_bundle": "/tmp/dsrl"},
    ],
)
def test_bundle_requires_explicit_checkpoint_policy(bundle_kwargs):
    args = serve_policy.Args(**bundle_kwargs)

    with pytest.raises(ValueError, match="requires explicit checkpoint policy"):
        serve_policy.create_policy(args)
