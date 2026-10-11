"""Tests for how `model-check` hands a provider/model to the probe.

The probe is what stands between "the model is configured" and a live run, and
it reads its identity from the environment rather than from its argv. So the
only thing that can go wrong here is silent: the probe validates whatever the
backend is *globally* configured with, and the caller is told the result as if
it were about the model it named.

That is not hypothetical. It is what happened: `model-check pi --provider
cc-switch-deep-seek --model deepseek-v4-pro` reported AUTHENTICATION_FAILED in
under a second, because the flags were accepted, used to build the Config, and
then dropped -- so the probe fell through to Pi's global default (a Claude
model at api.anthropic.com, where this user has no key) and got a 401. The
provider itself was working the whole time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import bootstrapping
    sys.path.insert(0, str(SRC))

from dream_rsi.cli import _probe_environment  # noqa: E402
from dream_rsi.config import load_config  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_identity_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PI_PROVIDER", "PI_MODEL", "DSH_PROVIDER", "DSH_MODEL",
                 "DREAM_RSI_PROVIDER", "DREAM_RSI_MODEL"):
        monkeypatch.delenv(name, raising=False)


def _config(**kwargs: object):
    return load_config(project_root=REPO, backend="pi", **kwargs)  # type: ignore[arg-type]


def test_explicit_provider_and_model_reach_the_probe() -> None:
    config = _config(provider="cc-switch-deep-seek", model="deepseek-v4-pro")
    env = _probe_environment(config, "pi")
    assert env["PI_PROVIDER"] == "cc-switch-deep-seek"
    assert env["PI_MODEL"] == "deepseek-v4-pro"


def test_the_generic_variables_are_honoured(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """`DREAM_RSI_*` outranks `PI_*`, and the probe must be told the winner.

    The probe only knows the backend-specific names. If the generic ones were
    not resolved down to them first, naming the model through `DREAM_RSI_*`
    would set the run's identity but not the probe's.
    """
    monkeypatch.setenv("DREAM_RSI_PROVIDER", "generic-provider")
    monkeypatch.setenv("DREAM_RSI_MODEL", "generic-model")
    monkeypatch.setenv("PI_PROVIDER", "stale-provider")
    env = _probe_environment(_config(), "pi")
    assert env["PI_PROVIDER"] == "generic-provider"
    assert env["PI_MODEL"] == "generic-model"


def test_an_explicit_flag_outranks_the_environment(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PI_PROVIDER", "env-provider")
    monkeypatch.setenv("PI_MODEL", "env-model")
    config = _config(provider="flag-provider", model="flag-model")
    env = _probe_environment(config, "pi")
    assert env["PI_PROVIDER"] == "flag-provider"
    assert env["PI_MODEL"] == "flag-model"


def test_nothing_is_invented_when_no_model_is_named() -> None:
    """An unnamed backend must stay unnamed at the probe too.

    The probe is the place a missing identity should surface. Filling in a
    default here would hide exactly the condition the probe exists to catch.
    """
    env = _probe_environment(_config(), "pi")
    assert "PI_PROVIDER" not in env
    assert "PI_MODEL" not in env


def test_dsh_gets_its_own_variable_names() -> None:
    config = load_config(project_root=REPO, backend="dsh",
                         provider="deepseek-official", model="deepseek-v4-flash")
    env = _probe_environment(config, "dsh")
    assert env["DSH_PROVIDER"] == "deepseek-official"
    assert env["DSH_MODEL"] == "deepseek-v4-flash"
    assert "PI_PROVIDER" not in env


def test_the_rest_of_the_environment_is_passed_through() -> None:
    """The probe subprocess still needs PATH and friends to find `pi`."""
    env = _probe_environment(_config(provider="p", model="m"), "pi")
    assert "PATH" in env
    for key, value in env.items():
        assert isinstance(value, str), key
