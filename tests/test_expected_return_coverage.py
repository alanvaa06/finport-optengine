"""An expected-return vector that does not cover the universe is never silently zeroed.

``resolve_expected_returns`` reindexed the vector onto the panel and filled
the gaps with 0.0 *before* any optimizer saw it, so the warning
``BaseOptimizer._mu_vector`` raises for a missing expected return could never
fire on the engine path. A misspelt key — ``EM_Equityy`` for ``EM_Equity`` —
therefore gave ``EM_Equity`` an expected return of exactly zero, with
``run.warnings`` empty and nothing in ``extras``. A zero expected return is an
active, bearish view, and the book was built on it.

The rule now:

* a panel asset with no expected return is still filled with 0.0, but the run
  says so — a ``UserWarning``, a line in ``run.warnings``, and
  ``extras["missing_expected_returns"]``;
* a vector that both misses panel assets *and* names assets the panel does not
  hold is refused, because that is what a misspelling looks like;
* a vector that covers the panel and names extra assets is used, and the extras
  are reported as ignored — the walk-forward hands a screened window a subset
  of the columns the config was written for, and that is not a mistake.

The CLI restricts the panel to the config's ``expected_returns`` before any of
this runs, and it used to drop both kinds of mismatch without a word.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.cli import main  # noqa: E402
from optimization_engine.config import EngineConfig, OptimizerSpec  # noqa: E402
from optimization_engine.data.covariance import covariance_matrix  # noqa: E402
from optimization_engine.data.loader import prices_to_returns, sample_dataset  # noqa: E402
from optimization_engine.engine import resolve_expected_returns, run_engine  # noqa: E402
from optimization_engine.optimizers import ConfigurationError  # noqa: E402


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset())


def _mean_variance(mu: dict[str, float], assets: list[str]) -> EngineConfig:
    return EngineConfig(
        optimizer=OptimizerSpec(name="mean_variance", risk_aversion=2.0),
        expected_returns=mu,
        bounds={a: [0.0, 0.3] for a in assets},
    )


def test_a_misspelt_key_is_refused_rather_than_zeroing_the_real_asset(returns):
    """The repro: EM_Equityy given, EM_Equity therefore at exactly 0.0."""
    assets = list(returns.columns)
    mu = {a: 0.06 for a in assets if a != "EM_Equity"}
    mu["EM_Equityy"] = 0.12
    with pytest.raises(ConfigurationError) as caught:
        run_engine(returns, _mean_variance(mu, assets))
    message = str(caught.value)
    assert "EM_Equityy" in message and "EM_Equity" in message.replace("EM_Equityy", "")


def test_a_forgotten_asset_is_reported_before_it_is_filled(returns):
    assets = list(returns.columns)
    mu = {a: 0.06 for a in assets if a != "EM_Equity"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = run_engine(returns, _mean_variance(mu, assets))

    assert run.expected_returns["EM_Equity"] == 0.0
    assert run.result.extras["missing_expected_returns"] == ["EM_Equity"]
    assert any("EM_Equity" in w and "expected return" in w for w in run.warnings)
    assert any("no expected return" in str(w.message) for w in caught)


def test_names_outside_the_panel_are_ignored_and_said_to_be(returns):
    """A superset is what a walk-forward window gets once a screen narrows it."""
    assets = list(returns.columns)
    mu = {a: 0.06 for a in assets}
    mu["OLD_FUND"] = 0.09
    run = run_engine(returns, _mean_variance(mu, assets))

    assert list(run.expected_returns.index) == assets
    assert run.result.extras["ignored_expected_returns"] == ["OLD_FUND"]
    assert any("OLD_FUND" in w for w in run.warnings)
    assert "missing_expected_returns" not in run.result.extras


def test_a_complete_vector_adds_nothing_to_the_run(returns):
    assets = list(returns.columns)
    run = run_engine(returns, _mean_variance({a: 0.06 for a in assets}, assets))
    assert "missing_expected_returns" not in run.result.extras
    assert "ignored_expected_returns" not in run.result.extras
    assert not [w for w in run.warnings if "expected return" in w]


def test_resolve_expected_returns_applies_the_same_rule(returns):
    """The CLI's check command and the MCP tool call it directly."""
    cov = covariance_matrix(returns)
    assets = list(returns.columns)
    partial = EngineConfig(expected_returns={a: 0.05 for a in assets[1:]})
    with pytest.warns(UserWarning, match="no expected return"):
        mu = resolve_expected_returns(partial, returns, cov)
    assert mu[assets[0]] == 0.0

    typo = {a: 0.05 for a in assets[1:]}
    typo[assets[0] + "x"] = 0.05
    with pytest.raises(ConfigurationError, match=assets[0]):
        resolve_expected_returns(EngineConfig(), returns, cov, pd.Series(typo))


def test_the_cli_says_which_assets_the_config_cut_from_the_panel(tmp_path, capsys):
    prices = sample_dataset()
    csv = tmp_path / "panel.csv"
    prices.to_csv(csv)
    assets = list(prices.columns)
    mu = {a: 0.06 for a in assets if a != "EM_Equity"}
    mu["EM_Equityy"] = 0.12
    config = tmp_path / "typo.yaml"
    config.write_text(
        yaml.safe_dump({"optimizer": "min_variance", "expected_returns": mu})
    )

    code = main(["check", "--config", str(config), "--prices", str(csv), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 0
    said = " ".join(payload["alignment"])
    assert "EM_Equityy" in said, "the config asked for a name the panel lacks"
    assert "EM_Equity" in said.replace("EM_Equityy", ""), "a priced name was cut"
    assert "EM_Equityy" in captured.err
