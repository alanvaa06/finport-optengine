"""What a scenario may say about risk, and what it is refused for saying.

Two gaps in the risk half of a shock.

A replacement covariance was checked for symmetry and nothing else, so a
hand-written crisis matrix with an eigenvalue of -0.467 was accepted, and a
book that happened not to load on that eigenvector got a plausible volatility
(0.1024) out of a matrix that is not a covariance. It is refused where it is
written now, as an asymmetric one already was.

A scalar ``covariance_scale`` scales every volatility and leaves correlations
alone, so ``volatility_ratio`` is exactly ``√scale`` for every book — a
concentrated one and a diversified one alike. The shipped "Liquidity squeeze"
scenario said it modelled diversification breaking down, which a scalar cannot
do. ``correlation_shift`` moves every correlation toward one and keeps every
volatility, so a scenario can say that, and the books it hurts most are the
ones that relied on diversification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.stress import (  # noqa: E402
    Shock,
    StressError,
    load_shocks,
    shocks_from_dicts,
    stress_test,
)

NAMES = ["EQ", "HY", "UST"]
VOLS = np.array([0.16, 0.10, 0.06])
CORR = np.array([[1.0, 0.5, -0.3], [0.5, 1.0, 0.0], [-0.3, 0.0, 1.0]])


@pytest.fixture
def base() -> pd.DataFrame:
    return pd.DataFrame(np.outer(VOLS, VOLS) * CORR, index=NAMES, columns=NAMES)


# ---------------------------------------------------------------------------
# A replacement matrix has to be a covariance
# ---------------------------------------------------------------------------


def test_a_crisis_matrix_that_is_not_psd_is_refused_where_it_is_written():
    """The repro: eigenvalues include -0.467, and it used to be accepted."""
    crisis = np.array([[1.0, 0.95, -0.9], [0.95, 1.0, 0.3], [-0.9, 0.3, 1.0]])
    matrix = pd.DataFrame(np.outer(VOLS, VOLS) * crisis, index=NAMES, columns=NAMES)
    with pytest.raises(StressError, match="not positive semi-definite"):
        Shock("crisis", {"EQ": -0.3}, covariance_scale=matrix)
    with pytest.raises(StressError, match="not positive semi-definite"):
        shocks_from_dicts(
            [{"name": "crisis", "returns": {"EQ": -0.3}, "covariance_scale": matrix.to_dict()}]
        )


def test_a_psd_replacement_matrix_is_still_accepted(base):
    Shock("ok", {"EQ": -0.3}, covariance_scale=base * 4.0)


# ---------------------------------------------------------------------------
# correlation_shift: correlations move, volatilities do not
# ---------------------------------------------------------------------------


def test_a_scalar_scale_moves_every_book_by_the_same_ratio(base):
    """Why a scalar cannot model a correlation breakdown."""
    books = [
        pd.Series({"EQ": 1.0, "HY": 0.0, "UST": 0.0}),
        pd.Series({"EQ": 0.4, "HY": 0.3, "UST": 0.3}),
    ]
    shock = Shock("liq", {"EQ": -0.1}, covariance_scale=6.25)
    ratios = [stress_test(b, [shock], cov_matrix=base).worst.volatility_ratio for b in books]
    assert ratios == pytest.approx([2.5, 2.5])


def test_a_correlation_shift_keeps_every_volatility(base):
    shock = Shock("breakdown", {"EQ": -0.1}, correlation_shift=0.5)
    stressed = shock.stressed_covariance(base)
    np.testing.assert_allclose(np.diag(stressed), np.diag(base))
    implied = stressed.to_numpy() / np.outer(VOLS, VOLS)
    np.testing.assert_allclose(implied, CORR + 0.5 * (1.0 - CORR))
    assert np.linalg.eigvalsh(stressed.to_numpy()).min() >= -1e-15


def test_a_full_shift_makes_every_asset_the_same_bet(base):
    book = pd.Series({"EQ": 0.4, "HY": 0.3, "UST": 0.3})
    shock = Shock("one bet", {"EQ": -0.1}, correlation_shift=1.0)
    report = stress_test(book, [shock], cov_matrix=base)
    assert report.worst.stressed_volatility == pytest.approx(float(book @ VOLS))


def test_a_correlation_shift_hurts_the_diversified_book_and_not_the_single_asset(base):
    concentrated = pd.Series({"EQ": 1.0, "HY": 0.0, "UST": 0.0})
    diversified = pd.Series({"EQ": 0.4, "HY": 0.3, "UST": 0.3})
    shock = Shock("breakdown", {"EQ": -0.1}, correlation_shift=0.8)
    alone = stress_test(concentrated, [shock], cov_matrix=base).worst.volatility_ratio
    spread = stress_test(diversified, [shock], cov_matrix=base).worst.volatility_ratio
    assert alone == pytest.approx(1.0)
    assert spread > 1.2


def test_a_correlation_shift_composes_with_a_scalar_scale(base):
    shock = Shock("both", {"EQ": -0.1}, covariance_scale=4.0, correlation_shift=0.5)
    alone = Shock("shift", {"EQ": -0.1}, correlation_shift=0.5)
    np.testing.assert_allclose(
        shock.stressed_covariance(base).to_numpy(),
        4.0 * alone.stressed_covariance(base).to_numpy(),
    )


@pytest.mark.parametrize("value", [-0.1, 1.5, float("nan")])
def test_a_correlation_shift_outside_zero_to_one_is_refused(value):
    with pytest.raises(StressError, match="correlation_shift"):
        Shock("bad", {"EQ": -0.1}, correlation_shift=value)


def test_a_correlation_shift_on_a_replacement_matrix_is_refused(base):
    """The matrix already states its correlations; the two cannot both hold."""
    with pytest.raises(StressError, match="correlation_shift"):
        Shock("both", {"EQ": -0.1}, covariance_scale=base, correlation_shift=0.5)


def test_a_correlation_shift_round_trips_and_is_described(base):
    shock = Shock("breakdown", {"EQ": -0.1}, covariance_scale=2.0, correlation_shift=0.5)
    again = Shock.from_dict(shock.to_dict())
    assert again.correlation_shift == 0.5
    assert "correlation" in shock.describe()
    # A shock without one serializes exactly as it always did.
    assert "correlation_shift" not in Shock("plain", {"EQ": -0.1}).to_dict()


def test_the_shipped_liquidity_squeeze_says_what_it_models():
    squeeze = {s.name: s for s in load_shocks(ROOT / "config" / "shocks.yaml")}[
        "Liquidity squeeze"
    ]
    assert "cannot produce" not in squeeze.notes
    assert "correlation" in squeeze.notes.lower()
