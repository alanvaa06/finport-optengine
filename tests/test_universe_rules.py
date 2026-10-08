"""The rules file: what it accepts, what it refuses, and what it composes to.

``universe/rules.py`` is the only module in the package that reads a disk, and
it is the only place a *characteristic* panel — ADV, market capitalisation, an
index-membership flag — enters the engine at all. Two properties matter more
than any individual rule:

* a document is parsed **before** any data is read, so a typo costs a message
  rather than a Parquet load; and
* a key nothing reads is **refused**, never dropped, because a misspelt
  ``windwo`` that loads cleanly substitutes a different mandate for the one
  that was signed off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.universe import UniverseError  # noqa: E402
from optimization_engine.universe.rules import (  # noqa: E402
    RUN_PANELS,
    PanelSpec,
    RuleSpec,
    UniverseRules,
    count_unresolved,
    load_universe,
    load_universe_rules,
)

ASSETS = ["AAA", "BBB", "CCC"]


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    """Twenty sessions, three names, one of which starts printing on row 5."""
    index = pd.bdate_range("2024-01-01", periods=20)
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(20, 3)), index=index, columns=ASSETS
    )
    frame.loc[index[:5], "CCC"] = np.nan
    return frame


@pytest.fixture
def adv(returns: pd.DataFrame, tmp_path: Path) -> Path:
    """A characteristic panel on disk, beside a rules file that will name it."""
    frame = pd.DataFrame(
        {"AAA": 2.0e7, "BBB": 8.0e6, "CCC": 1.0e5},
        index=returns.index,
    )
    path = tmp_path / "adv.csv"
    frame.to_csv(path)
    return path


def _write(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "universe.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Parsing refuses what it cannot honour
# ---------------------------------------------------------------------------


def test_a_misspelt_rule_key_is_refused_rather_than_ignored(tmp_path):
    path = _write(
        tmp_path,
        {"rules": [{"kind": "rolling", "panel": "returns", "windwo": 3,
                    "agg": "mean", "op": ">", "value": 0.0}]},
    )
    with pytest.raises(UniverseError, match="windwo"):
        load_universe_rules(path)


def test_a_misspelt_top_level_key_is_refused(tmp_path):
    path = _write(
        tmp_path,
        {"rulez": [], "rules": [{"kind": "rank", "panel": "returns", "top_n": 2}]},
    )
    with pytest.raises(UniverseError, match="rulez"):
        load_universe_rules(path)


def test_an_unknown_kind_operator_or_aggregation_names_what_is_available(tmp_path):
    with pytest.raises(UniverseError, match="threshold"):
        UniverseRules.from_dict({"rules": [{"kind": "quantile", "panel": "returns"}]})
    with pytest.raises(UniverseError, match="op="):
        UniverseRules.from_dict(
            {"rules": [{"kind": "threshold", "panel": "returns", "op": "≥", "value": 0}]}
        )
    with pytest.raises(UniverseError, match="agg="):
        UniverseRules.from_dict(
            {"rules": [{"kind": "rolling", "panel": "returns", "window": 3,
                        "agg": "kurtosis", "op": ">", "value": 0}]}
        )


def test_an_empty_or_missing_rules_list_is_not_a_universe():
    for document in ({}, {"rules": []}):
        with pytest.raises(UniverseError, match="non-empty 'rules' list"):
            UniverseRules.from_dict(document)


def test_an_unsupported_schema_version_is_refused():
    with pytest.raises(UniverseError, match="schema_version"):
        UniverseRules.from_dict(
            {"schema_version": 2, "rules": [{"kind": "rank", "panel": "x", "top_n": 1}]}
        )


def test_hysteresis_needs_an_exit_rule():
    base = {"rules": [{"kind": "rank", "panel": "returns", "top_n": 1}]}
    with pytest.raises(UniverseError, match="missing required key 'exit'"):
        UniverseRules.from_dict({**base, "hysteresis": {"initial": False}})
    with pytest.raises(UniverseError, match="nothing can leave"):
        UniverseRules.from_dict({**base, "hysteresis": {"exit": []}})


def test_a_panel_cannot_shadow_one_the_run_supplies():
    with pytest.raises(UniverseError, match="supplied by the run"):
        UniverseRules.from_dict(
            {
                "panels": {"returns": {"path": "returns.csv"}},
                "rules": [{"kind": "rank", "panel": "returns", "top_n": 1}],
            }
        )
    assert "returns" in RUN_PANELS


def test_parsing_reads_no_data_at_all(tmp_path):
    """A document naming a file that does not exist still parses.

    That is the split the module is built around: validate the mandate, then
    pay for its data. A typo in the fourth rule must not cost the load of the
    first three panels.
    """
    path = _write(
        tmp_path,
        {
            "panels": {"adv": {"path": "nowhere/adv.csv"}},
            "rules": [{"kind": "threshold", "panel": "adv", "op": ">", "value": 1.0}],
        },
    )
    rules = load_universe_rules(path)
    assert rules.panel_names == ("adv",)
    with pytest.raises(UniverseError, match="does not exist"):
        rules.build(returns=None)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_rules_over_the_run_panel_alone_need_no_files(returns):
    """The price-derived first cut: no ``panels`` block, no data beside it."""
    rules = UniverseRules.from_dict(
        {
            "combine": "all",
            "rules": [
                {"kind": "rolling", "panel": "returns", "window": 5, "agg": "count",
                 "op": ">=", "value": 5, "name": "printed all week"},
            ],
        }
    )
    assert rules.panels == ()
    universe = rules.build(returns=returns)

    frame = universe.frame
    # The rolling window is strictly prior, so row 5 is the first evaluable
    # one, and CCC — silent for rows 0-4 — is out until row 10.
    assert frame.iloc[4].isna().all()
    assert bool(frame.iloc[5]["AAA"]) is True
    assert bool(frame.iloc[9]["CCC"]) is False
    assert bool(frame.iloc[10]["CCC"]) is True


def test_a_characteristic_panel_arrives_through_the_rules_file(returns, adv, tmp_path):
    """The whole point of the ``panels`` block: data that is not prices.

    ``_prepare_inputs`` loads prices and nothing else, so an ADV screen has no
    other route in. The path is relative and resolves against the rules file's
    own directory, which is what lets the document and its data move together.
    """
    path = _write(
        tmp_path,
        {
            "panels": {"adv": adv.name},
            "rules": [
                {"kind": "threshold", "panel": "adv", "op": ">=", "value": 1.0e7,
                 "name": "ADV over $10m"}
            ],
        },
    )
    universe = load_universe(path, returns=returns)

    row = universe.as_of(returns.index[-1])
    assert bool(row["AAA"]) is True
    assert bool(row["BBB"]) is False
    assert bool(row["CCC"]) is False
    assert "ADV over $10m" in universe.explain(returns.index[-1], "BBB")


def test_combine_any_is_the_disjunction(returns, adv, tmp_path):
    document = {
        "panels": {"adv": adv.name},
        "combine": "any",
        "rules": [
            {"kind": "threshold", "panel": "adv", "op": ">=", "value": 1.0e7},
            {"kind": "threshold", "panel": "adv", "op": "<=", "value": 1.0e6},
        ],
    }
    universe = load_universe(_write(tmp_path, document), returns=returns)
    row = universe.as_of(returns.index[-1])
    # AAA clears the first, CCC the second; only BBB satisfies neither.
    assert [bool(row[a]) for a in ASSETS] == [True, False, True]


def test_hysteresis_makes_membership_easier_to_keep_than_to_gain(returns, tmp_path):
    """A name that enters once stays until the *exit* rule fires, not merely
    until the entry rule stops firing."""
    rising = pd.DataFrame(0.0, index=returns.index, columns=ASSETS)
    rising.iloc[3, 0] = 10.0  # AAA clears the entry bar on one date only
    panel = tmp_path / "score.csv"
    rising.to_csv(panel)

    document = {
        "panels": {"score": panel.name},
        "rules": [{"kind": "threshold", "panel": "score", "op": ">", "value": 5.0}],
        "hysteresis": {
            "exit": [{"kind": "threshold", "panel": "score", "op": "<", "value": -1.0}],
            "initial": False,
        },
    }
    universe = load_universe(_write(tmp_path, document), returns=returns)
    frame = universe.frame

    assert bool(frame.iloc[2]["AAA"]) is False
    assert bool(frame.iloc[3]["AAA"]) is True
    # The entry rule stopped firing on row 4 and the exit rule never fires.
    assert bool(frame.iloc[-1]["AAA"]) is True
    assert bool(frame.iloc[-1]["BBB"]) is False


def test_hold_through_freezes_the_verdict_between_reconstitutions(returns, tmp_path):
    flipping = pd.DataFrame(0.0, index=returns.index, columns=ASSETS)
    flipping["AAA"] = [10.0 if i < 8 else 0.0 for i in range(len(returns))]
    panel = tmp_path / "score.csv"
    flipping.to_csv(panel)

    document = {
        "panels": {"score": panel.name},
        "rules": [{"kind": "threshold", "panel": "score", "op": ">", "value": 5.0}],
        "hold_through": {"dates": [str(returns.index[5].date()), str(returns.index[15].date())]},
    }
    universe = load_universe(_write(tmp_path, document), returns=returns)
    frame = universe.frame

    # Nothing was reviewed before the first reconstitution.
    assert frame.iloc[4].isna().all()
    # The row-5 review found AAA in, and that verdict holds past row 8 where
    # the underlying rule went false.
    assert bool(frame.iloc[9]["AAA"]) is True
    # The row-15 review reads the fresher, false verdict.
    assert bool(frame.iloc[15]["AAA"]) is False


def test_the_composition_order_is_rules_then_hysteresis_then_hold_through(returns, tmp_path):
    """Pinned because the reverse order is a different, wrong universe.

    Held-through-then-hysteresis would apply the churn brake to a series that
    has already stopped churning, so the brake would do nothing. The
    description says the order, and so does the rule tree.
    """
    document = {
        "rules": [{"kind": "rolling", "panel": "returns", "window": 3, "agg": "count",
                   "op": ">=", "value": 3}],
        "hysteresis": {"exit": [{"kind": "threshold", "panel": "returns",
                                 "op": "<", "value": -99.0}]},
        "hold_through": {"dates": [str(returns.index[6].date())]},
    }
    universe = load_universe(_write(tmp_path, document), returns=returns)
    assert universe.rule.operator == "hold_through"
    assert universe.rule.operands[0].operator == "hysteresis"


# ---------------------------------------------------------------------------
# Reporting the cost of the collapse policy
# ---------------------------------------------------------------------------


def test_count_unresolved_counts_exactly_the_cells_the_policy_decides(returns):
    """The number the CLI prints when it picks ``exclude`` on your behalf."""
    universe = UniverseRules.from_dict(
        {"rules": [{"kind": "rolling", "panel": "returns", "window": 4,
                    "agg": "count", "op": ">=", "value": 4}]}
    ).build(returns=returns)

    cells, bars, names = count_unresolved(universe, returns.index, list(returns.columns))

    # Four warm-up rows across three names, and nothing else is unevaluable.
    assert cells == 4 * len(ASSETS)
    assert bars == 4
    assert set(names) == set(ASSETS)
    assert cells == int(universe.unknown_count().sum())


def test_an_asset_the_rules_never_mention_is_also_the_policys_decision(returns):
    universe = UniverseRules.from_dict(
        {"rules": [{"kind": "threshold", "panel": "returns", "op": ">", "value": -9.9}]}
    ).build(returns=returns[["AAA"]])

    cells, _bars, names = count_unresolved(
        universe, returns.index, ["AAA", "BBB", "ZZZ"]
    )
    # Two names the universe has never heard of, on every bar.
    assert cells == 2 * len(returns.index)
    assert set(names) == {"BBB", "ZZZ"}


# ---------------------------------------------------------------------------
# Small surfaces
# ---------------------------------------------------------------------------


def test_a_panel_entry_may_be_a_bare_path_or_a_mapping():
    assert PanelSpec.from_entry("adv", "adv.csv").path == "adv.csv"
    spec = PanelSpec.from_entry("adv", {"path": "book.xlsx", "sheet": "ADV"})
    assert (spec.path, spec.sheet) == ("book.xlsx", "ADV")
    with pytest.raises(UniverseError, match="Unknown panel"):
        PanelSpec.from_entry("adv", {"path": "a.csv", "shet": "ADV"})
    with pytest.raises(UniverseError, match="missing required key 'path'"):
        PanelSpec.from_entry("adv", {"sheet": "ADV"})


def test_a_generated_label_names_the_panel_it_read():
    spec = RuleSpec.from_dict(
        {"kind": "rolling", "panel": "adv", "window": 63, "agg": "mean",
         "op": ">=", "value": 1.0e7},
        "rules[0]",
    )
    assert spec.label() == "adv rolling mean over 63 prior periods >= 1e+07"
    assert "adv" in spec.describe()


def test_describe_names_every_block_the_document_uses(returns, tmp_path):
    document = {
        "rules": [{"kind": "rank", "panel": "returns", "top_n": 2}],
        "hysteresis": {"exit": [{"kind": "threshold", "panel": "returns",
                                 "op": "<", "value": -0.5}]},
        "hold_through": {"dates": ["2024-01-08"]},
    }
    text = load_universe_rules(_write(tmp_path, document)).describe()
    assert "top 2 by returns" in text
    assert "Hysteresis" in text and "unknown" in text
    assert "reconstitution" in text


def test_a_declared_panel_no_rule_reads_is_never_loaded(returns, tmp_path):
    """A shared document edited down for one run must not pay for the rest."""
    document = {
        "panels": {"unused": {"path": "nowhere/at/all.parquet"}},
        "rules": [{"kind": "rank", "panel": "returns", "top_n": 1}],
    }
    universe = load_universe(_write(tmp_path, document), returns=returns)
    assert universe.breadth().iloc[-1] == 1


def test_a_rule_naming_no_available_panel_says_which_ones_there_are(returns):
    rules = UniverseRules.from_dict(
        {"rules": [{"kind": "rank", "panel": "mcap", "top_n": 1}]}
    )
    with pytest.raises(UniverseError, match="not available"):
        rules.build(returns=returns)


def test_an_unknown_run_panel_is_refused(returns):
    rules = UniverseRules.from_dict(
        {"rules": [{"kind": "rank", "panel": "returns", "top_n": 1}]}
    )
    with pytest.raises(UniverseError, match="Unknown run panel"):
        rules.build(returns=returns, volumes=returns)


# ---------------------------------------------------------------------------
# What the loader reads before it knows the file is a rules document
# ---------------------------------------------------------------------------


def test_a_file_without_a_rules_extension_is_not_read(tmp_path, monkeypatch):
    """The suffix used to be consulted only to choose a parser, after the read.

    Anything that was not ``.json`` went to the YAML parser, so an
    extensionless credentials file was read whole and its second line came
    back inside the parse error.
    """
    credentials = tmp_path / "credentials"
    credentials.write_text("[default]\naws_access_key_id = AKIASECRETID\n")
    reads = []
    original = Path.read_text

    def spy(self, *args, **kwargs):
        reads.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    with pytest.raises(UniverseError, match=r"\.yaml"):
        load_universe_rules(credentials)
    assert reads == []


def test_a_parse_error_gives_the_position_not_the_text(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text("rules:\n  aws_secret: AKIA_SECRET\n broken: [SECRET_LINE\n")
    with pytest.raises(UniverseError) as excinfo:
        load_universe_rules(path)
    message = str(excinfo.value)
    assert "SECRET" not in message
    assert "line" in message


def test_an_oversized_rules_file_is_not_read(tmp_path, monkeypatch):
    import optimization_engine.universe.rules as rules_module

    path = tmp_path / "rules.yaml"
    path.write_text("rules: []\n" + "# padding\n" * 50)
    monkeypatch.setattr(rules_module, "MAX_RULES_BYTES", 64)
    with pytest.raises(UniverseError, match="bytes"):
        load_universe_rules(path)
