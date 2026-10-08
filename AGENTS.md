# Notes for coding agents

A map of this library, written for something that has to produce working
code without reading 82 modules first.

Install `finport-optengine`; import `optimization_engine`; the console script
is `optengine`. Those three names differ on purpose — the distribution name
was taken on PyPI.

```bash
pip install finport-optengine          # core: solve, backtest, analyse
pip install "finport-optengine[all]"   # + plotting, Excel, regression, data
```

Two places to look when this file does not answer the question. The **API
reference** — generated from the docstrings, so every exported name carries
its signature, parameters, units and errors — is published to GitHub Pages
from `main`, and builds locally with `pip install "finport-optengine[docs]"`
then `python scripts/build_api_docs.py`. **[`docs/ERRORS.md`](docs/ERRORS.md)**
is the refusal contract: which of the twenty-five exception types to catch,
which are recoverable, the CLI's exit codes, and — the part that catches
people — the failures that are *reported* rather than raised.

## The shortest correct program

```python
from optimization_engine import (
    EngineConfig, OptimizerSpec, prices_to_returns, run_engine, sample_dataset,
)

returns = prices_to_returns(sample_dataset())
run = run_engine(returns, EngineConfig(optimizer=OptimizerSpec(name="risk_parity")))
print(run.result.weights)
```

## Things that will bite you

Every item here was hit while writing real code against this package, not
inferred from the source. They are the cheapest paragraphs on this page.

**`sample_dataset()` returns prices, not returns.** Feeding it straight to
`run_engine` overflows and the solve dies with `Problem data contains NaN or
Inf`, which reads like a solver bug and is not one. Wrap it in
`prices_to_returns()`.

**`run_engine(returns, config)` — returns first, and positional.** Calling
`run_engine(config, returns=...)` raises `got multiple values for argument`.

**Optimizer constructors take `cov_matrix=`, not `cov=`.** `cov=` raises
`unexpected keyword argument`.

**The solver that answered is in `result.extras["solver"]`.** There is no
`result.solver`; reaching for it raises `AttributeError`. `extras` also
carries `optimizer`, `solver_status`, and `bounds_mode`.

**Backtest hashes live on `result.meta`, not on the result.** `meta.spec_hash`
and `meta.result_hash`. A `getattr(result, "spec_hash", None)` silently
returns `None` and looks like a run that produced no hash.

**`max_sharpe` with no constraints puts ~98.5% in cash on the sample panel.**
That is arithmetically right and useless as a demonstration. Use
`risk_parity` or `min_variance` for examples, or add bounds.

**`covariance_method="shrink"` is an alias for `ledoit_wolf`.** It used to
route to riskfolio-lib when installed, which meant the same config gave
different numbers on different machines. Removed in 0.4.1. Prefer
`ledoit_wolf` in new code.

**The default covariance estimator is Ledoit-Wolf, not the sample
covariance.** `covariance_matrix(returns)` shrinks unless told otherwise.

**An inaccurate solve now raises.** `solve_problem` used to settle for an
`optimal_inaccurate` answer with a logged warning. It refuses by default.
Pass `accept_inaccurate=True`, set `OptimizerSpec.accept_inaccurate`, or use
`--accept-inaccurate` when you genuinely want the degraded answer — and read
`analyze_feasibility` first, because the usual cause is a mandate that barely
has a solution.

**The Sharpe ratio is arithmetic, and so is the historical mean.** Both were
geometric. `sharpe_ratio(..., method="geometric")` and
`expected_returns_from_history(..., method="geometric_mean")` reproduce the
old numbers exactly. Sortino, Calmar and Martin keep geometric numerators —
that is deliberate and their docstrings say so.

**`extras["cvar_annualized"]` is deprecated.** It is a √T scaling, not an
annualization; the honest key is `cvar_sqrt_t_scaled`. The old keys are still
written for one release and warn at solve time. Same for
`cdar_solver_objective`, which held ζ and is now `cdar_solver_zeta`.

**`optengine check` exits 2 for an impossible mandate**, 1 for unusable data.
A finding that only says the solver could not answer is a warning, not a
fatal — a solver that crashed is not a mandate with no solution.

**`scenarios` is now `presets`.** The module was configuration persistence,
not stress testing; the old import path works for one release and warns.
Stress scenarios live in `stress.py`.

## Where things live

| You want | Import from `optimization_engine` |
| --- | --- |
| Run everything end to end | `run_engine`, `EngineConfig`, `OptimizerSpec`, `EngineRun` |
| Solve directly | `MinVarianceOptimizer`, `MaxSharpeOptimizer`, `RiskParityOptimizer`, `HRPOptimizer`, `HERCOptimizer`, `CVaROptimizer`, `CDaROptimizer`, `NCOOptimizer`, `BlackLittermanOptimizer`, `MaxDiversificationOptimizer`, `EqualWeightOptimizer`, `InverseVolatilityOptimizer`, `optimizer_factory` |
| Estimate inputs | `covariance_matrix`, `expected_returns_from_history`, `denoise_covariance`, `nearest_psd`, `james_stein_shrinkage` |
| Judge an allocation | `portfolio_diagnostics`, `risk_decomposition`, `effective_n`, `diversification_ratio`, `covariance_diagnostics` |
| Check before solving | `analyze_feasibility`, `analyze_prices`, `InfeasibleConstraintsError` |
| Backtest | `run_backtest`, `BacktestSpec`, `CostSpec`, `walk_forward_run`, `final_holdout_run`, `build_tearsheet`, `run_sweep` |
| Ask how much survives | `deflated_sharpe_ratio`, `probability_of_backtest_overfitting`, `minimum_track_record_length` |
| Load data | `load_prices`, `prices_to_returns`, `sample_dataset`, `load_prices_yahoo`, `load_fred_series` |
| Multi-provider ingestion | `optimization_engine.ingest` (subpackage), plus `IngestRequest`, `IngestResult`, `PricePanel`, `IngestError` at top level |
| Constraints | `ConstraintLayer`, `layer_from_mapping`, `currency_layer`, `effective_layers` |
| Restrict the investable set | `optimization_engine.universe` (subpackage): `Signal`, `Eligibility`, `Classification` |
| Shock a book | `Shock`, `stress_test`, `StressReport` |
| Save and reload a configuration | `Preset`, `save_presets`, `load_presets` |

`optimization_engine.ingest` is a subpackage with its own 44-name API. Its
field constants (`CLOSE`, `OPEN`, `VOLUME`) stay there on purpose — they are
too generic for the top-level namespace.

The package ships `py.typed`, so type checkers and editors resolve these
signatures without stubs.

## Shelling out instead

Four commands emit JSON on stdout with `--json`, with all human narration
moved to stderr, so stdout parses cleanly:

```bash
optengine describe risk_parity --json          # what a method needs and honours
optengine check --config c.yaml --sample --json  # is this mandate solvable
optengine optimize --config c.yaml --sample --json
optengine backtest --config c.yaml --sample --json
```

Every payload carries `schema_version`; check the major and refuse one you
do not know. A command that fails before producing a result still emits JSON
— an object with `error` and `exit_code` — so a caller never has to
distinguish "no output" from "output I could not parse".

Read `describe --json` before building a config. `supports.turnover` false
means a turnover budget is *ignored*, not rejected, and nothing will tell
you at runtime.

The payload builders are importable directly, if you are in-process rather
than shelling out: `optimization_engine.reporting.payloads`.

## Calling it as an MCP tool

For agents that call tools rather than write code:

```bash
pip install "finport-optengine[mcp]"   # needs Python 3.10+
optengine-mcp                          # speaks stdio
```

Five tools — `list_optimizers`, `describe_optimizer`, `check_mandate`,
`optimize`, `backtest` — returning the same payloads as `--json`, from the
same module. The server is a transport, not a second serialization layer.

Pass `sample=true` for the built-in panel, or `prices_path` for a file of
prices. Three of the five read filesystem paths — `config_path` and
`prices_path` — and nothing writes a file, fetches over the network, or
touches a keyed provider.

Solving blocks: a large solve is seconds of CPU and these tools are
synchronous. That is an optimizer working, not a hung server.

## What the library is opinionated about

Weights are half of a result. Every solve also reports which solver
answered, whether the constraints actually held, how well-conditioned the
covariance estimate was, and how concentrated the book is in *risk* rather
than capital — `effective_n` against `effective_n_risk` is the pair worth
reading together. Code that returns only `run.result.weights` throws away
the part that distinguishes this library.

## Working on the repository itself

```bash
pip install -e ".[all,ui,mcp,dev]"
pytest -q          # 850 tests, ~2 minutes
ruff check src app tests scripts
```

Without the `mcp` extra — or on Python 3.9, where the SDK will not install —
`tests/test_mcp_server.py` skips rather than failing, so the count is lower
and the suite still passes.

CI runs lint, the suite on Python 3.9–3.12, the Streamlit app tests, a
core-install job that asserts no optional dependency leaked into the core,
and a CLI smoke test. A separate docs workflow renders the API reference with
`--strict`, which refuses to skip a module it cannot import — so a missing
extra fails the build instead of quietly dropping a page.
`docs/RELEASING.md` covers publishing.

Docstrings are the API reference's only source, which makes them part of the
public surface rather than a courtesy. Every public function, class and method
carries one; every function with parameters documents them, with units where a
number has any (`cost_bps` per side, `execution_lag` in bars, `alpha` as a tail
probability); and every function that raises declares what and when. Adding a
public name without those is adding a hole in the published reference.
