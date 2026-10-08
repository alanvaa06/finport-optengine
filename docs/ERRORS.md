# Errors, and what to do about them

The engine's whole argument is that it refuses rather than fails obscurely.
That only helps if the refusal is legible, so this is the contract: every
exception the library raises on purpose, what causes it, whether it is
recoverable, and what to catch.

There are twenty-six exception classes. You almost never want to catch all of
them, because they mean three different things:

| It means | Do this | Examples |
| --- | --- | --- |
| **Your inputs are wrong** — a config, a universe, a constraint set | Fix the input. Retrying is pointless. | `SpecValidationError`, `LayerConfigurationError`, `BenchmarkError`, `ConfigurationError`, `SweepValidationError`, `StressError`, `UniverseError`, `NonPSDCovarianceError` |
| **Your mandate is impossible** — the constraints have no solution, or this method cannot meet them | Relax something, or pick another method. The exception says which. | `InfeasibleConstraintsError`, `InfeasibleBoundsError`, `SolverFailure`, `MandateViolationError`, `NoPositiveExcessReturnError` |
| **The world got in the way** — network, credentials, a vendor's bad day | Retry, or fix the environment. | `ProviderTransientError`, `ProviderCredentialsError`, `MissingDependencyError` |

Seven error types are exported from the package root — `IngestError`,
`InfeasibleConstraintsError`, `YahooFinanceError`, `FREDError`, `FXError`,
`StressError` and `UniverseError` — because they are what a caller writing a
`try` block realistically reaches for. The rest are importable from the module
that raises them.

---

## Data ingestion

Every ingest failure derives from `IngestError`, so one `except` catches the
lot:

```python
from optimization_engine import IngestError, IngestRequest, ingest

try:
    result = ingest.ingest(IngestRequest(identifiers=["SPY"], provider="tiingo"))
except IngestError as exc:
    ...
```

Catch the subclasses only when you intend to respond differently. The service
layer already does, which is why the taxonomy exists:

```
IngestError (RuntimeError)
├── ProviderNotFoundError        the provider name is not registered
├── ProviderConfigurationError   the provider cannot serve what was asked
├── ProviderCredentialsError     the API key is missing, malformed or rejected
├── ProviderTransientError       timeout, reset, 5xx, throttling
├── ProviderResponseError        it answered, the payload made no sense
├── IdentifierNotFoundError      no data for this one identifier
└── PanelValidationError         the panel failed a structural or economic check
```

| Error | Recoverable? | Scope | What the service does |
| --- | --- | --- | --- |
| `ProviderNotFoundError` | No — fix the name | The request | Raises. Check `optengine providers`. |
| `ProviderConfigurationError` | No — fix the request | The request | Raises **before any network call**: the interval, field or symbol grammar was rejected up front. |
| `ProviderCredentialsError` | Only by setting the key | The whole run | Aborts immediately. Every other identifier would fail identically, so retrying the rest is wasted time. |
| `ProviderTransientError` | Yes | One fetch | Retried automatically. It surfaces only after the retries are spent. |
| `ProviderResponseError` | Sometimes | One identifier | Recorded against the identifier; the run continues. |

An exception that is *not* an `IngestError` — a bug in a provider adapter —
does not take the run down either. It is logged with its full traceback and
reported against the identifier as its **type name only**. That is deliberate:
an unclassified exception is by definition one whose message nobody has vetted,
and these messages travel to a log, to the CLI's stderr and to the browser,
while a provider puts its API key in a request header.
| `IdentifierNotFoundError` | No — the universe is wrong | One identifier | Recorded and skipped. The run finishes with the rest of the universe. |
| `PanelValidationError` | No — the data is wrong | The panel | Raises. A panel that fails a sanity check is worse than no panel. |

**The important consequence:** a run that skips identifiers still *succeeds*.
Eleven of twelve assets arriving is not a success, and you should not have to
count columns to find out. Check the result rather than relying on the absence
of an exception:

```python
result = ingest.ingest(request)
if not result.is_complete:          # the CLI exits 1 on exactly this
    print(result.panel.coverage())  # per-identifier: what loaded, from where, over what span
```

### The legacy loaders

`data/yahoo.py`, `data/fred.py` and `data/fx.py` predate the ingest layer and
keep their own error types. They are flat `RuntimeError` subclasses with no
taxonomy — a missing dependency, a bad ticker and a network failure all arrive
as the same class, distinguishable only by message.

| Error | Raised by | Also covers |
| --- | --- | --- |
| `YahooFinanceError` | `load_prices_yahoo` | A missing `yfinance` install, an empty ticker list, an unknown ticker, an empty response |
| `FREDError` | `load_fred_series`, `load_risk_free_rate` | A malformed series id (only `A-Z`, `0-9`, `_` pass), an unreachable FRED, an empty series |
| `FXError` | `fetch_fx_to_base`, `convert_prices_to_base` | An unsupported currency (see `supported_currencies()`), a missing cross rate, a conversion that would drop every row |

Prefer the `ingest` layer for new code: it reports per-identifier provenance
and routes on the error type. These three stay for the paths that already use
them.

---

## Configuration and specs

These are all raised **before** anything is estimated or solved, which is the
point: an invalid mandate should cost you a millisecond, not a walk-forward.

| Error | Raised by | Causes |
| --- | --- | --- |
| `SpecValidationError` | `CostSpec` and `BacktestSpec` construction | A negative cost component; impact enabled with a non-positive participation; an unknown rebalance frequency; a negative `execution_lag` (it "would trade on a decision not yet taken"); `periods_per_year < 1`; a non-positive `initial_capital`; volume-priced impact on a fund too small to have a capacity problem |
| `SweepValidationError` | `SweepSpec` construction, `run_sweep`, `sweep_from_optimizers` | No parameters at all; a parameter with an empty value list; a non-positive `max_cells`; a dotted path that names nothing on the base configuration; a grid expanding past `max_cells` (200 by default, low on purpose, and itself under a hard cap) |
| `LayerConfigurationError` | `ConstraintLayer`, `layer_from_mapping`, `effective_layers` | An unknown `basis`; a layer expressing limits as a share of its parent while naming no parent; a named parent layer that does not exist; buckets that sit in more than one parent bucket, so "30% of the parent" has no single meaning; a layer entry that is neither a mapping nor a `ConstraintLayer` |
| `BenchmarkError` | `BenchmarkSpec` construction, `resolve_benchmark` | An unknown `kind` or `rebalance` rule; `single_asset` with no asset, or one outside the universe; `custom_weights` with no vector, or naming assets outside the universe; weights summing to zero under `normalize`; an empty universe |
| `ConfigurationError` | `optimizer_factory` | The method requires expected returns, a covariance matrix or a returns frame and got none; a benchmark-relative method with no benchmark **weights**; a tracking-error or active-share budget set without benchmark weights; a malformed Black-Litterman view; Black-Litterman risk-aversion calibration with no market return to calibrate against |
| `HoldoutViolationError` | `assert_within_holdout`, on every gated path | A frame handed to a gated run carries a row past the holdout boundary. A guardrail against look-ahead, not a bug in your data |
| `NonPSDCovarianceError` | `BaseOptimizer.optimize`, before any solve | A covariance whose smallest eigenvalue is below −1e-8 of its trace, so some portfolios would have negative variance. The engine's estimators repair their output with `nearest_psd`, so only a matrix handed to an optimizer directly gets here; the message names the repair. `exc.min_eigenvalue` and `exc.trace` carry the numbers |

`SpecValidationError`, `SweepValidationError`, `LayerConfigurationError`,
`BenchmarkError`, `ConfigurationError` and `NonPSDCovarianceError` subclass
`ValueError`;
`HoldoutViolationError` subclasses `RuntimeError`.

Two adjacent cases raise plain builtins rather than a named type: bounds with
`lb > ub` element-wise is a `ValueError`, and so is a holdout boundary that
leaves no held-out segment to evaluate.

### On `ConfigurationError`

It lives in `optimization_engine.optimizers`, not the package root, and it is
the one you will hit first when trying a new method:

```python
from optimization_engine.optimizers import ConfigurationError
```

`optengine describe <name>` prints what each method requires before you find
out the hard way. A related trap that is *not* an exception: a mean-variance
solve given a **flat** expected-return vector carries no cross-sectional view.
The engine says so rather than returning a `NaN` you discover three steps
later.

---

## Stress scenarios and the point-in-time universe

Two flat `ValueError` subclasses, one per module, both exported from the
package root. They share a posture: an input that cannot be honoured is refused
where it was written, and a question with no honest answer is not guessed at.

| Error | Raised by | Causes |
| --- | --- | --- |
| `StressError` | `stress_test`; `Shock` construction and `Shock.from_dict`; `shocks_from_dicts` (so `EngineConfig(stress=…)` and `EngineConfig.from_dict` too); `load_shocks`, `load_shocks_yaml` | A shock with no name, no returns, or a non-finite return; an unknown key in a scenario mapping; two scenarios sharing a name; a stress test with **no** shocks (an empty report reads exactly like a passing one); a shock naming an asset the book cannot hold, unless `unknown_assets="ignore"` says otherwise; a covariance that does not cover the book, or a `covariance_scale` that is neither a number nor a full matrix; a shocks document with an unsupported `schema_version` or neither of its two accepted shapes |
| `UniverseError` | `Signal` construction and `to_boolean_frame`; the `Eligibility` rule builders (`from_threshold`, `from_rank`, `from_rolling`, `with_hysteresis`, `hold_through`); `to_mask` and `collapse`; `point_in_time_mask`; `Classification.label`; the whole of `universe.rules` | An index that is not dates, or is numeric — reading integers as nanoseconds since the epoch would invent a calendar; a duplicated date, which leaves an as-of lookup with no single answer; an unknown comparison operator, rolling aggregation or `top_n` below 1; a mask policy that is not one of the three; **`"raise"` meeting a cell nothing evaluated**; a point-in-time label asked for with no `as_of`; and in a rules file, an unknown key at any level, an unsupported `schema_version`, an empty `rules` list, a `hysteresis` block with no `exit`, a panel that cannot be read, or a rule naming a panel nothing supplied |

Three things are worth knowing before you catch either.

**An empty stress test raises rather than passing.** `stress_test` with no
shocks has nothing to report, and a report with no findings is indistinguishable
from one that found nothing wrong. So `run_engine(..., run_stress=True)` on a
config with no `stress` block is a documented no-op — it attaches no report —
while calling `stress_test` directly with an empty list is an error.

**A shock naming an asset outside the book is the same defect as a view on
one.** It is refused, not zeroed, for the reason `build_pick_matrix` refuses an
out-of-universe Black-Litterman view: a scenario about a name you cannot hold
is a scenario about a different portfolio. `unknown_assets="ignore"` is the
explicit way out, and it records what it dropped.

**`to_mask` has no default policy, and that is not an oversight.** A universe
rule with a warm-up leaves cells nothing has evaluated, and there is no safe
reading of them: `"exclude"` silently shrinks the book at every warm-up,
`"include"` silently admits names nothing screened, and `"raise"` stops any run
whose rules warm up at all — which is every run with a rolling rule. The library
therefore makes the caller name one. `optengine backtest --universe` picks
`"exclude"`, because a non-interactive run cannot ask, and prints to stderr how
many date/asset cells that decided:

```
  Universe: 819 date/asset cell(s) across 63 bar(s) and 13 name(s) were not
  evaluable, and the 'exclude' policy — not a screen — reads them as
  ineligible: US_Equity, Intl_Equity, EM_Equity, Real_Estate, Commodities ….
```

`--universe-policy raise` turns that count into a refusal, and it is the one
`UniverseError` you are most likely to meet: it means your rules have a warm-up,
not that your data is bad.

---

## Conventions a caller has to know

These are not errors, but every one of them has been mistaken for one.

**Risk aversion is `μ'w − λ·w'Σw`, with no ½.** The convention is documented
once, on `OptimizerSpec.risk_aversion`, and repeated here because the other
common form carries a half in front of the variance term: a `risk_aversion`
tuned against that form is twice as risk-averse here as intended.
Black-Litterman passes `δ/2` for exactly this reason.

**CVaR `alpha` is the tail probability**, not the confidence level.
`alpha=0.05` means the worst 5% of outcomes.

**The Sharpe ratio is arithmetic per period, annualized.** It is the quantity
the probabilistic and deflated Sharpe ratios and the minimum track-record
length are derived on, so they now agree with it. The geometric form is
`sharpe_ratio(..., method="geometric")` and appears in `summary_stats` as
`"Sharpe Ratio (geometric)"`. Sortino, Calmar and Martin keep geometric
numerators; their docstrings say so.

**`expected_returns_from_history("mean")` is the arithmetic mean.**
Mean-variance is a single-period model, so a geometric μ against an
arithmetic Σ is a mismatch. The geometric form is `"geometric_mean"`.

**`cvar_sqrt_t_scaled` is a √T scaling, not an annualization.** It holds
under iid-Gaussian returns and not otherwise, which is why the key no longer
says "annualized".

**An answer no solver can verify is refused, not returned.** When every
solver in the fallback chain reports `optimal_inaccurate`, the solve raises
`SolverFailure` with `status="optimal_inaccurate"` rather than hand back
weights labelled optimal that are not. There are four ways to opt in, and they
cover different things:

| Opt-in | Covers |
| --- | --- |
| `accept_inaccurate: true` under the config's `optimizer` block | every solve the run's optimizers make, nested ones included |
| `--accept-inaccurate` on `optengine optimize` / `backtest` | the same, by setting that field |
| `--accept-inaccurate` on `optengine check` | the pre-flight's reachable-return LPs — `check` builds no optimizer |
| `solve_problem(problem, accept_inaccurate=True)` | that one call |

When you do opt in, the answer says so: `solver_status` on the result reads
`optimal_inaccurate`, a warning is logged, and the UI's compliance banner shows
it.

Two edges are worth knowing. The flag is a no-op for the methods that never
reach a solver — HRP, HERC and the naive weightings — but it is **not** a
no-op for NCO, whose two layers are each solved by a real optimizer. And one
call site accepts an inaccurate answer regardless of the setting: the
projection in `_bounds.project_to_constraints`, which is the dust cleanup
*after* a solve rather than the solve, where refusing would fail every
soft-bounds method whose real answer had already arrived.

The pre-flight `analyze_feasibility` also runs inside `optimize` and
`backtest`, and there it keeps the default whatever the flag says: a range no
solver could verify is reported as a `solver_error` warning rather than as a
range, and the solve goes ahead. "We could not tell you the range" has never
been a reason to refuse to optimize.

**A book that breaks the mandate is reported, not raised — unless you say
otherwise.** Every solve audits its own weights on the way out and attaches the
result as `result.audit`, an `AuditReport` whose violations carry the limit, the
actual figure and the distance between them as numbers:

```python
run = run_engine(returns, config)
if not run.result.audit.is_clean:
    print(run.result.audit.describe())      # one line per breach
    print(run.result.audit.worst.magnitude) # the biggest one, in weight terms
```

The same check runs on weights from anywhere — a spreadsheet, a backtest's
schedule, a book you did not solve for — through
`optimization_engine.optimizers.audit.audit_weights(weights, assets,
constraints, cov_matrix)`. Pass `assets` and an asset the weights never mention
is audited at zero, so a floor it misses is a breach rather than an absence.

Set `strict_mandate: true` in the config to make that a refusal instead. The
default is off because the methods that apply bounds by *projection* — HRP,
HERC, NCO, the naive weightings — can legitimately return a book their mandate
does not permit, and refusing would make them unusable. What `bounds_mode`
promises is narrower than what the audit checks: a `"hard"` method puts the box
and the bucket budgets into the convex program, and still drops a turnover
budget if it is one of the homogeneous solves (`ignored_constraints` names
them). Two limits are dropped by every projection, so they are the ones a
soft-bounds method most often breaches: a **turnover budget** (a projection is
not a trade) and a **tracking-error budget** (a risk statement, not a weights
one). An audit that ran without a covariance matrix could not check the second
at all, and comes back clean because it did not look.

## Infeasible mandates

The difference between these four matters.

**`InfeasibleConstraintsError`** is the good one. It is raised by the
pre-flight analysis, before a solver is ever called, and it carries a full
`FeasibilityReport` on the exception object:

```python
from optimization_engine.optimizers.feasibility import InfeasibleConstraintsError

try:
    run = run_engine(returns, config, raise_on_infeasible=True)
except InfeasibleConstraintsError as exc:
    print(exc.report.describe())   # names the binding constraint and the fix
```

Note `raise_on_infeasible`. It defaults to `False`, in which case the engine
reports the problem in `run.warnings` and proceeds. Pass `True` — the CLI's
`--strict` — when you would rather stop. `optengine check` runs the same
analysis without solving. It exits `2` when the mandate itself is impossible
and `1` when the data is unusable — different problems, so a script can tell
them apart. A finding that only says the solver could not answer is a warning,
not a fatal: a solver that crashed is not a mandate that has no solution.

**`InfeasibleBoundsError`** is narrower and comes from the projection step
rather than the mandate analysis. Three ways in: the minima sum above 1 or the
maxima sum below it, so the bounds and the unit budget contradict each other
outright; the projection runs out of slack on one side and cannot absorb the
residual; or the layered projection finds no allocation satisfying the weight
bounds, the budget and the bucket budgets at once — that last one wraps the
underlying `SolverFailure` and quotes it.

**`SolverFailure`** means the pre-flight passed and the solve still did not
produce a usable answer. It carries two attributes worth reading:

```python
from optimization_engine.optimizers._cvxpy_helpers import SolverFailure

try:
    ...
except SolverFailure as exc:
    exc.status     # 'infeasible', 'unbounded', 'optimal_inaccurate', ...
    exc.attempts   # every solver tried, in order
```

The message already interprets the common statuses — `infeasible` means no
allocation satisfies every constraint at once, `unbounded` means the objective
improves without limit and you are missing a bound or a budget. The engine
walks a solver fallback chain before giving up, so a `SolverFailure` means
every solver declined, not just the first.

`optimal_inaccurate` is the third, and it is the one that reads oddly at
first: a solution *was* found, and refused, because no solver would vouch for
it. See the convention above for the opt-in. Two details matter when you catch
it. The status is `optimal_inaccurate` whenever an inaccurate answer was on
the table and nothing better arrived, **even if a later solver in the chain
said something else** — a fallback that then claims `infeasible` about a
problem another solver has already found a point in is reporting its own
numerical trouble, not a property of your mandate, and sending you to check
the constraints would be sending you after the wrong thing. And it is
recoverable in a way the other two are not: `infeasible` needs the mandate
changed, whereas this one only needs you to decide whether an approximate book
is worth having. `analyze_feasibility()` says which constraint is making the
problem this hard.

The most common cause that looks like a solver bug and is not: a
tracking-error or active-share budget. A benchmark holding an asset your
bounds cap below its index weight sets a *floor* on tracking error that no
allocation can go under. Raise the limit or relax the bound.

**`NoPositiveExcessReturnError`** is max-Sharpe's own, and a `ValueError`. It
means no allocation the constraints allow earns more than the risk-free rate,
so every Sharpe ratio on offer is at or below zero and there is no tangency
portfolio. It is decided over the feasible set, not asset by asset: a
long-only box that caps the only asset above cash raises it (and says what the
best feasible excess return is, on `exc.best_excess_return`), while a
long-short book whose assets all trail cash does not, when a spread between
them beats it. NCO catches it inside a cluster and solves that cluster for
minimum variance instead, naming it in `extras["nco_min_variance_fallback"]`.

**`MandateViolationError`** is the fourth, and the only one raised *after* a
successful solve. It means the answer arrived and does not comply, and you had
asked to be told loudly:

```python
from optimization_engine.optimizers.audit import MandateViolationError

try:
    run = run_engine(returns, config)          # config.strict_mandate = True
except MandateViolationError as exc:
    exc.report                                 # the AuditReport
    exc.report.worst.describe()                # the biggest breach, named
```

It is the most recoverable of the four, and in a different way: the mandate is
not unsatisfiable, this *method* did not satisfy it. Three fixes, in order of
how often they are the right one — pick a method whose `bounds_mode` is
`"hard"` and have the limit enforced inside the convex program rather than
projected onto afterwards; loosen the limit the report names; or, having
decided a near-miss is acceptable, leave `strict_mandate` off and read
`result.audit` yourself. It subclasses `ValueError`, like
`InfeasibleConstraintsError`, and carries the report on `exc.report` for the
same reason.

---

## Missing optional dependencies

The core install is small on purpose. Reaching a feature whose extra is absent
raises `MissingDependencyError`, which subclasses `ImportError` — so
`except ImportError` around an optional feature keeps working — and whose
message is the install command that fixes it:

```
plotly.express is required for plotting, and is not installed.
Install it with: pip install 'finport-optengine[viz]'
```

| Extra | Gates |
| --- | --- |
| `viz` | Every figure in `reporting.plots` |
| `excel` | Reading and writing `.xlsx` |
| `stats` | Benchmark-relative OLS metrics (beta, regression stats) |
| `data` | The Yahoo provider, and Parquet panels |
| `mcp` | The MCP server (needs Python 3.10+) |
| `ui` | The Streamlit app |

Importing the module never raises; reaching the code path does. That is what
lets `import optimization_engine.reporting.plots` succeed on a core install and
`plot_correlation_heatmap(...)` fail with something useful.

---

## Exit codes

The CLI does not leak tracebacks. Every expected failure is caught, printed to
stderr, and turned into a code:

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | The command ran and the answer is "no" — `check` found the mandate infeasible or the data unusable, `ingest` completed but the panel is incomplete |
| `2` | The command could not run — bad config, unresolvable benchmark, data error under `--strict`, infeasible constraints, solver failure, provider error |

The distinction is worth honouring in a script: `1` means the engine worked and
is telling you something, `2` means it never got that far.

`--json` keeps those codes for anything the command *returns*, with one
addition: an exception that escapes a command is caught, its traceback printed
to stderr, and the run exits `1` rather than dying with a traceback on an empty
stdout. So in JSON mode `1` also covers "the command raised".

The payload is unconditional either way. `optimize`, `backtest`, `check` and
`describe` put exactly one parseable document on stdout — narration is
redirected to stderr for the duration — and a run that fails before producing a
result still emits one:

```json
{
  "schema_version": "...",
  "command": "optimize",
  "error": "SpecValidationError: execution_lag cannot be negative; got -1.",
  "exit_code": 1
}
```

A caller parsing stdout never has to tell "no JSON" apart from "JSON I could
not read". Note that a run which raised reports the failure *even if it had
already captured a payload*: emitting that payload under a non-zero exit would
describe a result the command did not finish producing.

---

## What does *not* raise

Some things are reported instead, and are easy to miss if you only guard with
`try`:

- **Constraint breaches after a solve.** A solver can return an answer that
  violates a constraint within tolerance, and a projecting method can return one
  that breaches a limit it was never able to impose. Both are reported — in the
  run's diagnostics, in `run.warnings`, and structured on `result.audit` — and
  never silently accepted. Raised only under `strict_mandate`, which is off by
  default.
- **Data-quality problems.** Gaps, stale feeds, suspected unadjusted splits and
  thin samples come back in `analyze_prices(...).errors` and `.warnings`. Only
  `--strict` turns an error into a refusal.
- **Degraded cost models.** A universe with no volume — an index panel, say —
  cannot price market impact from participation. The run does not fail; impact
  falls back to a fixed participation rate and every trade that did so is named
  in the run's degradation notes.
- **Skipped identifiers.** Covered above: check `result.is_complete`.

The pattern throughout is the same. A refusal is for something the engine
cannot proceed past. Everything else is reported, attached to the result that
depends on it, and left for you to decide about.

---

## See also

- [AGENTS.md](../AGENTS.md) — the API map and the `--json` CLI contract
- [README](../README.md#install) — what each extra pulls in
- The API reference — every `Raises:` section, generated from the docstrings
