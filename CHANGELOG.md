# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0.0, the public API may change in a minor
release. Every such change is listed here under **Changed** or **Removed**,
with what to do about it.

## [Unreleased]

## [0.7.0] — 2026-09-03

### Fixed

- **A max-diversification mandate the solver could not honour was returned
  violated with status `optimal`; it now raises.** `_solve` routed every
  failure — `SolverFailure("infeasible")` included — into the projection
  fallback, which re-solves unconstrained, projects onto the box, and then
  reported the *inner* solve's `solver_status="optimal"`. Infeasible and
  unbounded are properties of the problem, not of the solver, so they now
  re-raise; only numerical failure earns the fallback, and the fallback
  reports `solver_status="fallback_projection"` with a `dropped_constraints`
  list naming what projection cannot represent. A configuration that used to
  "succeed" with a violated tracking-error budget now raises. That is the fix.
- **The deflated Sharpe counted only the cells that solved.** `n_trials` came
  from `n_ok` while `n_failed`'s own docstring said failed cells still count
  as trials — a configuration you tried and that did not work is still a
  configuration you tried. It now counts `n_cells`, at both sites (the sweep
  module and the CLI's walk-forward sweep). Every deflated Sharpe moves toward
  zero: on fifty skill-free trials, 0.4585 → 0.4562. Separately, the trial
  distribution was computed on full-length streams while the PBO matrix
  inner-joined and dropped incomplete rows, so DSR and CSCV scored different
  samples whenever the grid swept an estimation window; both now read the
  aligned matrix.
- **Returns no longer depend on which pandas is installed.** `pct_change` was
  called with no `fill_method` at four sites. pandas 2.0 through 2.2 pad by
  default, turning a gap into a zero return followed by a compounded jump,
  while 3.0 does not fill — the same source produced two different return
  series depending on the environment. All four now pass `fill_method=None`.
  Users on pandas below 3 with gappy data will see different returns.
- **Schedule dates off the returns index now trade.** They were excluded from
  the decision set, so under `frequency="none"` a weekly schedule stamped on
  Sundays became buy-and-hold. Every off-index date now maps to the first bar
  at or after it; exact matches are unchanged, so schedules that already
  worked cannot move. `meta.notes` records what was moved, what was dropped
  past the last bar, and — the case the design did not consider — which dates
  collapsed onto a shared bar.
- **A failed first walk-forward solve holds cash instead of shortening the
  evaluation.** The schedule was only written when a previous book existed and
  the window started at the first row of weights, so a failed opening solve
  silently deleted every period before the first success, contradicting the
  module's own promise that a failed solve is a row and never a drop. The
  window now starts at the first decision. An all-cash track record is still
  refused rather than returned as a result.
- **A Black-Litterman view the universe cannot express is refused.** A view
  whose assets were all outside the universe was dropped whole; a basket view
  with one leg outside was reduced to the legs that happened to be held,
  turning "long A against short B" into "long A". Both now raise, naming every
  missing asset. The Ω floor is the same failure in numerical form: a pick row
  projecting onto zero prior variance was clamped up to 1e-12, becoming a view
  of near-infinite confidence that then dominated the posterior. It now raises.
- **A refused cache publish is retried rather than reported as a failure.**
  `PanelCache.store` finished with a bare `os.replace`, which on Windows
  raises `PermissionError` whenever another process holds the target open —
  exactly what a concurrent reader does. It is now attempted five times with
  linear backoff, and a target published by another writer of the same key
  counts as success.

- **A failed frontier anchor is reported instead of vanishing.** A GMV or
  tangency solve that failed disappeared from the chart through a bare
  `except: pass`. Both now record the reason on
  `FrontierResult.anchor_failures`, and the chart footnotes it.
- **A target return below the minimum-variance return no longer returns an
  inefficient portfolio.** It was imposed as an equality, so a target under
  the GMV return put you on the dominated lower branch with no warning. It is
  now a floor. `extras` reports whether it bound. Because a floor cannot reach
  it, the dominated branch is no longer traceable for a mean-variance sweep
  and its chart trace is gone; `show_dominated` remains as a documented no-op.
  A mean-CVaR sweep can still produce genuinely dominated points, so the
  efficiency flag stays.
- **Inverse volatility refuses a zero-variance asset instead of dropping it.**
  It gave such an asset zero weight and only raised when *every* asset had
  zero variance, so a name could vanish from the book silently.
- **Michaud resampling counts the draws that failed.** They were discarded
  with a bare `continue` and the average taken over whatever survived, which
  biases the result toward the draws where the mandate did not bind. There
  were two such drops, not one — the second is a draw that solves too few
  points to rank. `resampled_efficient_frontier` now returns a
  `ResampledFrontier` carrying `n_draws`, `n_failed` and `first_error`, and
  refuses when more draws failed than succeeded. `bootstrap_frontier` already
  counted its failures but discarded the exception; it now keeps the message.
- **EWMA covariance is denoised against its effective sample.** The nominal
  row count was passed regardless of method. At `ewma_lambda=0.94` the
  effective sample is about 17, so on 1007 rows the Marchenko-Pastur ratio was
  77 rather than 1.3 and the noise edge 1.24 rather than 3.51 — most
  eigenvalues were classed as signal on the strength of observations the
  estimator had already discounted to nothing. Below a ratio of one the cutoff
  is not estimable by this procedure at all, because the limiting law carries
  an atom at zero the fitted density does not model; that case still refuses,
  but the message now names the effective sample.
- **Bayes-Stein reports zero intensity when it shrinks nothing.** It returned
  `1.0` with an unshrunk vector — the formula's analytic limit, but not a
  description of what the estimator did.
- **The CLI's shared input path aligns the panel and says what it dropped.**
  All three commands that read prices — `optimize`, `backtest`, `check` —
  went through `dropna(how="any")` and then failed with "No usable returns
  after alignment", naming a step that was never performed. They now call
  `align_panel`, narrate the log on stderr and carry it in `--json` and in the
  MCP payloads as `alignment`. For a late listing the surviving rows are the
  same; for an interior gap the numbers genuinely differ, because differencing
  first cost two observations where aligning first costs one.
- **Backtest notes keep their values in JSON.** `meta.notes` is a mapping and
  was serialised with the helper for lists of messages, which yields only the
  keys — so a reader learned that `missing_returns` had been recorded but
  never which assets were missing.
- **NCO sub-solves go through `optimize()`.** Both layers called `_solve()`
  directly, skipping the weight cleaning, bounds recording and compliance
  diagnostics — the one method that solves twice was the one whose
  intermediate results were never checked. Weights move by 1e-07; a solve
  costs 20–24% more.
- **Importing the package no longer installs a warnings filter.** It is
  installed on the first solve instead, still process-wide.

### Changed

- **The headline Sharpe ratio is arithmetic.** Three definitions coexisted:
  geometric in `sharpe_ratio`, arithmetic in `rolling_metrics`, arithmetic
  per-period in the selection module — so the deflated Sharpe was deflating an
  arithmetic figure against a distribution of geometric ones. The per-period
  arithmetic Sharpe, `mean(excess) · ppy / (σ · √ppy)`, is now canonical,
  because it is the quantity PSR, DSR and MinTRL are derived on. The geometric
  form, `annualize_returns(excess) / annualize_volatility`, survives as
  `sharpe_ratio(..., method="geometric")` and reproduces the old number
  exactly. `summary_stats` keeps it visible for one release as
  `"Sharpe Ratio (geometric)"`.

  On the sample panel, equal-weight: **0.5950 → 0.6238**, 4.8% higher. At a 3%
  risk-free rate the same series moves 0.2441 → 0.2852, 16.8% higher — the gap
  widens as the excess mean shrinks.

  Sortino, Calmar and Martin keep geometric numerators; changing them was out
  of scope, but their docstrings and `summary_stats` now say so, because an
  arithmetic Sharpe standing silently beside a geometric Sortino is the same
  disagreement this change exists to remove.
- **The backtest result hash rounds to twelve significant figures and includes
  the weight path.** It rounded to twelve *absolute* decimals, which is stable
  at the default `initial_capital=1.0` but sits below float64's resolution
  once a caller sets realistic capital, so one ulp of BLAS noise flipped it.
  The digest also ignored holdings, so two runs with identical NAV and trades
  but different books collided. `RunMeta` gains `hash_version = 2`; hashes
  stored by earlier versions are not comparable to new ones.
- **`pandas>=2.2`.** `fill_method=None` is silent from 2.1, but the backtest
  calendar and the reporting resamplers use the `ME`/`QE`/`YE` frequency
  aliases, which arrived in 2.2 — so 2.1 was never a version this code could
  run on. The declared floor said otherwise until a CI cell pinned to it
  failed 43 tests with `Invalid frequency: ME`.
- **`expected_returns_from_history("mean")` is the arithmetic annualized
  mean.** It returned the geometric mean, so every μ-driven optimizer paired a
  geometric μ with an arithmetic Σ — a single-period model fed a multi-period
  estimate. The old formula is `"geometric_mean"` and reproduces the previous
  numbers exactly. On the sample panel, `max_sharpe` moves from 7.05% to 8.51%
  expected return and its Sharpe from 0.595 to 0.681; per-asset μ moves
  between +13bp and +244bp, except Cash, which falls 3bp because its
  compounding term exceeds half its variance. The same formula was written out
  in six places — including twice inside the covariance module, in the shrunk
  mean and in CAPM's market return — and now has one definition site, which a
  test enforces.
- **`cvar_annualized` → `cvar_sqrt_t_scaled`, `var_annualized` →
  `var_sqrt_t_scaled`.** A √T scaling holds under iid-Gaussian returns and is
  not an annualization. The old keys are written alongside the new ones for
  one release, with a `DeprecationWarning` raised at solve time — a warning on
  *read* is impossible here, because both serialization paths splat or copy
  the mapping and never call `__getitem__`.
- **`cdar_solver_objective` → `cdar_solver_zeta`**, which is what it held: the
  drawdown threshold, the same quantity CVaR calls `cvar_solver_zeta`.
  `cdar_solver_objective` now holds the objective.
- **`max_diversification` declares `bounds_mode="hard_or_projected"`.** It
  advertised `"hard"` for a method that can run projected.
- **A payload schema bump to 1.1** for the added `alignment` key.
- **`accept_inaccurate` defaults to `False`.** The solver chain settled for an
  `optimal_inaccurate` answer with a logged warning; a degraded answer returned
  by default is how a wrong book ships. It now raises, and the message names
  every way to proceed. The refusal also reports the right status: it used to
  raise with whatever the *last* solver in the chain said, so a run that found
  an unverifiable answer and then hit a solver returning `infeasible` was told
  its mandate had no solution. The setting travels as an ambient scope, because
  NCO builds its sub-optimizers internally and constructor threading cannot
  reach them.
- **`optengine check` exits 2 for an impossible mandate** and keeps 1 for
  unusable data. A finding that only says the solver could not answer is a
  warning, not a fatal.
- **Payload schema 2.0, then 2.1.** Feasibility issues are now objects with
  `code`, `severity`, `message` and `suggestion` instead of stringified
  dataclass reprs, and carry `stage_reached` and `reachable_return`; 2.1 adds
  the `audit` key, and 2.2 the `stress` key — stress reached the console and
  the Excel export but not the payload, so an agent calling the CLI could not
  see it at all.

### Added

- **Type checking in CI.** `py.typed` has shipped since 0.4 with nothing
  checking it. mypy now runs over the package; the 45 modules that do not yet
  pass are listed in an allowlist that a test holds to a ceiling and that may
  only shrink.
- **`config/shocks.yaml` and `config/universe.yaml`**, so the documented
  `--stress` and `--universe` commands run from a clean checkout. The
  universe example reads only the reserved `returns` panel, needs no data
  file beside it, and produces all three eligibility states on the sample
  panel — which is the point of shipping it.
- **A Stress tab and a Universe tab in the app.** The universe heatmap gives
  "not evaluable" its own colour rather than a shade of "ineligible", in
  three flat bands so it cannot read as half-eligible, with the state named
  in words on hover.
- **A Windows test cell and a pandas 2.2 cell** in CI. The first exercises the
  cache's `os.replace`; the second pins the declared floor, which is the only
  version in the matrix whose `pct_change` still pads and so the only one
  that can see that fix regress.
- **A point-in-time universe layer**, `optimization_engine.universe`. `Signal`
  is a date-by-asset frame with three states — true, false, and *not
  evaluable* — under Kleene logic, because a rolling liquidity rule knows
  nothing during its warm-up and answering "ineligible" there excludes names
  for a reason that has nothing to do with them. `Eligibility` builds
  membership from threshold, rank and rolling rules (windows strictly prior to
  the evaluation date), with hysteresis, reconstitution-date holding, breadth,
  turnover and an `explain(date, asset)` that names the deciding clause.
  `Classification` gives labels an effective date and refuses to answer
  without one. `run_backtest` and `walk_forward_run` both take a `universe`;
  the walk-forward masks the solve window's columns, and delisting is measured
  only on data up to the decision date. `to_mask` has no default policy — the
  caller names one.
- **`ConstraintLayer.from_classification`**, building a layer from labels as of
  a date.
- **A post-solve mandate audit.** `audit_weights` gives the compliance check a
  public surface, `OptimizationResult.audit` carries it, and
  `EngineConfig.strict_mandate` turns a reported breach into a raised
  `MandateViolationError`. Most of the checking already existed; what is new is
  that the audit reindexes onto the *declared universe* first, so an asset
  missing from the weight vector can no longer escape its own lower bound, and
  that a clean report records the tolerance it was clean at. NCO's sub-solves
  opt out — their constraint set deliberately omits the mandate's bounds,
  which are applied by projection after both layers.
- **A structural feasibility stage that needs no solver.** `analyze_feasibility`
  now answers box capacity against the budget, per-bucket capacity, parent
  coherence and instructions naming absent assets by arithmetic, before any
  solver is involved, and reports `stage_reached`. Box-and-budget mandates get
  their reachable return range from a fractional knapsack in closed form —
  exact, and zero solves where it used to cost two.
- **`--accept-inaccurate`, `--strict-mandate` and `--stress` on the CLI**, and
  an Advanced panel in the app for the first.

### Removed

- **The frontier's "Dominated (below min-variance)" trace.** A return target is
  now a floor, so a mean-variance sweep cannot reach the lower branch to draw
  it. `show_dominated` remains as a documented no-op for one release.



## [0.5.3] — 2026-09-01

Five numerical fixes from a full-package review, and the interface fixes
that followed it; the review itself is in
`docs/reviews/2026-09-01-code-review.md`. Each numerical fix changes a
number a user may already be looking at, and each is listed with what moves.

### Fixed

- **Backtest drift no longer forces full investment.** The simulation core
  renormalized drifted weights by the sum of the positions, which is the
  book's growth only when the book is fully invested. Any cash residual was
  silently converted into positions after one period, and a 60/−40 long-short
  book became 264/−164 on its second bar. Weights now drift by the book's own
  growth, `1 + w·r`. A fully invested target replays exactly as before; every
  non-fully-invested backtest — `fully_invested=False`, a partial target, a
  long-short book — moves.
- **A missing return no longer poisons the NAV path.** `0 × NaN` made every
  period from the first gap onward NaN, even for a name the book never held.
  A gap is now a flat period for that asset alone; a gap on a *held* asset is
  recorded on `meta.notes["missing_returns"]` so it reads as the data problem
  it is.
- **Comparisons score both streams over their common window.**
  `compare_in_and_out_of_sample` and `compare_performance` padded the shorter
  stream with NaN, which `annualize_returns` and `hit_rate` counted as zero-
  return periods and `var_historic` turned into NaN. On the sample panel that
  reported a walk-forward CAGR of −2.9% for a stream that made −5.8%, and
  flattered the `Degradation` column whenever the out-of-sample run was
  losing. The two functions now align on the periods both streams cover, and
  the three metrics count observations rather than rows, so an isolated gap
  anywhere is neither a return nor a loss.
- **Black-Litterman with no views now returns the market portfolio.** The
  equilibrium prior `π = δΣw` is the first-order condition of
  `μ'w − (δ/2)·w'Σw`, but the mean-variance sub-solve it fed maximizes
  `μ'w − λ·w'Σw` with no half, so the effective aversion was doubled and the
  no-view answer sat exactly halfway between the market and the minimum-
  variance portfolio. The sub-solve now receives `λ = δ/2`. Every
  `black_litterman` allocation moves, toward the market portfolio.
- **`BlackLittermanOptimizer.optimize()` is idempotent.** It used to write the
  posterior back over `cov_matrix` and `expected_returns`, so a second call
  reverse-optimized from the first call's answer and the weights drifted on
  every solve. The posterior now lives on the optimizer privately; the
  result's return and risk are still reported against it, and the inputs are
  left as given. Code that read `optimizer.cov_matrix` *after* a solve and
  expected the posterior will now see the prior.
- **Max-Sharpe and max-diversification honour a `leverage` cap.** The
  homogeneous reformulation translated bounds, layers and the benchmark
  budgets into ray space and skipped the gross-exposure cap, so a 1.2× mandate
  could come back at 2.5× gross with nothing in `ignored_constraints`. The cap
  is now a hard constraint of the ray-space solve. `fully_invested=False`
  cannot be expressed on a ray at all; both optimizers now warn and record it
  in `result.extras["ignored_constraints"]` instead of silently returning a
  fully invested book.
- **Bayes-Stein shrinkage is computed on per-period moments.** Jorion's
  intensity `(N+2) / ((N+2) + T·q)` counts observations in `T`, so the
  quadratic form `q` has to be per-observation too; `shrunk_mean` handed it
  annualized means and covariance, inflating `q` by the annualization basis
  and leaving daily data some 250 times under-shrunk. `james_stein_shrinkage`
  takes a `periods_per_year` (default 252, matching
  `expected_returns_from_history`) and divides the quadratic form by it.
  Every `expected_returns_method="shrunk_mean"` vector moves, toward the
  minimum-variance portfolio's return.
- **`check`, `optimize` and `backtest` build their inputs the same way.** The
  three commands each assembled config, panel, currency, universe and
  benchmark by hand, and had drifted: `optimize` refused a config with no
  `expected_returns` block that `check` had just called ready; `check` never
  saw `--base-currency`, `--benchmark` or the two active-risk limits;
  `backtest` seeded zero expected returns for its first solve, so a return
  target with no explicit vector was infeasible before the walk-forward
  started. One `_prepare_inputs` now serves all three, `check` accepts the
  flags `optimize` does, and a config that relies on `expected_returns_method`
  solves everywhere it pre-flights. When `--ingest-currency` has already
  converted the panel, `config.currencies` is no longer applied a second time.
- **`--json` failures carry their reason.** A command that *returned* a
  non-zero code, rather than raising, emitted `"the command exited before
  producing a result"`; the message it printed to stderr now travels in the
  payload's `error` field.
- **MCP `optimizer=` keeps the rest of the mandate.** It replaced the whole
  optimizer block, so a max-Sharpe override solved against a risk-free rate
  of zero on a config that said 4%, and dropped the return target with it.
- **MCP `optimize` reports an infeasible mandate as a `ToolError`.** The
  branch that wrapped the feasibility report was unreachable, because the
  engine's default lets the solver fail instead; the client saw a wrapped,
  message-less exception. Solver failures are wrapped the same way.
- **MCP `backtest` is shaped by its config.** The spec is now built on the
  config's `periods_per_year` (a monthly config was simulated as daily),
  trades only when the process re-solves (`frequency="none"`, matching the
  CLI and `EngineRun.walk_forward_run`), measures Sharpe against the
  config's risk-free rate, and defaults `lookback`/`rebalance_every` from the
  basis rather than from hard-coded daily counts.
- **The ingest cache refuses what it must not keep.** A panel with a failed
  or missing identifier was cached and served for the TTL as "the provider
  returned no series", never retried; a panel whose currency conversion had
  fallen back to native quotes was cached under a fingerprint claiming the
  base currency. Neither is stored now, and the run says `Not cached:` with
  the reason. The cache key also covers the provider options, so the `file`
  provider no longer serves file A's panel for file B.
- **FX alignment no longer back-fills from the future.** A leading gap in
  the rate history — prices starting before the first known rate — was
  filled with a *later* rate without limit. It is now filled for at most
  `MAX_LEADING_FX_GAP` (5) rows, the case of a request that starts on a
  holiday, and refused with an `FXError` beyond that; `fill="bfill"` asks for
  it explicitly. `fill="bfill"` also works on pandas 3, where the
  `fillna(method=)` it used to call no longer exists.
- **A config with a key the loader does not read is refused.** `max_tracking_eror:
  0.03` loaded cleanly and constrained nothing; `EngineConfig.from_dict` now
  raises `ConfigurationError` naming the unknown key, and an unknown optimizer
  key raises the same rather than a bare `TypeError`. The shipped
  `config/indices.yaml` spelled its currency block `asset_currency`, which
  nothing read; it is `currencies` now, so its non-USD indices are converted.

## [0.5.2] — 2026-09-01

Documentation only. No behaviour changed, no signature moved, and the test
suite is untouched — but the reference the library never had now exists, and
CI fails if it stops building.

### Added

- **A generated API reference.** `scripts/build_api_docs.py` renders every
  public module with pdoc and a new `docs` extra installs it. A `Docs`
  workflow builds it on every pull request and publishes it to GitHub Pages
  from `main`. The build runs `--strict`, which refuses to skip a module it
  cannot import: a missing extra now fails the job rather than quietly
  dropping `mcp_server` from the published pages. Modules are discovered by
  walking the package, so a new one is documented the day it lands.
- **`docs/ERRORS.md`** — the refusal contract, written down. All twenty-one
  exception types grouped by what they mean (your inputs are wrong; your
  mandate is impossible; the world got in the way), the full `IngestError`
  hierarchy with what the service does about each subclass, the CLI's exit
  codes and the one case where `--json` changes them, and a closing section on
  the failures that are *reported* rather than raised — degraded cost models,
  skipped identifiers, post-solve constraint breaches, data-quality findings.
  Linked from the README, `AGENTS.md` and `llms.txt`.

### Changed

- **Docstring coverage went from 74.6% to 100%** across the 677 public
  definitions: 166 that had none now do. The parameter documentation moved
  further — 86 functions documented their arguments before, 379 do now — and
  it carries units where a number has any: `cost_bps` per side, `execution_lag`
  in bars, `alpha` as a tail probability rather than a confidence level, VaR
  levels in percent rather than as fractions.
- **Every public function that raises now declares what and when.** There were
  364 `raise` statements behind 67 `Raises:` sections; the gap is closed. The
  same for returns: what a function hands back, in what units, and what it
  does with the values it could not compute.
- `AGENTS.md` states that docstrings are the API reference's only source, and
  therefore part of the public surface rather than a courtesy.


## [0.5.1] — 2026-08-31

A patch: one bug, in the part of `--json` that only a failing run reaches.
Nothing an optimizer, estimator or backtest produces is different.

### Fixed

- `--json` emitted nothing at all when a command raised. The mode promised
  that a failure still produces a parseable document, and delivered it only
  for a command that *returned* a non-zero code — a raised exception, an
  unreadable config being the cheapest way in, propagated instead: traceback
  on stderr, stdout empty. That is precisely the "no output versus output I
  could not parse" ambiguity the flag exists to remove, and it was reachable
  from all four commands. The payload now survives the exception and carries
  the exception's type and message; the traceback still goes to stderr, where
  someone debugging it looks.

### Changed

- The README announces the agent-facing surface instead of leaving it in the
  changelog: the MCP server has its own section with the install, the
  registration for a JSON-config client and for `claude mcp add`, and what
  each of the five tools answers; `--json` is documented in the CLI section
  with real `jq` output rather than a sketch of it. The development install
  gains the `mcp` extra it needed to run the MCP tests.

## [0.5.0] — 2026-08-31

A minor rather than a patch: this adds public API surface — an optional
dependency, a second console script, and a function lifted out of
`run_engine` — on top of a fix to a pre-flight check that was validating a
different mandate from the one it preceded.

### Fixed

- `check` validated a different mandate from the one `optimize` solved. On a
  config with no `expected_returns` block it derived the vector as zeros
  while `run_engine` derived it from the return history, so the pre-flight
  reported a reachable return range of exactly zero to zero and would call a
  target unreachable that the solve then reached. Both now go through
  `resolve_expected_returns`, extracted from `run_engine` so the two cannot
  drift apart again. Affects `optengine check` and the MCP `check_mandate`.

### Added

- CI runs the MCP suite. The `mcp` extra is in no other job's install — it
  cannot go in the 3.9-inclusive matrix — so those tests would have
  skipped everywhere and looked like coverage. The Streamlit job now
  covers both optional-extra surfaces and is named for it, and asserts
  the `optengine-mcp` console script lands on PATH.
- An MCP server, `optengine-mcp`, behind the `mcp` extra (Python 3.10+).
  Five tools — `list_optimizers`, `describe_optimizer`, `check_mandate`,
  `optimize`, `backtest` — returning the same payloads as `--json`, from the
  same module, so the two cannot disagree. Anticipated failures raise the
  SDK's `ToolError`, which is the only class whose message reaches the
  client; anything else is wrapped as "Error executing tool optimize" with
  the reason discarded.

- `--json` on `optimize`, `backtest`, `check` and `describe`. Human
  narration moves to stderr, so stdout is one parseable document and an
  agent or pipeline can act on a result without scraping a formatted table.
  Every payload carries `schema_version`, and a command that fails before
  producing a result still emits JSON — an object with `error` and
  `exit_code` — so a caller never has to distinguish "no output" from
  "output I could not parse".
- `optimization_engine.reporting.payloads`, the JSON contract as an
  importable module rather than formatting buried in the CLI. Keys are
  chosen for the reader instead of inherited from the attributes they came
  from, and a value absent this run is `null` rather than a missing key, so
  a consumer can test a value and never a key.
- `AGENTS.md` and `llms.txt` — an API map for coding agents, including the
  mistakes that cost real debugging time here: `sample_dataset()` returns
  prices rather than returns, the solver is in `result.extras`, the
  backtest hashes are on `result.meta`, and unconstrained `max_sharpe` puts
  98.5% in cash on the sample panel.

## [0.4.1] — 2026-08-31

A correctness and packaging release. Two of the three entries below only
reach you if you had `riskfolio-lib` installed or read the project page on
PyPI; the third is the reason the first two are worth a release at all.

### Fixed

- `covariance_method="shrink"` no longer means different mathematics on
  different machines. It routed through `riskfolio-lib` when that package
  happened to be installed and fell back to scikit-learn's Ledoit-Wolf when
  it did not — a silent fork, since `CovarianceDiagnostics` recorded no such
  thing, so the same config on the same data produced different numbers with
  nothing in the output saying which estimator had run. On the sample panel
  the two differ by 8.3% of the largest element, which is an estimator
  change, not a rounding difference. Worse, the fallback caught bare
  `Exception`, so any riskfolio error or API change swapped the estimator
  silently too.

  `shrink` is now a documented alias for `ledoit_wolf`, so configs and saved
  scenarios written against it keep loading and now reproduce anywhere. The
  `extras` optional dependency, whose only purpose was this route, is gone.

- The README renders on PyPI. Every image and repository link was a relative
  path, which resolves against the repo on GitHub and against nothing on
  PyPI — all nine figures were broken on the project page, along with the
  links to the licence and the research notes. They are absolute now.

### Changed

- The README leads with the install command and a runnable quickstart
  instead of burying `pip install` 600 lines down. The example's printed
  output is checked against what the code in that same block actually
  produces, so it cannot drift into fiction.

## [0.4.0] — 2026-08-31

The release that makes the project installable from PyPI. No optimizer,
estimator or backtest changed behaviour: every number this version produces
matches 0.3.0.

### Changed

- **The distribution is now named `finport-optengine`.** `optimization-engine`
  is taken on PyPI by an unrelated energy-forecasting package. The import
  name (`optimization_engine`) and the console script (`optengine`) are
  unchanged, so only the install command moves:
  `pip install finport-optengine`.
- **The core install is smaller by roughly 190MB.** `plotly`, `openpyxl`,
  `xlsxwriter` and `statsmodels` are no longer mandatory dependencies; they
  now sit behind the `viz`, `excel` and `stats` extras. Anything the core
  can already do — every optimizer, the covariance estimators, the backtest,
  the analytics — still works on a bare `pip install finport-optengine`.

  Reaching a feature whose extra is absent now raises
  `optimization_engine._optional.MissingDependencyError` (a subclass of
  `ImportError`) naming the install command, instead of a
  `ModuleNotFoundError` for an import the caller never wrote.

  To keep the previous all-inclusive install, use
  `pip install "finport-optengine[all]"`.

  `scikit-learn` deliberately stays in the core: it backs the *default*
  covariance estimator, so demoting it would either break the first call in
  the README on a fresh install or silently change the default to the
  sample covariance.

### Added

- A release workflow (`.github/workflows/release.yml`) publishing through
  Trusted Publishing (OIDC), so no API token is stored in the repository.
  A manual run publishes a fresh `.devN` to TestPyPI and then installs it
  back *from* TestPyPI to prove the artifact is reachable; pushing a `v*`
  tag publishes to PyPI behind a required-reviewer gate. The build refuses
  to proceed if `pyproject.toml` and `__init__.py` disagree on the version,
  or if a tag does not match the version it claims.
- `docs/RELEASING.md` — the one-time Trusted Publishing setup each index
  needs, the release procedure, and what to do about a bad release.

- `optimization_engine.ingest` is reachable as an attribute of the top-level
  package, and its four most-used names — `IngestRequest`, `IngestResult`,
  `IngestError`, `PricePanel` — are re-exported at the top level. The
  subpackage's full 44-name API is unchanged and stays where it was; the
  generic field constants (`CLOSE`, `OPEN`, `VOLUME`, …) are deliberately
  *not* raised to the top level.
- `py.typed`, so type checkers actually consume the project's annotations.
  The marker had been declared in `pyproject.toml` since the package was
  laid out, but the file itself was never added — meaning no downstream
  user has ever received the types.
- `[project.urls]`, so the PyPI page links to the repository, the issue
  tracker and this file.
- An `all` extra, covering what the CLI and the README's worked examples
  assume.
- This changelog.

### Removed

- `matplotlib` and `seaborn` as dependencies. Nothing in the project has
  ever imported either one — the plotting is entirely Plotly. Removing them
  changes no behaviour.
- `requirements.txt`. It listed `streamlit`, `yfinance`, `riskfolio-lib` and
  `ipywidgets` as mandatory, contradicting `pyproject.toml`, which has them
  as extras. `pyproject.toml` is now the single source of truth; install the
  development set with `pip install -e ".[all,ui,extras,dev]"`.

### Fixed

- The release workflow could never publish. `pypa/gh-action-pypi-publish`
  was referenced by commit SHA — the usual supply-chain advice, and wrong
  for this action: it is a Docker action, and the runner pulls
  `ghcr.io/pypa/gh-action-pypi-publish` tagged with whatever ref the `uses:`
  line carries. PyPA publishes that image only under release tags, so the
  SHA resolved to no manifest and the step died with `manifest unknown`
  before reaching the index. Now referenced as `@v1.14.2`, a tag confirmed
  to exist in the registry, with the reasoning recorded in
  `docs/RELEASING.md` so nobody "hardens" it back.

- Excluded `scs` 3.3.0. Its wheel ships an incomplete Intel oneMKL bundle,
  so importing it and solving aborts the interpreter — "Cannot load
  libmkl_avx512.so.3 or libmkl_def.so.3" — rather than raising. SCS sits in
  `SOLVER_FALLBACK`, so any solve that got past CLARABEL and ECOS would take
  the process down with it, with no traceback and nothing for the fallback
  chain to catch. Reproduced on GitHub's runners and in a local container;
  3.2.11 is unaffected. `scs` is now named directly in the dependencies,
  since the engine reaches for it by name rather than leaving the choice to
  cvxpy. Remove the exclusion once a fixed release ships.

## [0.3.0] and earlier

Released before this changelog was kept. The repository's commit history is
the record; the headline work was the data-ingestion spine (one panel, many
providers, with per-identifier provenance), the stateless backtest core with
its cost model and trial counting, the walk-forward and final-holdout audit
path, and the constraint-layer editor in the Streamlit app.

[Unreleased]: https://github.com/alanvaa06/finport-optengine/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/alanvaa06/finport-optengine/compare/v0.5.3...v0.7.0
[0.5.3]: https://github.com/alanvaa06/finport-optengine/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/alanvaa06/finport-optengine/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/alanvaa06/finport-optengine/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/alanvaa06/finport-optengine/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/alanvaa06/finport-optengine/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/alanvaa06/finport-optengine/releases/tag/v0.4.0
