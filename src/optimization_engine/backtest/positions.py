"""Position statistics: the round trips hiding inside a weight path.

An allocation backtest has no trades in the retail sense — nobody buys 100
shares and sells them later. But it does have *episodes*: an asset goes from
unheld to held, stays in the book for a while contributing to the portfolio's
return, and eventually leaves. Those episodes are round trips, and the usual
trade statistics apply to them.

They answer questions the aggregate return curve cannot. A strategy with a
respectable Sharpe and a 30% hit rate on positions is making its money in a
handful of names — which is fine, but it is a very different risk than the
same Sharpe earned across two hundred small wins. A short average holding
period paired with high turnover says the optimizer is churning.

P&L for an episode is the sum of that asset's contribution to portfolio
return while it was held, ``Σ w_t · r_t``. That is a contribution, not an
asset return: an asset that rose 40% at a 0.5% weight contributed less than a
flat one held at 30%, and it is the contribution the portfolio actually felt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from optimization_engine.backtest.results import RunResult

#: Weights below this are treated as flat rather than as a tiny position.
_HELD_EPS = 1e-8


@dataclass(frozen=True)
class PositionStats:
    """Round-trip statistics over completed position episodes.

    Every field is ``None`` when undefined, with the explanation in
    ``reasons``. A run with no closed episodes has nothing to report and says
    so, rather than reporting zeros that read like real measurements.

    Attributes:
        n_positions: Completed episodes.
        n_open_at_end: Episodes still open when the sample ran out. These are
            excluded from every statistic below: their outcome is unknown.
        win_rate: Fraction of episodes with positive contribution.
        avg_win: Mean contribution of the winners.
        avg_loss: Mean contribution of the losers (negative).
        profit_factor: Gross wins over gross losses. Below 1 the losers won.
        payoff_ratio: Average win over average loss.
        avg_holding_periods: Mean episode length.
        best / worst: The extremes, in contribution terms.
    """

    n_positions: int | None
    n_open_at_end: int
    win_rate: float | None
    avg_win: float | None
    avg_loss: float | None
    profit_factor: float | None
    payoff_ratio: float | None
    avg_holding_periods: float | None
    best: float | None
    worst: float | None
    reasons: dict[str, str] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """These statistics as a one-column frame, ready to render or export.

        Returns:
            A frame indexed by the human-readable statistic name, with a single
            ``value`` column. ``None`` entries stay ``None``: a win rate that
            could not be computed is missing, not zero.
        """
        return pd.DataFrame(
            {
                "value": {
                    "Closed positions": self.n_positions,
                    "Open at end": float(self.n_open_at_end),
                    "Win rate": self.win_rate,
                    "Average win": self.avg_win,
                    "Average loss": self.avg_loss,
                    "Profit factor": self.profit_factor,
                    "Payoff ratio": self.payoff_ratio,
                    "Average holding (periods)": self.avg_holding_periods,
                    "Best position": self.best,
                    "Worst position": self.worst,
                }
            }
        )


@dataclass(frozen=True)
class PositionEpisode:
    """One asset's uninterrupted stay in the book."""

    asset: str
    start: pd.Timestamp
    end: pd.Timestamp
    periods: int
    contribution: float
    average_weight: float
    closed: bool


def position_episodes(
    weights: pd.DataFrame, returns: pd.DataFrame
) -> list[PositionEpisode]:
    """Split a weight path into per-asset episodes, in a stable order.

    Assets are walked in sorted order and episodes in date order, so the same
    run always yields the same list — which is what lets the statistics below
    be part of a reproducible report.

    Args:
        weights: The realized weight path, dates down the index.
        returns: The asset return history the episodes are scored against.

    Returns:
        One :class:`PositionEpisode` per uninterrupted stay in the book,
        including any still open at the end of the sample.
    """
    aligned = returns.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
    episodes: list[PositionEpisode] = []
    for asset in sorted(weights.columns):
        w = weights[asset].to_numpy(dtype=float)
        r = aligned[asset].to_numpy(dtype=float)
        held = np.abs(w) > _HELD_EPS
        if not held.any():
            continue
        start_position: int | None = None
        for position in range(len(held)):
            if held[position] and start_position is None:
                start_position = position
            closing = start_position is not None and (
                not held[position] or position == len(held) - 1
            )
            if not closing:
                continue
            # A position that is still held on the final bar ends there and is
            # marked open; one that fell to zero ended on the previous bar.
            still_open = held[position]
            end_position = position if still_open else position - 1
            window = slice(start_position, end_position + 1)
            episodes.append(
                PositionEpisode(
                    asset=asset,
                    start=weights.index[start_position],
                    end=weights.index[end_position],
                    periods=end_position - start_position + 1,
                    contribution=float(np.dot(w[window], r[window])),
                    average_weight=float(np.mean(w[window])),
                    closed=not still_open,
                )
            )
            start_position = None
    return episodes


def episodes_frame(episodes: list[PositionEpisode]) -> pd.DataFrame:
    """The episodes as a tidy frame, newest-contributing first.

    Args:
        episodes: The episodes to tabulate.

    Returns:
        One row per episode with its asset, entry and exit dates, holding
        length and contribution.
    """
    if not episodes:
        return pd.DataFrame(
            columns=[
                "asset", "start", "end", "periods", "contribution",
                "average_weight", "closed",
            ]
        )
    frame = pd.DataFrame([e.__dict__ for e in episodes])
    return frame.sort_values("contribution", ascending=False).reset_index(drop=True)


def compute_position_stats(run: RunResult, returns: pd.DataFrame) -> PositionStats:
    """Round-trip statistics over the run's completed position episodes.

    Args:
        run: The finished run, whose weight path defines when each position
            opened and closed.
        returns: The asset return history the episodes are scored against.

    Returns:
        Win rate, average win and loss, profit factor, payoff ratio, average
        holding period and the best and worst positions. Any statistic that
        could not be computed — no closed positions, no losses to average — is
        ``None`` rather than zero, with the reason recorded in
        :attr:`~PositionStats.reasons`.
    """
    episodes = position_episodes(run.weights, returns)
    closed = [e for e in episodes if e.closed]
    n_open = len(episodes) - len(closed)
    reasons: dict[str, str] = {}

    if not closed:
        message = (
            "no closed positions - every position held at the end is still open"
            if episodes
            else "the book was never invested"
        )
        for name in (
            "n_positions", "win_rate", "avg_win", "avg_loss",
            "profit_factor", "payoff_ratio", "avg_holding_periods", "best", "worst",
        ):
            reasons[name] = message
        return PositionStats(
            n_positions=None,
            n_open_at_end=n_open,
            win_rate=None,
            avg_win=None,
            avg_loss=None,
            profit_factor=None,
            payoff_ratio=None,
            avg_holding_periods=None,
            best=None,
            worst=None,
            reasons=reasons,
        )

    contributions = [e.contribution for e in closed]
    wins = [c for c in contributions if c > 0.0]
    losses = [c for c in contributions if c < 0.0]
    gross_win = float(sum(wins))
    gross_loss = float(sum(losses))

    avg_win = gross_win / len(wins) if wins else None
    if avg_win is None:
        reasons["avg_win"] = "no winning positions"
    avg_loss = gross_loss / len(losses) if losses else None
    if avg_loss is None:
        reasons["avg_loss"] = "no losing positions"

    if gross_loss == 0.0:
        profit_factor: float | None = None
        reasons["profit_factor"] = "no losing positions - profit factor is undefined"
    else:
        profit_factor = gross_win / abs(gross_loss)

    if avg_win is None or avg_loss is None or avg_loss == 0.0:
        payoff_ratio: float | None = None
        reasons["payoff_ratio"] = "needs both a winning and a losing position"
    else:
        payoff_ratio = avg_win / abs(avg_loss)

    return PositionStats(
        n_positions=len(closed),
        n_open_at_end=n_open,
        win_rate=len(wins) / len(closed),
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        payoff_ratio=payoff_ratio,
        avg_holding_periods=float(np.mean([e.periods for e in closed])),
        best=max(contributions),
        worst=min(contributions),
        reasons=reasons,
    )


__all__ = [
    "PositionEpisode",
    "PositionStats",
    "compute_position_stats",
    "episodes_frame",
    "position_episodes",
]
