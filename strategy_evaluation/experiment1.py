import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from pathlib import Path
import util as ut
from marketsimcode import compute_portvals_from_trades
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner


def author():
    return "crailton3"


def study_group():
    return []


SYMBOL = "JPM"
SV = 100000
COMMISSION = 9.95
IMPACT = 0.005

IS_SD, IS_ED = dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)
OOS_SD, OOS_ED = dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31)


def _normalize(s: pd.Series) -> pd.Series:
    return s / s.iloc[0]


def _prices(symbol, sd, ed):
    dates = pd.date_range(sd, ed)
    return ut.get_data([symbol], dates)[[symbol]].dropna()


def _benchmark_trades(prices):
    t = pd.DataFrame(0, index=prices.index, columns=["Trades"])
    t.iloc[0, 0] = 1000
    t.iloc[-1, 0] = -1000
    return t


def _stats(vals: pd.Series) -> dict:
    vals = vals.astype(float)
    daily = vals.pct_change().dropna()
    cr = vals.iloc[-1] / vals.iloc[0] - 1.0
    adr = daily.mean()
    sdr = daily.std()
    sharpe = np.sqrt(252) * adr / sdr if sdr != 0 else 0.0
    return {
        "Cumulative Return": float(cr),
        "Average Daily Return": float(adr),
        "Std Daily Return": float(sdr),
        "Sharpe Ratio": float(sharpe),
        "Final Value": float(vals.iloc[-1]),
    }


def _summary_table(name_to_vals: dict) -> pd.DataFrame:
    rows = {name: _stats(vals) for name, vals in name_to_vals.items()}
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df[
        [
            "Cumulative Return",
            "Average Daily Return",
            "Std Daily Return",
            "Sharpe Ratio",
            "Final Value",
        ]
    ]


def _plot_three_way(symbol, sd, ed, ms_trades, sl_trades, out_png, title):
    prices = _prices(symbol, sd, ed)

    ms_trades = ms_trades.reindex(prices.index).fillna(0).astype(int)
    sl_trades = sl_trades.reindex(prices.index).fillna(0).astype(int)
    bm_trades = _benchmark_trades(prices)

    pv_ms = compute_portvals_from_trades(
        ms_trades, symbol, start_val=SV, commission=COMMISSION, impact=IMPACT
    )["portval"]

    pv_sl = compute_portvals_from_trades(
        sl_trades, symbol, start_val=SV, commission=COMMISSION, impact=IMPACT
    )["portval"]

    pv_bm = compute_portvals_from_trades(
        bm_trades, symbol, start_val=SV, commission=COMMISSION, impact=IMPACT
    )["portval"]

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(_normalize(pv_ms), label="Manual Strategy", linewidth=2)
    plt.plot(_normalize(pv_sl), label="Strategy Learner", linewidth=2)
    plt.plot(_normalize(pv_bm), label="Benchmark", linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    return pv_ms, pv_sl, pv_bm


def run():
    out_dir = Path("images/exp1")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Manual Strategy
    ms = ManualStrategy(verbose=False)
    ms_is = ms.testPolicy(SYMBOL, IS_SD, IS_ED, SV, COMMISSION, IMPACT)
    ms_oos = ms.testPolicy(SYMBOL, OOS_SD, OOS_ED, SV, COMMISSION, IMPACT)

    # Strategy Learner
    sl = StrategyLearner(verbose=False, impact=IMPACT, commission=COMMISSION)
    sl.add_evidence(symbol=SYMBOL, sd=IS_SD, ed=IS_ED, sv=SV)

    sl_is = sl.testPolicy(symbol=SYMBOL, sd=IS_SD, ed=IS_ED, sv=SV)
    sl_oos = sl.testPolicy(symbol=SYMBOL, sd=OOS_SD, ed=OOS_ED, sv=SV)

    # Plots + collect portvals
    is_ms_pv, is_sl_pv, is_bm_pv = _plot_three_way(
        SYMBOL,
        IS_SD,
        IS_ED,
        ms_is,
        sl_is,
        out_png=out_dir / "experiment1_in_sample.png",
        title="Experiment 1 — In-sample (JPM, 2008–2009)",
    )

    oos_ms_pv, oos_sl_pv, oos_bm_pv = _plot_three_way(
        SYMBOL,
        OOS_SD,
        OOS_ED,
        ms_oos,
        sl_oos,
        out_png=out_dir / "experiment1_out_sample.png",
        title="Experiment 1 — Out-of-sample (JPM, 2010–2011)",
    )

    # Write summary tables to CSV (no printing)
    is_table = _summary_table(
        {
            "Manual Strategy": is_ms_pv,
            "Strategy Learner": is_sl_pv,
            "Benchmark": is_bm_pv,
        }
    )
    oos_table = _summary_table(
        {
            "Manual Strategy": oos_ms_pv,
            "Strategy Learner": oos_sl_pv,
            "Benchmark": oos_bm_pv,
        }
    )

    is_table.round(6).to_csv(out_dir / "experiment1_in_sample_metrics.csv")
    oos_table.round(6).to_csv(out_dir / "experiment1_out_sample_metrics.csv")


if __name__ == "__main__":
    run()
