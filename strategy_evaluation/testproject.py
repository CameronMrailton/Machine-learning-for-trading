import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import datetime as dt
from util import get_data
from ManualStrategy import ManualStrategy, plot_equity_with_entries, perf_table
from StrategyLearner import StrategyLearner
from marketsimcode import compute_portvals_from_trades
import experiment1
import experiment2


def author():
    return "crailton3"


def study_group():
    return []


def _prices(symbol: str, sd: dt.datetime, ed: dt.datetime) -> pd.DataFrame:
    dates = pd.date_range(sd, ed)
    return get_data(symbols=[symbol], dates=dates)[[symbol]].dropna()


def _stats(vals: pd.Series) -> dict:
    vals = vals.astype(float)
    daily = vals.pct_change().dropna()
    cr = vals.iloc[-1] / vals.iloc[0] - 1.0
    adr = daily.mean()
    sdr = daily.std()
    sharpe = (np.sqrt(252) * adr / sdr) if sdr != 0 else 0.0
    return {
        "Cumulative Return": float(cr),
        "Average Daily Return": float(adr),
        "Std Daily Return": float(sdr),
        "Sharpe Ratio": float(sharpe),
        "Final Value": float(vals.iloc[-1]),
    }


def _bench_trades(symbol: str, sd: dt.datetime, ed: dt.datetime) -> pd.DataFrame:
    idx = _prices(symbol, sd, ed).index
    t = pd.DataFrame(0, index=idx, columns=["Trades"])
    if len(idx) > 0:
        t.iloc[0, 0] = 1000
        t.iloc[-1, 0] = -1000
    return t


def _normalize(s: pd.Series) -> pd.Series:
    return s / s.iloc[0]


def _summary_table(name_to_vals: dict) -> pd.DataFrame:
    rows = {k: _stats(v) for k, v in name_to_vals.items()}
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


def _plot_combo(portvals_dict, out_png, title):
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 4.5))
    for label, vals in portvals_dict.items():
        plt.plot(_normalize(vals), label=label, linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    SYMBOL = "JPM"
    SV = 100000
    COMM = 9.95
    IMPACT = 0.005

    IS_SD, IS_ED = dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)
    OOS_SD, OOS_ED = dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31)

    #Manual Strategy
    ms = ManualStrategy()
    ms_is_trades = ms.testPolicy(
        symbol=SYMBOL, sd=IS_SD, ed=IS_ED, sv=SV, commission=COMM, impact=IMPACT
    )
    ms_oos_trades = ms.testPolicy(
        symbol=SYMBOL, sd=OOS_SD, ed=OOS_ED, sv=SV, commission=COMM, impact=IMPACT
    )

    #Strategy Learner
    learner = StrategyLearner(verbose=False, impact=IMPACT, commission=COMM)
    learner.add_evidence(symbol=SYMBOL, sd=IS_SD, ed=IS_ED, sv=SV)

    sl_is_trades = learner.testPolicy(symbol=SYMBOL, sd=IS_SD, ed=IS_ED, sv=SV)
    sl_oos_trades = learner.testPolicy(symbol=SYMBOL, sd=OOS_SD, ed=OOS_ED, sv=SV)

    #Benchmark
    bm_is_trades = _bench_trades(SYMBOL, IS_SD, IS_ED)
    bm_oos_trades = _bench_trades(SYMBOL, OOS_SD, OOS_ED)

    #Portvals
    is_ms = compute_portvals_from_trades(
        ms_is_trades, SYMBOL, start_val=SV, commission=COMM, impact=IMPACT
    )["portval"]
    is_sl = compute_portvals_from_trades(
        sl_is_trades, SYMBOL, start_val=SV, commission=COMM, impact=IMPACT
    )["portval"]
    is_bm = compute_portvals_from_trades(
        bm_is_trades, SYMBOL, start_val=SV, commission=COMM, impact=IMPACT
    )["portval"]

    oos_ms = compute_portvals_from_trades(
        ms_oos_trades, SYMBOL, start_val=SV, commission=COMM, impact=IMPACT
    )["portval"]
    oos_sl = compute_portvals_from_trades(
        sl_oos_trades, SYMBOL, start_val=SV, commission=COMM, impact=IMPACT
    )["portval"]
    oos_bm = compute_portvals_from_trades(
        bm_oos_trades, SYMBOL, start_val=SV, commission=COMM, impact=IMPACT
    )["portval"]


    is_tbl = _summary_table(
        {"ManualStrategy": is_ms, "StrategyLearner": is_sl, "Benchmark": is_bm}
    )
    oos_tbl = _summary_table(
        {"ManualStrategy": oos_ms, "StrategyLearner": oos_sl, "Benchmark": oos_bm}
    )
    is_tbl.to_csv("images/in_sample_summary.csv")
    oos_tbl.to_csv("images/out_sample_summary.csv")

    plot_equity_with_entries(
        symbol=SYMBOL,
        sd=IS_SD,
        ed=IS_ED,
        trades_df=ms_is_trades,
        title="ManualStrategy vs Benchmark (In-sample) — JPM",
        out_png="images/manual_in_sample.png",
        sv=SV,
        commission=COMM,
        impact=IMPACT,
    )

    plot_equity_with_entries(
        symbol=SYMBOL,
        sd=OOS_SD,
        ed=OOS_ED,
        trades_df=ms_oos_trades,
        title="ManualStrategy vs Benchmark (Out-of-sample) — JPM",
        out_png="images/manual_out_sample.png",
        sv=SV,
        commission=COMM,
        impact=IMPACT,
    )

    experiment1.run()

    #exp2: impact sensitivity

    experiment2.run()

if __name__ == "__main__":
    main()
