import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from pathlib import Path
from util import get_data
from marketsimcode import compute_portvals_from_trades
import StrategyLearner as sl

def author():
    return "crailton3"

def study_group():
    return []

# In sample data
IS_SD, IS_ED = dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)
SYMBOL = "JPM"
SV = 100000

# Commission
COMM = 0.00

# Impacts
IMPACTS = [0.000, 0.005, 0.020]


def _stats_from_portvals(portvals: pd.Series) -> dict:
    vals = portvals.astype(float)
    daily = vals.pct_change().dropna()
    cr = vals.iloc[-1] / vals.iloc[0] - 1.0
    avg_daily_return = daily.mean()
    std_daily_return = daily.std()
    sharpe = (
        np.sqrt(252) * avg_daily_return / std_daily_return
        if std_daily_return != 0
        else 0.0
    )
    return {
        "cum_return": float(cr),
        "sharpe": float(sharpe),
        "avg_daily_return": float(avg_daily_return),
        "std_daily_return": float(std_daily_return),
        "final_value": float(vals.iloc[-1]),
    }


def run():
    out_dir = Path("images/exp2")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for imp in IMPACTS:
        learner = sl.StrategyLearner(verbose=False, impact=imp, commission=COMM)
        learner.add_evidence(symbol=SYMBOL, sd=IS_SD, ed=IS_ED, sv=SV)
        trades = learner.testPolicy(symbol=SYMBOL, sd=IS_SD, ed=IS_ED, sv=SV)

        portvals = compute_portvals_from_trades(
            trades, SYMBOL, start_val=SV, commission=COMM, impact=imp
        )["portval"]

        stats = _stats_from_portvals(portvals)
        n_trades = int((trades["Trades"] != 0).sum())

        rows.append(
            {
                "impact": imp,
                "n_trades": n_trades,
                "cum_return": stats["cum_return"],
                "sharpe": stats["sharpe"],
                "avg_daily_return": stats["avg_daily_return"],
                "std_daily_return": stats["std_daily_return"],
                "final_value": stats["final_value"],
            }
        )

    dfm = pd.DataFrame(rows).sort_values("impact")
    dfm.to_csv(out_dir / "exp2_metrics.csv", index=False)

    #number of trade days vs impact
    plt.figure(figsize=(8, 5))
    plt.plot(dfm["impact"], dfm["n_trades"], marker="o", linewidth=2)
    for x, y in zip(dfm["impact"], dfm["n_trades"]):
        plt.text(x, y, f"{y}", ha="center", va="bottom", fontsize=9)
    plt.title("Number of trade days vs. impact (in-sample)")
    plt.xlabel("Impact")
    plt.ylabel("Count of trade days")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "p8_exp2_trades_vs_impact.png", dpi=150)
    plt.close()

    #cumulative return vs impact
    plt.figure(figsize=(8, 5))
    plt.plot(dfm["impact"], dfm["cum_return"], marker="o", linewidth=2)
    for x, y in zip(dfm["impact"], dfm["cum_return"]):
        plt.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=9)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.title("Experiment 2: Cumulative return vs. impact (in-sample)")
    plt.xlabel("Impact")
    plt.ylabel("Cumulative return")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "p8_exp2_cumret_vs_impact.png", dpi=150)
    plt.close()

    #sharpe ratio vs impact
    plt.figure(figsize=(8, 5))
    plt.plot(dfm["impact"], dfm["sharpe"], marker="o", linewidth=2)
    for x, y in zip(dfm["impact"], dfm["sharpe"]):
        plt.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=9)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.title("Experiment 2: Sharpe ratio vs. impact (in-sample)")
    plt.xlabel("Impact")
    plt.ylabel("Sharpe ratio")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "p8_exp2_sharpe_vs_impact.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run()
