
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime as dt

from util import get_data
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from marketsim import compute_portvals_from_trades

# ---------------- required APIs ----------------
def author():
    return "crailton3"

def study_group():
    return []

# ==== Summary + Charts Utilities (put this in testproject.py) ====
def _stats_from_portvals(vals: pd.Series) -> dict:
    """Compute CR, ADR, SDR, Sharpe (252) and Final Value from a portvals Series."""
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

def _trade_count(trades_df: pd.DataFrame) -> int:
    """Count non-zero trade rows. Expects a 'Trades' column."""
    if trades_df is None or "Trades" not in trades_df.columns or trades_df.empty:
        return 0
    return int((trades_df["Trades"] != 0).sum())

def _final_holdings(trades_df: pd.DataFrame) -> int:
    """Final share holdings = cumulative sum of trades on the last day."""
    if trades_df is None or "Trades" not in trades_df.columns or trades_df.empty:
        return 0
    return int(trades_df["Trades"].cumsum().iloc[-1])

def summarize_strategies(
    manual_vals: pd.Series,
    learner_vals: pd.Series,
    tos_vals: pd.Series,
    manual_trades: pd.DataFrame,
    learner_trades: pd.DataFrame,
    tos_trades: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the final comparison table:
      ManualStrategy, StrategyLearner, Benchmark
    Columns: CumRet, ADR, SDR, Sharpe, Final Value, Total Trades, Final Holdings
    """
    rows = {
        "ManualStrategy": {
            **_stats_from_portvals(manual_vals),
            "Total Trades": _trade_count(manual_trades),
            "Final Holdings": _final_holdings(manual_trades),
        },
        "StrategyLearner": {
            **_stats_from_portvals(learner_vals),
            "Total Trades": _trade_count(learner_trades),
            "Final Holdings": _final_holdings(learner_trades),
        },
        "Benchmark": {
            **_stats_from_portvals(tos_vals),
            "Total Trades": _trade_count(tos_trades),
            "Final Holdings": _final_holdings(tos_trades),
        },
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df[
        [
            "Cumulative Return",
            "Average Daily Return",
            "Std Daily Return",
            "Sharpe Ratio",
            "Final Value",
            "Total Trades",
            "Final Holdings",
        ]
    ]

def make_summary_charts(
    df: pd.DataFrame,
    out_dir: str = "images",
    fname_table: str = "p8_final_summary_table.png",
    fname_bars1: str = "p8_final_perf_bars.png",
    fname_bars2: str = "p8_final_trades_bars.png",
    fname_bars3: str = "p8_final_holdings_bars.png",
):
    """
    Save 4 visuals:
      1) A figure rendering the summary table (PNG)
      2) Bar chart: Cumulative Return, Sharpe, Final Value (twin axis)
      3) Bar chart: Total Trades
      4) Bar chart: Final Holdings
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ---------- 1) TABLE AS FIGURE ----------
    cols = [
        "Cumulative Return",
        "Average Daily Return",
        "Std Daily Return",
        "Sharpe Ratio",
        "Final Value",
        "Total Trades",
        "Final Holdings",
    ]
    df_disp = df.copy()[cols]
    df_fmt = df_disp.copy()
    for c in ["Cumulative Return", "Average Daily Return", "Std Daily Return", "Sharpe Ratio"]:
        df_fmt[c] = df_fmt[c].map(lambda x: f"{x:.6f}")
    df_fmt["Final Value"] = df_fmt["Final Value"].map(lambda x: f"{x:,.2f}")
    df_fmt["Total Trades"] = df_fmt["Total Trades"].astype(int).astype(str)
    df_fmt["Final Holdings"] = df_fmt["Final Holdings"].astype(int).astype(str)

    fig, ax = plt.subplots(figsize=(12, 2 + 0.55 * len(df_fmt)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df_fmt.values,
        rowLabels=df_fmt.index.tolist(),
        colLabels=df_fmt.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.35)
    ax.set_title("Final Strategy Comparison — Stats (6-dec precision)", pad=16, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{fname_table}", dpi=200, bbox_inches="tight")
    plt.close()

    # ---------- 2) PERFORMANCE BARS ----------
    labels = df.index.tolist()
    x = np.arange(len(labels))
    width = 0.27

    cumret = df["Cumulative Return"].values
    sharpe = df["Sharpe Ratio"].values
    finalv = df["Final Value"].values

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(x - width, cumret, width, label="Cumulative Return")
    ax1.bar(x, sharpe, width, label="Sharpe Ratio")

    ax2 = ax1.twinx()
    ax2.bar(x + width, finalv, width, label="Final Value ($)", zorder=0)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Strategy")
    ax1.set_ylabel("CumRet / Sharpe")
    ax2.set_ylabel("Final Value ($)")
    ax1.set_title("Performance Comparison — CumRet, Sharpe, Final Value")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{fname_bars1}", dpi=200, bbox_inches="tight")
    plt.close()

    # ---------- 3) TOTAL TRADES BARS ----------
    trades = df["Total Trades"].values.astype(int)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, trades)
    ax.set_title("Total Trades by Strategy")
    ax.set_ylabel("Trade Count")
    ax.set_xlabel("Strategy")
    for i, v in enumerate(trades):
        ax.text(i, v + max(1, trades.max()) * 0.03, str(v), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{fname_bars2}", dpi=200, bbox_inches="tight")
    plt.close()
    # normalized
    plt.figure()
    plt.plot(portvals_manual / portvals_manual.iloc[0], label="Manual")
    plt.plot(portvals_learner / portvals_learner.iloc[0], label="Learner")
    plt.plot(portvals_bench / portvals_bench.iloc[0], label="Benchmark")
    plt.legend()
    plt.title("Normalized Portfolio Values")
    plt.ylabel("Normalized Value")
    plt.savefig("images/p8_normalized.png", dpi=200)

    # non-normalized
    plt.figure()
    plt.plot(portvals_manual, label="Manual ($)")
    plt.plot(portvals_learner, label="Learner ($)")
    plt.plot(portvals_bench, label="Benchmark ($)")
    plt.legend()
    plt.title("Portfolio Values (Non-Normalized)")
    plt.ylabel("Portfolio Value ($)")
    plt.savefig("images/p8_non_normalized.png", dpi=200)


    # ---------- 4) FINAL HOLDINGS BARS ----------
    holdings = df["Final Holdings"].values.astype(int)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, holdings)
    ax.set_title("Final Holdings by Strategy")
    ax.set_ylabel("Shares at End")
    ax.set_xlabel("Strategy")
    for i, v in enumerate(holdings):
        ax.text(
            i,
            v + (max(1, abs(holdings).max()) * 0.03) * (1 if v >= 0 else -1),
            str(v),
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{fname_bars3}", dpi=200, bbox_inches="tight")
    plt.close()

def _as_series(x):
    """Return a Series if x is a 1-col DataFrame or already a Series."""
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x

def build_and_plot_final_summary(
    portvals_manual, portvals_learner, portvals_tos,
    trades_manual, trades_learner, trades_tos,
    out_dir: str = "images"
):
    summary_table = summarize_strategies(
        manual_vals=_as_series(portvals_manual),
        learner_vals=_as_series(portvals_learner),
        tos_vals=_as_series(portvals_tos),
        manual_trades=trades_manual,
        learner_trades=trades_learner,
        tos_trades=trades_tos,
    )
    print("\n========== Final Strategy Comparison ==========")
    print(summary_table.round(
        {"Cumulative Return": 6, "Average Daily Return": 6, "Std Daily Return": 6, "Sharpe Ratio": 6}
    ).to_string())
    make_summary_charts(summary_table, out_dir=out_dir)
    print(
        "Saved charts:",
        f"{out_dir}/p8_final_summary_table.png,",
        f"{out_dir}/p8_final_perf_bars.png,",
        f"{out_dir}/p8_final_trades_bars.png,",
        f"{out_dir}/p8_final_holdings_bars.png",
    )
# ==== end utilities ====


def _get_prices(symbol: str, sd: dt.datetime, ed: dt.datetime) -> pd.DataFrame:
    """Return a 1-col prices DF (Adjusted Close) for symbol over [sd, ed]."""
    dates = pd.date_range(sd, ed)
    prices_all = get_data([symbol], dates)  # includes SPY by default; we’ll slice
    return prices_all[[symbol]]  # keep only the chosen symbol as 1-col DF


if __name__ == "__main__":
    # ----- Config per project spec -----
    symbol = "JPM"
    sv = 100000
    commission = 9.95
    impact = 0.005

    # In-sample period
    sd_in = dt.datetime(2008, 1, 1)
    ed_in = dt.datetime(2009, 12, 31)

    # ----- Load prices for IN-SAMPLE -----
    prices_in = _get_prices(symbol, sd_in, ed_in)

    # ----- Manual Strategy (in-sample) -----
    ms = ManualStrategy()
    trades_manual = ms.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    # Ensure index aligns to prices_in
    trades_manual = trades_manual.reindex(prices_in.index).fillna(0)
    portvals_manual = compute_portvals_from_trades(
        trades_manual, prices_in, start_val=sv, commission=commission, impact=impact
    )

    # ----- Strategy Learner (train in-sample, test in-sample for comparison figure 1) -----
    sl = StrategyLearner(verbose=False, impact=impact, commission=commission)
    sl.add_evidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    trades_learner = sl.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    trades_learner = trades_learner.reindex(prices_in.index).fillna(0)
    portvals_learner = compute_portvals_from_trades(
        trades_learner, prices_in, start_val=sv, commission=commission, impact=impact
    )

    # ----- Benchmark (buy 1000 at start, sell 1000 at end) -----
    trades_bench = pd.DataFrame(0, index=prices_in.index, columns=["Trades"])
    trades_bench.iloc[0, 0] = 1000
    trades_bench.iloc[-1, 0] = -1000
    portvals_bench = compute_portvals_from_trades(
        trades_bench, prices_in, start_val=sv, commission=commission, impact=impact
    )

    # ----- Build & save summary charts (in-sample) -----
    build_and_plot_final_summary(
        portvals_manual=portvals_manual,
        portvals_learner=portvals_learner,
        portvals_tos=portvals_bench,         # treat benchmark as “tos” column here
        trades_manual=trades_manual,
        trades_learner=trades_learner,
        trades_tos=trades_bench,
        out_dir="images"                     # saves into images/
    )

    # --------------------------------------------------------------------
    # OPTIONAL: Out-of-sample summary (uncomment if you also want 2010–2011)
    # --------------------------------------------------------------------
    sd_out = dt.datetime(2010, 1, 1)
    ed_out = dt.datetime(2011, 12, 31)
    prices_out = _get_prices(symbol, sd_out, ed_out)

    # Manual OOS (same rules, just different dates)
    trades_manual_oos = ms.testPolicy(symbol=symbol, sd=sd_out, ed=ed_out, sv=sv)
    trades_manual_oos = trades_manual_oos.reindex(prices_out.index).fillna(0)
    portvals_manual_oos = compute_portvals_from_trades(
        trades_manual_oos, prices_out, start_val=sv, commission=commission, impact=impact
    )

    # Learner OOS (must NOT learn here)
    trades_learner_oos = sl.testPolicy(symbol=symbol, sd=sd_out, ed=ed_out, sv=sv)
    trades_learner_oos = trades_learner_oos.reindex(prices_out.index).fillna(0)
    portvals_learner_oos = compute_portvals_from_trades(
        trades_learner_oos, prices_out, start_val=sv, commission=commission, impact=impact
    )

    # Benchmark OOS
    trades_bench_oos = pd.DataFrame(0, index=prices_out.index, columns=["Trades"])
    trades_bench_oos.iloc[0, 0] = 1000
    trades_bench_oos.iloc[-1, 0] = -1000
    portvals_bench_oos = compute_portvals_from_trades(
        trades_bench_oos, prices_out, start_val=sv, commission=commission, impact=impact
    )

    build_and_plot_final_summary(
        portvals_manual=portvals_manual_oos,
        portvals_learner=portvals_learner_oos,
        portvals_tos=portvals_bench_oos,
        trades_manual=trades_manual_oos,
        trades_learner=trades_learner_oos,
        trades_tos=trades_bench_oos,
        out_dir="images/oos"   # saves a second set into images/oos/
    )
