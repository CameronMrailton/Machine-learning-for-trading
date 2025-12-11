import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from pathlib import Path
import util as ut
import indicators as ind
from marketsimcode import compute_portvals_from_trades
from matplotlib.lines import Line2D

def author():
    return "crailton3"

def study_group():
    return []

class ManualStrategy:

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _signals(self, price: pd.Series) -> pd.Series:
        #wma
        wma_fast = ind.weighted_moving_average(price, window=12)
        wma_slow = ind.weighted_moving_average(price, window=30)
        trend_raw = (wma_fast - wma_slow) / price

        trend_score = np.tanh(trend_raw * 15.0)
        trend_score = pd.Series(trend_score, index=price.index).fillna(0.0)

        #BBP
        bbp = ind.bollinger_band_percent(price, window=20)
        bbp_score = 1.0 - 2.0 * bbp
        bbp_score = bbp_score.clip(-1.0, 1.0).fillna(0.0)

        #RSI
        rsi_val = ind.rsi(price, window=10)
        rsi_score = (50.0 - rsi_val) / 20.0
        rsi_score = rsi_score.clip(-1.0, 1.0).fillna(0.0)

        #Momentum
        mom = ind.momentum(price, window=7)
        mom_score = np.tanh(mom * 10.0)
        mom_score = pd.Series(mom_score, index=price.index).fillna(0.0)

        #MACD
        macd_hist = ind.macd(price).fillna(0.0)
        macd_std = macd_hist.rolling(30).std().replace(0, np.nan)
        macd_norm = macd_hist / macd_std
        macd_score = np.tanh(macd_norm * 1.0)
        macd_score = macd_score.fillna(0.0)

        #Weighting
        composite = (0.35 * trend_score + 0.20 * bbp_score + 0.20 * rsi_score + 0.00 * mom_score + 0.00 * macd_score)

        sig = pd.Series(0, index=price.index, dtype=int)
        long_th = 0.25
        short_th = -0.25

        for i in range(len(composite)):
            val = composite.iloc[i]
            if i == 0:
                if val > long_th:
                    sig.iloc[i] = 1
                elif val < short_th:
                    sig.iloc[i] = -1
                else:
                    sig.iloc[i] = 0
            else:
                if val > long_th:
                    sig.iloc[i] = 1
                elif val < short_th:
                    sig.iloc[i] = -1
                else:
                    sig.iloc[i] = 0

        return sig.shift(1).fillna(0).astype(int)


    def _positions_from_signal(self, signal: pd.Series) -> pd.Series:
        pos = pd.Series(0, index=signal.index, dtype=int)
        for i in range(len(signal)):
            if signal.iloc[i] > 0:
                pos.iloc[i] = 1000
            elif signal.iloc[i] < 0:
                pos.iloc[i] = -1000
            else:
                pos.iloc[i] = 0 if i == 0 else pos.iloc[i - 1]
        return pos

    def _trades_from_positions(self, positions: pd.Series) -> pd.DataFrame:
        trades = positions.diff().fillna(positions).astype(int)
        trades = trades.clip(-2000, 2000)
        return pd.DataFrame(trades, columns=["Trades"])


    def testPolicy(self, symbol="JPM",
                   sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31),
                   sv=100000,
                   commission=9.95,
                   impact=0.005) -> pd.DataFrame:

        dates = pd.date_range(sd, ed)
        prices = ut.get_data([symbol], dates)[[symbol]].dropna()
        price = prices[symbol]

        signal = self._signals(price)
        positions = self._positions_from_signal(signal)
        df_trades = self._trades_from_positions(positions)

        if self.verbose:
            print("First 10 trades:\n", df_trades.head(10))

        return df_trades

def _normalize(s: pd.Series) -> pd.Series:
    return s / s.iloc[0]


def plot_equity_with_entries(symbol, sd, ed, trades_df, title, out_png,
                             sv=100000, commission=9.95, impact=0.005):

    Path(Path(out_png).parent).mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(sd, ed)
    prices = ut.get_data([symbol], dates)[[symbol]].dropna()
    trades = trades_df.reindex(prices.index).fillna(0)

    bench_trades = pd.DataFrame(0, index=prices.index, columns=["Trades"])
    bench_trades.iloc[0, 0] = 1000
    bench_trades.iloc[-1, 0] = -1000

    pv_manual = compute_portvals_from_trades(
        trades, symbol, start_val=sv, commission=commission, impact=impact
    )["portval"]
    pv_bench = compute_portvals_from_trades(
        bench_trades, symbol, start_val=sv, commission=commission, impact=impact
    )["portval"]

    long_entries = trades[trades["Trades"] > 0].index
    short_entries = trades[trades["Trades"] < 0].index

    plt.figure(figsize=(10, 6))
    plt.plot(_normalize(pv_manual), color="red", label="Manual Strategy", linewidth=2)
    plt.plot(_normalize(pv_bench), color="purple", label="Benchmark", linewidth=2)

    for d in long_entries:
        plt.axvline(d, color="blue", linewidth=1, alpha=0.8)
    for d in short_entries:
        plt.axvline(d, color="black", linewidth=1, alpha=0.8)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")

    legend_lines = [
        Line2D([0], [0], color="red",    lw=2, label="Manual Strategy"),
        Line2D([0], [0], color="purple", lw=2, label="Benchmark"),
        Line2D([0], [0], color="blue",   lw=2, label="Long Entry (BUY)"),
        Line2D([0], [0], color="black",  lw=2, label="Short Entry (SELL)"),
    ]
    plt.legend(handles=legend_lines, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def perf_table(symbol, start, end, trades_df,
               sv=100000, commission=9.95, impact=0.005) -> pd.DataFrame:
    """
    Return performance table comparing Benchmark vs Manual.
    Rows: Benchmark, Manual
    Cols: cum_return, stdev_daily, mean_daily
    """
    dates = pd.date_range(start, end)
    prices = ut.get_data([symbol], dates)[[symbol]].dropna()
    trades = trades_df.reindex(prices.index).fillna(0)

    # Manual
    portvals = compute_portvals_from_trades(
        trades, symbol, start_val=sv, commission=commission, impact=impact
    )["portval"]

    # Benchmark
    bench_trades = pd.DataFrame(0, index=prices.index, columns=["Trades"])
    bench_trades.iloc[0, 0] = 1000
    bench_trades.iloc[-1, 0] = -1000
    bench_vals = compute_portvals_from_trades(
        bench_trades, symbol, start_val=sv, commission=commission, impact=impact
    )["portval"]

    def _stats(vals: pd.Series) -> pd.Series:
        dr = vals.pct_change().dropna()
        cr = vals.iloc[-1] / vals.iloc[0] - 1.0
        return pd.Series({
            "cum_return": float(cr),
            "stdev_daily": float(dr.std()),
            "mean_daily": float(dr.mean())
        })

    table = pd.concat(
        {"Benchmark": _stats(bench_vals), "Manual": _stats(portvals)},
        axis=1
    ).T

    return table.applymap(lambda x: float(f"{x:.6f}"))
