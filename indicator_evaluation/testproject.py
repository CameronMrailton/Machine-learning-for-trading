import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data
import indicators as ind
import TheoreticallyOptimalStrategy as tos
from marketsimcode import compute_portvals_from_trades

def author():
    return "crailton3"

def study_group():
    return []

def stats(portvals):
    vals = portvals.iloc[:, 0]
    daily_returns = vals.pct_change().dropna()
    cumulative_return = vals.iloc[-1] / vals.iloc[0] - 1.0
    std = daily_returns.std()
    mean = daily_returns.mean()
    sharpe_ratio = np.sqrt(252) * mean / std if std != 0 else np.nan
    return cumulative_return, std, mean, sharpe_ratio

def main():
    symbol = "JPM"
    sd, ed = dt.datetime(2008,1,1), dt.datetime(2009,12,31)
    sv = 100000
    ind.run(symbol, sd, ed)

    # TOS
    df_trades = tos.testPolicy(symbol, sd, ed, sv)
    portvals_tos = compute_portvals_from_trades(df_trades, symbol, sv, 0, 0)

    # Benchmark
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates)[[symbol]].dropna().rename(columns={symbol:"price"})
    bench_trades = pd.DataFrame(index=prices.index, data={"Trades": 0.0})
    bench_trades.iloc[0,0] = 1000.0
    portvals_bench = compute_portvals_from_trades(bench_trades, symbol, sv, 0, 0)

    norm_bench = portvals_bench / portvals_bench.iloc[0,0]
    norm_tos = portvals_tos / portvals_tos.iloc[0,0]

    plt.figure()
    plt.title("JPM Theoretically Optimal Strategy vs Benchmark (Normalized)")
    plt.plot(norm_bench, color="purple", label="Benchmark")
    plt.plot(norm_tos, color="red", label="TOS")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.savefig("./images/tos_vs_benchmark.png"); plt.close()

    bench_cum_ret, bench_std_daily, bench_mean_daily, bench_sharpe = stats(portvals_bench)
    tos_cum_ret, tos_std_daily, tos_mean_daily, tos_sharpe = stats(portvals_tos)

    output = (
        f"Cumulative Return, {bench_cum_ret:.6f}, {tos_cum_ret:.6f}\n"
        f"Std of Daily Returns, {bench_std_daily:.6f}, {tos_std_daily:.6f}\n"
        f"Mean of Daily Returns, {bench_mean_daily:.6f}, {tos_mean_daily:.6f}\n"
        f"Sharpe Ratio, {bench_sharpe:.6f}, {tos_sharpe:.6f}\n"
    )

    with open("images/p6_results.txt", "w") as f:
        f.write(output)

if __name__ == "__main__":
    main()
