import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from pathlib import Path
from util import get_data

def author():
    return "crailton3"

def study_group():
    return []

def weighted_moving_average(price, window=20):
    weights = np.arange(1, window + 1)
    wma = price.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    return wma.squeeze()

def bollinger_band_percent(price, window=20):
    wma_val = weighted_moving_average(price, window)
    rolling_std = price.rolling(window).std()
    upper_band = wma_val + 2 * rolling_std
    lower_band = wma_val - 2 * rolling_std
    bbp = (price - lower_band) / (upper_band - lower_band)
    return bbp.squeeze()

def momentum(price, window=10):
    return (price / price.shift(window)) - 1

def macd(price, fast=12, slow=26, signal=9):
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return hist.squeeze()

def rsi(price, window=14):
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50).squeeze()

def compute_wma_cross_signals(price, wma_series):
    """
    Return boolean Series for BUY/SELL crossover events based on price vs WMA.
      BUY  when price crosses above WMA (yesterday <=, today >)
      SELL when price crosses below WMA (yesterday >=, today <)
    """
    prev_price = price.shift(1)
    prev_wma = wma_series.shift(1)

    cross_up = (prev_price <= prev_wma) & (price > wma_series)
    cross_dn = (prev_price >= prev_wma) & (price < wma_series)

    return cross_up.fillna(False), cross_dn.fillna(False)

def run(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31)):

    prices = get_data([symbol], pd.date_range(sd, ed))[[symbol]].dropna()
    price = prices[symbol]

    # Weighted Moving Average
    wma_val = weighted_moving_average(price, window=20)
    buy_sig, sell_sig = compute_wma_cross_signals(price, wma_val)
    buy_idx = price.index[buy_sig]
    sell_idx = price.index[sell_sig]

    plt.figure(figsize=(10, 6))
    plt.title(f"{symbol} WMA(20) Buy/Sell Signals")
    plt.plot(price, color="blue", label="Price", linewidth=1.5)
    plt.plot(wma_val, color="orange", label="WMA(20)", linewidth=1.8)
    plt.scatter(buy_idx, price.loc[buy_idx], marker="^", s=150, color="limegreen",edgecolor="black", linewidths=0.7, alpha=0.9, label="BUY", zorder=5)
    plt.scatter(sell_idx, price.loc[sell_idx], marker="v", s=150, color="red",edgecolor="black", linewidths=0.7, alpha=0.9, label="SELL", zorder=5)
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("images/indicator_wma_signals.png")
    plt.close()

    # # Bollinger Bands %
    bbp = bollinger_band_percent(price, window=20)
    buy_sig_bb = bbp < 0.2
    sell_sig_bb = bbp > 0.8

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.set_title(f"{symbol} Bollinger Band %B Buy/Sell Signals", fontsize=12)
    ax1.plot(price, color="black", label="Price", linewidth=2.0, alpha=0.8)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax2 = ax1.twinx()
    ax2.plot(bbp, color="royalblue", label="%B", linewidth=2)
    ax2.axhline(0.2, color="limegreen", linestyle="--", linewidth=1.2)
    ax2.axhline(0.8, color="red", linestyle="--", linewidth=1.2)

    ax2.scatter(bbp.index[buy_sig_bb], bbp[buy_sig_bb], marker="^", s=130, color="limegreen", edgecolor="black",
        linewidths=0.8, alpha=0.95, label="BUY", zorder=5)
    ax2.scatter(bbp.index[sell_sig_bb], bbp[sell_sig_bb], marker="v", s=130, color="red", edgecolor="black",
        linewidths=0.8, alpha=0.95, label="SELL", zorder=5)
    ax2.set_ylabel("Bollinger Band %", color="royalblue")
    ax2.tick_params(axis="y", labelcolor="royalblue")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("images/indicator_bbp_signals.png", dpi=200)
    plt.close()

    # RSI
    rsi_val = rsi(price, window=14)
    buy_signals = (rsi_val < 30)
    sell_signals = (rsi_val > 70)

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.set_title(f"{symbol} RSI(14) with Buy/Sell Signals", fontsize=12)
    ax1.plot(price, color="black", linewidth=2, alpha=0.8, label="Price")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    ax2 = ax1.twinx()
    ax2.plot(rsi_val, color="royalblue", linewidth=2, label="RSI")
    ax2.axhline(70, color="red", linestyle="--", linewidth=1.2, label="Overbought (70)")
    ax2.axhline(30, color="limegreen", linestyle="--", linewidth=1.2, label="Oversold (30)")

    ax2.scatter(rsi_val.index[buy_signals], rsi_val[buy_signals], marker="^", s=130, color="limegreen", edgecolor="black", linewidths=0.8,
        alpha=0.95, label="BUY", zorder=5)
    ax2.scatter(rsi_val.index[sell_signals], rsi_val[sell_signals], marker="v", s=130, color="red", edgecolor="black", linewidths=0.8,
        alpha=0.95, label="SELL", zorder=5)
    ax2.set_ylabel("RSI", color="royalblue")
    ax2.tick_params(axis="y", labelcolor="royalblue")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left", frameon=True, facecolor="white", framealpha=0.8, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("images/indicator_rsi_signals.png", dpi=200)
    plt.close()

    # Momentum
    mom = momentum(price, window=10)
    buy_signals = mom > 0.08
    sell_signals = mom < -0.08

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.set_title(f"{symbol} Momentum(10) with Buy/Sell Signals", fontsize=12)
    ax1.plot(price, color="black", linewidth=2, alpha=0.8, label="Price")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax2 = ax1.twinx()
    ax2.plot(mom, color="royalblue", linewidth=2, label="Momentum")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1.2)

    ax2.scatter(mom.index[buy_signals], mom[buy_signals], marker="^", s=130, color="limegreen", edgecolor="black", linewidths=0.8,
        alpha=0.95, label="BUY", zorder=5)
    ax2.scatter(mom.index[sell_signals], mom[sell_signals], marker="v", s=130, color="red", edgecolor="black", linewidths=0.8,
        alpha=0.95, label="SELL", zorder=5)

    ax2.set_ylabel("Momentum", color="royalblue")
    ax2.tick_params(axis="y", labelcolor="royalblue")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left", frameon=True, facecolor="white", framealpha=0.8, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("images/indicator_momentum_signals.png", dpi=200)
    plt.close()

    # MACD
    macd_hist = macd(price)
    prev_hist = macd_hist.shift(1)
    buy_sig_macd  = (prev_hist <= 0) & (macd_hist > 0)
    sell_sig_macd = (prev_hist >= 0) & (macd_hist < 0)
    buy_idx  = macd_hist.index[buy_sig_macd]
    sell_idx = macd_hist.index[sell_sig_macd]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title(f"{symbol} MACD Histogram Signals with Buy/Sell Signals")
    ax1.plot(price, color="black", linewidth=2.0, alpha=0.8, label="Price")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax2 = ax1.twinx()
    ax2.plot(macd_hist, color="royalblue", linewidth=2, label="MACD Hist")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1.2)

    ax2.scatter(buy_idx,  macd_hist.loc[buy_idx], marker="^", s=130, color="limegreen", edgecolor="black",
        linewidths=0.8, alpha=0.95, label="BUY", zorder=5)
    ax2.scatter(sell_idx, macd_hist.loc[sell_idx], marker="v", s=130, color="red", edgecolor="black",
        linewidths=0.8, alpha=0.95, label="SELL", zorder=5)

    ax2.set_ylabel("MACD Histogram", color="royalblue")
    ax2.tick_params(axis="y", labelcolor="royalblue")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=9)

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("images/indicator_macd_signals.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    run()
