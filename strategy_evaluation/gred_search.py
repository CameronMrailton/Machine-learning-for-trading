# gridsearch_all_indicators.py
import datetime as dt
import numpy as np
import pandas as pd

import util as ut
import indicators as ind
from marketsimcode import compute_portvals_from_trades


def author():
    return "crailton3"


def study_group():
    return []


# ----------------- config -----------------
SYMBOL = "JPM"
SV = 100000
COMMISSION = 9.95
IMPACT = 0.005
IS_SD, IS_ED = dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)

# Weight grids (pre-normalization) for ALL 5 indicators
TREND_WS = [0.10, 0.20, 0.30, 0.40]
BBP_WS   = [0.10, 0.20, 0.30, 0.40]
RSI_WS   = [0.10, 0.20, 0.30, 0.40]
MOM_WS   = [0.00, 0.10, 0.20, 0.30]
MACD_WS  = [0.00, 0.05, 0.10, 0.20]

# Composite thresholds to try
THRESHOLDS = [0.20, 0.25, 0.30, 0.35]


def _build_scores(price: pd.Series) -> pd.DataFrame:
    """
    Build the SAME indicator scores as in ManualStrategy._signals,
    but just return them in a DataFrame for reuse in grid search.
    """

    # WMA trend 12 vs 30
    wma_fast = ind.weighted_moving_average(price, window=12)
    wma_slow = ind.weighted_moving_average(price, window=30)
    trend_raw = (wma_fast - wma_slow) / price
    trend_score = np.tanh(trend_raw * 15.0)
    trend_score = pd.Series(trend_score, index=price.index).fillna(0.0)

    # Bollinger %B (20)
    bbp = ind.bollinger_band_percent(price, window=20)
    bbp_score = 1.0 - 2.0 * bbp
    bbp_score = bbp_score.clip(-1.0, 1.0).fillna(0.0)

    # RSI(10)
    rsi_val = ind.rsi(price, window=10)
    rsi_score = (50.0 - rsi_val) / 20.0
    rsi_score = rsi_score.clip(-1.0, 1.0).fillna(0.0)

    # Momentum(7)
    mom = ind.momentum(price, window=7)
    mom_score = np.tanh(mom * 10.0)
    mom_score = pd.Series(mom_score, index=price.index).fillna(0.0)

    # MACD histogram
    macd_hist = ind.macd(price).fillna(0.0)
    macd_std = macd_hist.rolling(30).std().replace(0, np.nan)
    macd_norm = macd_hist / macd_std
    macd_score = np.tanh(macd_norm * 1.0).fillna(0.0)

    scores = pd.DataFrame(
        {
            "trend": trend_score,
            "bbp": bbp_score,
            "rsi": rsi_score,
            "mom": mom_score,
            "macd": macd_score,
        },
        index=price.index,
    )

    return scores


def _signals_from_scores(
    scores: pd.DataFrame,
    w_trend: float,
    w_bbp: float,
    w_rsi: float,
    w_mom: float,
    w_macd: float,
    long_th: float,
    short_th: float,
) -> pd.Series:
    """
    Combine indicator scores with given weights into a composite signal and
    apply the same hysteresis rule as ManualStrategy.
    """

    composite = (
        w_trend * scores["trend"]
        + w_bbp * scores["bbp"]
        + w_rsi * scores["rsi"]
        + w_mom * scores["mom"]
        + w_macd * scores["macd"]
    )

    sig = pd.Series(0, index=scores.index, dtype=int)

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
            prev = sig.iloc[i - 1]
            if val > long_th:
                sig.iloc[i] = 1
            elif val < short_th:
                sig.iloc[i] = -1
            else:
                sig.iloc[i] = 0

    # trade on the NEXT day (no look-ahead)
    return sig.shift(1).fillna(0).astype(int)


def _positions_from_signal(signal: pd.Series) -> pd.Series:
    pos = pd.Series(0, index=signal.index, dtype=int)
    for i in range(len(signal)):
        if signal.iloc[i] > 0:
            pos.iloc[i] = 1000
        elif signal.iloc[i] < 0:
            pos.iloc[i] = -1000
        else:
            pos.iloc[i] = 0 if i == 0 else pos.iloc[i - 1]
    return pos


def _trades_from_positions(positions: pd.Series) -> pd.DataFrame:
    trades = positions.diff().fillna(positions).astype(int)
    trades = trades.clip(-2000, 2000)
    return pd.DataFrame(trades, columns=["Trades"])


def _evaluate_trades(trades: pd.DataFrame) -> dict:
    portvals = compute_portvals_from_trades(
        trades, SYMBOL, start_val=SV, commission=COMMISSION, impact=IMPACT
    )["portval"]

    vals = portvals.astype(float)
    daily = vals.pct_change().dropna()

    cr = vals.iloc[-1] / vals.iloc[0] - 1.0
    adr = daily.mean()
    sdr = daily.std()
    sharpe = np.sqrt(252) * adr / sdr if sdr != 0 else 0.0

    return {
        "cum_return": float(cr),
        "adr": float(adr),
        "sdr": float(sdr),
        "sharpe": float(sharpe),
        "final_value": float(vals.iloc[-1]),
    }


def grid_search_all_indicators():
    # in-sample prices
    dates = pd.date_range(IS_SD, IS_ED)
    prices = ut.get_data([SYMBOL], dates)[[SYMBOL]].dropna()
    price = prices[SYMBOL]

    scores = _build_scores(price)

    results = []
    best_overall = None
    best_macd = None  # best config with MACD weight >= 0.05

    # build all raw weight combos
    weight_combos = []
    for wt in TREND_WS:
        for wb in BBP_WS:
            for wr in RSI_WS:
                for wmom in MOM_WS:
                    for wmacd in MACD_WS:
                        s = wt + wb + wr + wmom + wmacd
                        if s <= 0:
                            continue
                        weight_combos.append((wt, wb, wr, wmom, wmacd))

    total_configs = len(weight_combos) * len(THRESHOLDS)
    cfg_idx = 0
    print(f"=== Grid search over ALL indicators (total combos: {total_configs}) ===")

    for (wt_raw, wb_raw, wr_raw, wmom_raw, wmacd_raw) in weight_combos:
        s = wt_raw + wb_raw + wr_raw + wmom_raw + wmacd_raw

        # normalized weights
        w_trend = wt_raw / s
        w_bbp = wb_raw / s
        w_rsi = wr_raw / s
        w_mom = wmom_raw / s
        w_macd = wmacd_raw / s

        for th in THRESHOLDS:
            cfg_idx += 1
            long_th = th
            short_th = -th

            print(
                f"[{cfg_idx}/{total_configs}] "
                f"w_trend={w_trend:.3f}, w_bbp={w_bbp:.3f}, "
                f"w_rsi={w_rsi:.3f}, w_mom={w_mom:.3f}, w_macd={w_macd:.3f}, "
                f"long_th={long_th:.3f}"
            )

            sig = _signals_from_scores(
                scores, w_trend, w_bbp, w_rsi, w_mom, w_macd, long_th, short_th
            )
            positions = _positions_from_signal(sig)
            trades = _trades_from_positions(positions)
            perf = _evaluate_trades(trades)

            score = (perf["sharpe"], perf["cum_return"])

            results.append(
                {
                    "w_trend": w_trend,
                    "w_bbp": w_bbp,
                    "w_rsi": w_rsi,
                    "w_mom": w_mom,
                    "w_macd": w_macd,
                    "long_th": long_th,
                    "short_th": short_th,
                    "sharpe": perf["sharpe"],
                    "cum_return": perf["cum_return"],
                    "final_value": perf["final_value"],
                }
            )

            # best overall
            if best_overall is None or score > best_overall["score"]:
                best_overall = {
                    "weights": (w_trend, w_bbp, w_rsi, w_mom, w_macd),
                    "long_th": long_th,
                    "short_th": short_th,
                    "perf": perf,
                    "score": score,
                }

            # best with MACD weight >= 0.05
            if w_macd >= 0.05:
                if best_macd is None or score > best_macd["score"]:
                    best_macd = {
                        "weights": (w_trend, w_bbp, w_rsi, w_mom, w_macd),
                        "long_th": long_th,
                        "short_th": short_th,
                        "perf": perf,
                        "score": score,
                    }

    df = pd.DataFrame(results)

    print("\n--- Top 15 configs by Sharpe (in-sample) ---")
    df_sorted = df.sort_values(["sharpe", "cum_return"], ascending=False).head(15)
    print(
        df_sorted.to_string(
            index=False,
            formatters={
                "w_trend": lambda x: f"{x:.3f}",
                "w_bbp": lambda x: f"{x:.3f}",
                "w_rsi": lambda x: f"{x:.3f}",
                "w_mom": lambda x: f"{x:.3f}",
                "w_macd": lambda x: f"{x:.3f}",
                "long_th": lambda x: f"{x:.3f}",
                "short_th": lambda x: f"{x:.3f}",
                "sharpe": lambda x: f"{x:.6f}",
                "cum_return": lambda x: f"{x:.6f}",
                "final_value": lambda x: f"{x:.2f}",
            },
        )
    )

    print("\n===== BEST OVERALL CONFIG (in-sample 2008–2009) =====")
    bo = best_overall
    wt, wb, wr, wmom, wmacd = bo["weights"]
    print(
        f"w_trend={wt:.3f}, w_bbp={wb:.3f}, w_rsi={wr:.3f}, "
        f"w_mom={wmom:.3f}, w_macd={wmacd:.3f}, "
        f"long_th={bo['long_th']:.3f}, short_th={bo['short_th']:.3f}"
    )
    print(
        f"Sharpe={bo['perf']['sharpe']:.6f}, "
        f"CumReturn={bo['perf']['cum_return']:.6f}, "
        f"FinalValue={bo['perf']['final_value']:.2f}"
    )

    if best_macd is not None:
        print("\n===== BEST CONFIG WITH MACD WEIGHT ≥ 0.05 =====")
        bm = best_macd
        wt, wb, wr, wmom, wmacd = bm["weights"]
        print(
            f"w_trend={wt:.3f}, w_bbp={wb:.3f}, w_rsi={wr:.3f}, "
            f"w_mom={wmom:.3f}, w_macd={wmacd:.3f}, "
            f"long_th={bm['long_th']:.3f}, short_th={bm['short_th']:.3f}"
        )
        print(
            f"Sharpe={bm['perf']['sharpe']:.6f}, "
            f"CumReturn={bm['perf']['cum_return']:.6f}, "
            f"FinalValue={bm['perf']['final_value']:.2f}"
        )
    else:
        print("\n(No configs with MACD weight ≥ 0.05 found.)")

    return best_overall, best_macd


if __name__ == "__main__":
    grid_search_all_indicators()
