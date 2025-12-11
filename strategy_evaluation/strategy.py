

# =============================
# File: StrategyLearner.py
# =============================
import numpy as np
import pandas as pd
import datetime as dt
from util import get_data
import indicators as ind
from BagLearner import BagLearner
from RTLearner import RTLearner

# ----- API -----
def author():
    return "crailton3"

def study_group():
    return []

class StrategyLearner:
    def __init__(self, verbose=False, impact=0.0, commission=0.0,
                 bags=20, leaf_size=5, n_bins=7, lookahead=5):
        self.verbose = verbose
        self.impact = float(impact)
        self.commission = float(commission)
        self.bags = int(bags)
        self.leaf_size = max(5, int(leaf_size))  # per spec: >=5 to avoid degenerate overfit
        self.n_bins = int(n_bins)
        self.lookahead = int(lookahead)
        self.model = None
        self.features_cols = None

    # ---------- Feature engineering (must match ManualStrategy indicators) ----------
    def _build_features(self, price: pd.Series) -> pd.DataFrame:
        wma20 = ind.weighted_moving_average(price, window=20)
        bbp20 = ind.bollinger_band_percent(price, window=20)
        rsi14 = ind.rsi(price, window=14)
        mom10 = ind.momentum(price, window=10)
        feats = pd.DataFrame({
            "px_wma_spread": (price - wma20) / price,   # scale-free
            "bbp": bbp20,
            "rsi": rsi14 / 100.0,                      # 0..1
            "mom10": mom10,
        }, index=price.index)
        return feats

    def _discretize(self, s: pd.Series, n_bins: int) -> pd.Series:
        # robust cuts with quantiles
        qs = np.linspace(0, 1, n_bins+1)
        edges = s.dropna().quantile(qs).values
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        return pd.cut(s, bins=edges, labels=False, include_lowest=True)

    def _labels(self, price: pd.Series) -> pd.Series:
        # Forward return over lookahead, adjusted for impact as a gap (wider no-trade band)
        fwd = price.shift(-self.lookahead) / price - 1.0
        thr = 2 * self.impact  # simple mapping of impact -> indecision band
        y = pd.Series(0, index=price.index)
        y[fwd > thr] = 1
        y[fwd < -thr] = -1
        return y

    def add_evidence(self, symbol="AAPL", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):
        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates)[[symbol]].dropna()
        price = prices[symbol]

        X = self._build_features(price)
        # discretize each column to n_bins
        Xd = X.apply(lambda col: self._discretize(col, self.n_bins))
        y = self._labels(price)

        df = pd.concat([Xd, y.rename("y")], axis=1).dropna()
        Xtrain = df[X.columns].values.astype(float)
        ytrain = df["y"].values.astype(int)

        # Map to classification targets {0,1,2} to avoid negative labels in some trees
        ytrain = (ytrain + 1).astype(int)  # -1->0, 0->1, +1->2

        base = RTLearner(leaf_size=self.leaf_size, verbose=False)
        self.model = BagLearner(learner=RTLearner, kwargs={"leaf_size": self.leaf_size},
                                bags=self.bags, boost=False, verbose=False, classification=True)
        # Our BagLearner should interpret classification=True to take mode at query
        self.model.add_evidence(Xtrain, ytrain)
        self.features_cols = list(X.columns)

    def testPolicy(self, symbol="AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv=100000):
        # IMPORTANT: No learning here; must be deterministic and fast.
        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates)[[symbol]].dropna()
        price = prices[symbol]

        X = self._build_features(price)
        Xd = X.apply(lambda col: self._discretize(col, self.n_bins)).dropna()
        # Align to Xd index only
        idx = Xd.index
        preds = self.model.query(Xd.values.astype(float))  # returns {0,1,2}
        preds = pd.Series(preds, index=idx).astype(int) - 1  # back to {-1,0,1}

        # Map class -> target position {-1000,0,1000}; keep prior when class==0
        pos = pd.Series(0, index=prices.index, dtype=int)
        for i, t in enumerate(preds.index):
            if i == 0:
                pos.loc[t] = 0 if preds.iloc[i] == 0 else (1000 if preds.iloc[i] > 0 else -1000)
            else:
                if preds.iloc[i] == 0:
                    pos.loc[t] = pos.iloc[i-1]
                else:
                    pos.loc[t] = 1000 if preds.iloc[i] > 0 else -1000
        pos = pos.reindex(prices.index).ffill().fillna(0).astype(int)

        trades = pos.diff().fillna(pos).astype(int).clip(-2000, 2000)
        return pd.DataFrame(trades, columns=["Trades"])  # grader expects column name


# =============================
# File: experiment1.py
# =============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from pathlib import Path
from util import get_data
from marketsimcode import compute_portvals_from_trades
from ManualStrategy import ManualStrategy, plot_equity_with_entries, _normalize
import StrategyLearner as sl


def author():
    return "crailton3"

def study_group():
    return []

IS_SD, IS_ED = dt.datetime(2008,1,1), dt.datetime(2009,12,31)
OOS_SD, OOS_ED = dt.datetime(2010,1,1), dt.datetime(2011,12,31)
SYMBOL = "JPM"
SV = 100000


def run():
    Path("images").mkdir(exist_ok=True)

    ms = ManualStrategy()
    is_trades = ms.testPolicy(symbol=SYMBOL, sd=IS_SD, ed=IS_ED, sv=SV)
    oos_trades = ms.testPolicy(symbol=SYMBOL, sd=OOS_SD, ed=OOS_ED, sv=SV)

    plot_equity_with_entries(SYMBOL, IS_SD, IS_ED, is_trades,
                             title=f"ManualStrategy vs Benchmark (In-sample) — {SYMBOL}",
                             out_png="images/p8_ms_vs_bench_in_sample.png")
    plot_equity_with_entries(SYMBOL, OOS_SD, OOS_ED, oos_trades,
                             title=f"ManualStrategy vs Benchmark (Out-of-sample) — {SYMBOL}",
                             out_png="images/p8_ms_vs_bench_out_sample.png")

    learner = sl.StrategyLearner(verbose=False, impact=0.005, commission=9.95)
    learner.add_evidence(symbol=SYMBOL, sd=IS_SD, ed=IS_ED, sv=SV)

    is_trades_learn = learner.testPolicy(symbol=SYMBOL, sd=IS_SD, ed=IS_ED, sv=SV)
    oos_trades_learn = learner.testPolicy(symbol=SYMBOL, sd=OOS_SD, ed=OOS_ED, sv=SV)

    # Combined charts (MS, SL, Benchmark) — In-sample
    def chart_combo(sd, ed, ms_trd, sl_trd, out_png, title):
        dates = pd.date_range(sd, ed)
        prices = get_data([SYMBOL], dates)[[SYMBOL]].dropna()
        bench_trades = pd.DataFrame(0, index=prices.index, columns=["Trades"])  # B&H 1000
        bench_trades.iloc[0, 0] = 1000
        bench_trades.iloc[-1, 0] = -1000

        pv_ms = compute_portvals_from_trades(ms_trd, prices, start_val=SV, commission=9.95, impact=0.005).iloc[:,0]
        pv_sl = compute_portvals_from_trades(sl_trd, prices, start_val=SV, commission=9.95, impact=0.005).iloc[:,0]
        pv_bm = compute_portvals_from_trades(bench_trades, prices, start_val=SV, commission=9.95, impact=0.005).iloc[:,0]

        plt.figure(figsize=(10,6))
        plt.plot(_normalize(pv_ms), label="ManualStrategy", linewidth=2)
        plt.plot(_normalize(pv_sl), label="StrategyLearner", linewidth=2)
        plt.plot(_normalize(pv_bm), label="Benchmark", linewidth=2)
        plt.title(title); plt.xlabel("Date"); plt.ylabel("Normalized Value")
        plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

    chart_combo(IS_SD, IS_ED, is_trades, is_trades_learn, "images/p8_combo_in_sample.png",
                f"Manual vs Learner vs Benchmark (In-sample) — {SYMBOL}")
    chart_combo(OOS_SD, OOS_ED, oos_trades, oos_trades_learn, "images/p8_combo_out_sample.png",
                f"Manual vs Learner vs Benchmark (Out-of-sample) — {SYMBOL}")


if __name__ == "__main__":
    run()


# =============================
# File: experiment2.py
# =============================
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

IS_SD, IS_ED = dt.datetime(2008,1,1), dt.datetime(2009,12,31)
SYMBOL = "JPM"
SV = 100000
COMM = 0.00  # per spec for Exp 2 charts

IMPACTS = [0.000, 0.005, 0.020]


def run():
    Path("images").mkdir(exist_ok=True)
    metrics = []

    for imp in IMPACTS:
        learner = sl.StrategyLearner(verbose=False, impact=imp, commission=COMM)
        learner.add_evidence(symbol=SYMBOL, sd=IS_SD, ed=IS_ED, sv=SV)
        trades = learner.testPolicy(symbol=SYMBOL, sd=IS_SD, ed=IS_ED, sv=SV)

        dates = pd.date_range(IS_SD, IS_ED)
        prices = get_data([SYMBOL], dates)[[SYMBOL]].dropna()
        pv = compute_portvals_from_trades(trades, prices, start_val=SV, commission=COMM, impact=imp).iloc[:,0]

        # Metrics: number of trades, cumulative return
        n_trades = (trades["Trades"] != 0).sum()
        cum_ret = pv.iloc[-1] / pv.iloc[0] - 1.0
        metrics.append({"impact": imp, "n_trades": n_trades, "cum_return": cum_ret})

    dfm = pd.DataFrame(metrics)
    dfm.sort_values("impact", inplace=True)
    dfm.to_csv("images/exp2_metrics.csv", index=False)

    # Chart 1: #Trades vs Impact
    plt.figure(figsize=(8,5))
    plt.bar(dfm["impact"].astype(str), dfm["n_trades"])
    plt.title("Experiment 2: Number of trades vs. impact (in-sample)")
    plt.xlabel("Impact"); plt.ylabel("Count of trade days")
    plt.tight_layout(); plt.savefig("images/p8_exp2_trades_vs_impact.png", dpi=150); plt.close()

    # Chart 2: Cum Return vs Impact
    plt.figure(figsize=(8,5))
    plt.bar(dfm["impact"].astype(str), dfm["cum_return"])
    plt.title("Experiment 2: Cumulative return vs. impact (in-sample)")
    plt.xlabel("Impact"); plt.ylabel("Cumulative return")
    plt.tight_layout(); plt.savefig("images/p8_exp2_cumret_vs_impact.png", dpi=150); plt.close()


if __name__ == "__main__":
    run()


# =============================
# File: testproject.py
# =============================
import datetime as dt
from pathlib import Path
import pandas as pd
from ManualStrategy import ManualStrategy, perf_table
import StrategyLearner as sl
import experiment1
import experiment2


def author():
    return "crailton3"

def study_group():
    return []

SYMBOL = "JPM"
IS_SD, IS_ED = dt.datetime(2008,1,1), dt.datetime(2009,12,31)
OOS_SD, OOS_ED = dt.datetime(2010,1,1), dt.datetime(2011,12,31)
SV = 100000
COMM, IMPACT = 9.95, 0.005


def main():
    Path("images").mkdir(exist_ok=True)

    # --- Manual Strategy tables & charts ---
    ms = ManualStrategy()
    ms_is_trades = ms.testPolicy(symbol=SYMBOL, sd=IS_SD, ed=IS_ED, sv=SV, commission=COMM, impact=IMPACT)
    ms_oos_trades = ms.testPolicy(symbol=SYMBOL, sd=OOS_SD, ed=OOS_ED, sv=SV, commission=COMM, impact=IMPACT)

    tbl_is = perf_table(SYMBOL, IS_SD, IS_ED, ms_is_trades, sv=SV, commission=COMM, impact=IMPACT)
    tbl_oos = perf_table(SYMBOL, OOS_SD, OOS_ED, ms_oos_trades, sv=SV, commission=COMM, impact=IMPACT)
    pd.concat({"In-sample": tbl_is, "Out-of-sample": tbl_oos}, axis=0).to_csv("images/p8_ms_perf_tables.csv")

    # --- Experiment 1 charts (MS vs SL both IS and OOS) ---
    experiment1.run()

    # --- Experiment 2 charts ---
    experiment2.run()


if __name__ == "__main__":
    main()
