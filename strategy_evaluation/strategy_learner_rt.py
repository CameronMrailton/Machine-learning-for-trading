"""
Template for implementing StrategyLearner using an RT-based supervised learner
(c) 2015–2018 Georgia Tech (header kept as required)

Student Name: Cameron Railton
GT User ID: crailton3
GT ID: 900897987
"""

import datetime as dt
import numpy as np
import pandas as pd

from util import get_data
from BagLearner import BagLearner
from RTLearner import RTLearner


class StrategyLearnerRT(object):
    """
    A strategy learner that uses a bag of Random Tree learners to predict
    future 5-day returns and map them to {-1, 0, +1} signals.

    Public API matches StrategyLearner:
      - __init__(verbose=False, impact=0.0, commission=0.0)
      - add_evidence(symbol, sd, ed, sv=100000)
      - testPolicy(symbol, sd, ed, sv=100000)
    """

    def author(self):
        return "crailton3"

    def __init__(self,
                 verbose=False,
                 impact=0.0,
                 commission=0.0,
                 leaf_size=5,
                 bags=20,
                 lookback=20,
                 horizon=5,
                 long_thresh=0.03,
                 short_thresh=-0.03,
                 pred_long_thresh=0.5,
                 pred_short_thresh=-0.5):
        """
        :param impact: market impact per trade (kept for compatibility)
        :param commission: fixed commission per order (kept for compatibility)
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

        self.leaf_size = leaf_size
        self.bags = bags
        self.lookback = lookback
        self.horizon = horizon
        self.long_thresh = long_thresh
        self.short_thresh = short_thresh
        self.pred_long_thresh = pred_long_thresh
        self.pred_short_thresh = pred_short_thresh

        self.learner = None
        self.trained_symbol = None

    # ------------------ internal helpers ------------------ #
    def _prices(self, symbol, sd, ed):
        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates)[[symbol]].dropna()
        return prices

    def _build_features(self, symbol, sd, ed):
        """
        Build indicator matrix X and future-return labels y for [sd, ed].
        X index will be aligned with y (we drop NaNs and the last horizon days).
        """
        prices = self._prices(symbol, sd, ed)
        sym = prices.columns[0]

        # --- basic indicators (lagged-only, no look-ahead) --- #
        sma = prices.rolling(self.lookback).mean()
        price_sma = (prices / sma) - 1.0

        rolling_std = prices.rolling(self.lookback).std()
        bb_value = (prices - sma) / (2.0 * rolling_std)

        momentum = prices / prices.shift(self.lookback) - 1.0

        daily_ret = prices.pct_change()
        volatility = daily_ret.rolling(self.lookback).std()

        feats = pd.concat(
            [price_sma.rename(columns={sym: "price_sma"}),
             bb_value.rename(columns={sym: "bb"}),
             momentum.rename(columns={sym: "mom"}),
             volatility.rename(columns={sym: "vol"})],
            axis=1
        ).dropna()

        # Align prices with features
        prices = prices.loc[feats.index]

        # --- future horizon return for labels --- #
        future_price = prices.shift(-self.horizon)
        fut_ret = (future_price / prices) - 1.0
        fut_ret = fut_ret.loc[feats.index]
        fut_ret = fut_ret.iloc[:-self.horizon]
        feats = feats.iloc[:-self.horizon]

        y_cont = fut_ret.iloc[:, 0].values

        # Map to discrete signals {-1, 0, +1}
        y_disc = np.zeros_like(y_cont)
        y_disc[y_cont > self.long_thresh] = 1
        y_disc[y_cont < self.short_thresh] = -1

        if self.verbose:
            print("Built features:")
            print("  X shape:", feats.shape)
            print("  y distribution:",
                  {k: int((y_disc == k).sum()) for k in [-1, 0, 1]})

        return feats.values, y_disc, feats.index

    def _signals_to_trades(self, dates, signals, n_shares=1000):
        """
        Convert per-day position signal (-1,0,1) into Trades DF obeying
        +/-n_shares position constraint.
        """
        pos_desired = signals * n_shares
        pos_desired = pd.Series(pos_desired, index=dates)

        trades = pos_desired.diff().fillna(pos_desired.iloc[0])
        trades = trades.to_frame(name="Trades")

        # Enforce holdings constraint
        holdings = trades["Trades"].cumsum()
        holdings = holdings.clip(-n_shares, n_shares)
        trades["Trades"] = holdings.diff().fillna(holdings.iloc[0])

        return trades

    # ------------------ public API ------------------ #
    def add_evidence(self, symbol="JPM", sd=dt.datetime(2008, 1, 1),
                     ed=dt.datetime(2009, 12, 31), sv=100000):
        """
        Train the RT-based strategy learner on the given in-sample period.
        """
        X, y, idx = self._build_features(symbol, sd, ed)

        self.learner = BagLearner(
            learner=RTLearner,
            kwargs={"leaf_size": self.leaf_size},
            bags=self.bags,
            boost=False,
            verbose=False,
        )
        self.learner.add_evidence(X, y)
        self.trained_symbol = symbol

        if self.verbose:
            print("StrategyLearnerRT trained on", symbol,
                  "from", idx[0].date(), "to", idx[-1].date())

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31), sv=100000):
        """
        Use the trained learner to generate trades for [sd, ed].
        """
        if self.learner is None or self.trained_symbol != symbol:
            # safety: train if not already trained for this symbol
            self.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)

        X, _, idx = self._build_features(symbol, sd, ed)
        raw_preds = self.learner.query(X)

        # Convert predictions to {-1, 0, 1} using thresholds
        signals = np.zeros_like(raw_preds, dtype=int)
        signals[raw_preds > self.pred_long_thresh] = 1
        signals[raw_preds < self.pred_short_thresh] = -1

        trades = self._signals_to_trades(idx, signals, n_shares=1000)
        return trades


if __name__ == "__main__":
    print("StrategyLearnerRT module. Try running testproject_rt.py.")
