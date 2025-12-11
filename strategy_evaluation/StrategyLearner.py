import datetime as dt
import numpy as np
import pandas as pd
import util as ut
import indicators as ind
import QLearner as ql

def author():
    return "crailton3"

def study_group():
    return []

class StrategyLearner(object):
    A_SHORT = 0
    A_CASH = 1
    A_LONG = 2
    NUM_ACTIONS = 3
    MAX_HOLDINGS = 1000
    DEFAULT_WMA_FAST = 12
    DEFAULT_WMA_SLOW = 30
    DEFAULT_BB = 20
    DEFAULT_RSI = 10
    DEFAULT_MOM = 7
    DEFAULT_MACD_SIGNAL = 9

    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = float(impact)
        self.commission = float(commission)

        self.wma_fast = self.DEFAULT_WMA_FAST
        self.wma_slow = self.DEFAULT_WMA_SLOW
        self.bb_window = self.DEFAULT_BB
        self.rsi_window = self.DEFAULT_RSI
        self.mom_window = self.DEFAULT_MOM
        self.macd_signal = self.DEFAULT_MACD_SIGNAL

        # 5 indicators with 3 bins each
        self.num_states = 243

        self.learner = ql.QLearner(
            num_states=self.num_states,
            num_actions=self.NUM_ACTIONS,
            alpha=0.1,
            gamma=0.9,
            rar=0.3,
            radr=0.99,
            dyna=0,
            verbose=False,
        )


    def _get_prices(self, symbol, sd, ed):
        dates = pd.date_range(sd, ed)
        df = ut.get_data([symbol], dates)[[symbol]]
        return df.dropna()

    def _build_indicators(self, price: pd.Series) -> pd.DataFrame:
        wma_fast = ind.weighted_moving_average(price, self.wma_fast)
        wma_slow = ind.weighted_moving_average(price, self.wma_slow)
        trend_spread = ((wma_fast - wma_slow) / price).rename("trend")
        bbp = ind.bollinger_band_percent(price, self.bb_window).rename("bbp")
        rsi = ind.rsi(price, self.rsi_window).rename("rsi")
        mom = ind.momentum(price, self.mom_window).rename("mom")
        macd_hist = ind.macd(price, signal=self.macd_signal).rename("macd")

        feats = pd.concat([trend_spread, bbp, rsi, mom, macd_hist], axis=1)
        feats = feats.replace([np.inf, -np.inf], np.nan).dropna()

        return feats

    def _bin3(self, val, low, high):
        """3-bin helper: <low → 0, between → 1, >high → 2"""
        if val < low:
            return 0
        elif val > high:
            return 2
        return 1

    def _row_to_state(self, row) -> int:
        b_trend = self._bin3(row["trend"], -0.01, 0.01)
        b_bbp   = self._bin3(row["bbp"], 0.33, 0.66)
        b_rsi   = self._bin3(row["rsi"], 35, 65)
        b_mom   = self._bin3(row["mom"], -0.01, 0.01)
        b_macd  = self._bin3(row["macd"], -0.005, 0.005)

        state = (
            b_trend
            + 3 * b_bbp
            + 9 * b_rsi
            + 27 * b_mom
            + 81 * b_macd
        )
        return int(state)


    def _action_to_target_holding(self, a: int) -> int:
        if a == self.A_SHORT:
            return -self.MAX_HOLDINGS
        if a == self.A_CASH:
            return 0
        return self.MAX_HOLDINGS

    def _apply_action(self, a: int, prev_holding: int):

        target = self._action_to_target_holding(a)
        trade = target - prev_holding

        trade = int(np.clip(trade, -2000, 2000))

        new_holding = prev_holding + trade
        new_holding = int(np.clip(new_holding, -self.MAX_HOLDINGS, self.MAX_HOLDINGS))

        trade = new_holding - prev_holding
        return trade, new_holding

    def _step_reward(self, v0: float, v1: float) -> float:
        if v0 <= 0:
            return 0.0
        return (v1 / v0) - 1.0

    def _best_action(self, s: int) -> int:
        row = self.learner.qtable[s, :]
        maxv = row.max()
        ties = np.flatnonzero(row == maxv)
        return int(ties[0])


    def add_evidence(self, symbol="IBM",
                     sd=dt.datetime(2008, 1, 1),
                     ed=dt.datetime(2009, 1, 1),
                     sv=100000):

        prices_df = self._get_prices(symbol, sd, ed)
        price = prices_df[symbol]

        feats = self._build_indicators(price)
        prices = price.reindex(feats.index).dropna()
        feats = feats.reindex(prices.index)

        if len(prices) < 2:
            return

        idx = prices.index
        n_days = len(idx)

        self.learner.qtable[:] = 0.0
        self.learner.rar = 0.3
        self.learner.radr = 0.99

        n_episodes = 100

        for ep in range(n_episodes):
            cash = sv
            holdings = 0

            s0 = self._row_to_state(feats.iloc[0])
            p0 = prices.iloc[0]
            a0 = self.learner.querysetstate(s0)
            trade, holdings = self._apply_action(a0, holdings)

            if trade != 0:
                eff = p0 * (1 + self.impact if trade > 0 else 1 - self.impact)
                cash -= trade * eff + self.commission

            v0 = cash + holdings * p0

            for t in range(1, n_days):
                pt = prices.iloc[t]
                v1 = cash + holdings * pt
                r = self._step_reward(v0, v1)

                s1 = self._row_to_state(feats.iloc[t])
                a1 = self.learner.query(s1, r)

                trade, holdings = self._apply_action(a1, holdings)
                if trade != 0:
                    eff = pt * (1 + self.impact if trade > 0 else 1 - self.impact)
                    cash -= trade * eff + self.commission

                v0 = cash + holdings * pt


    def testPolicy(self, symbol="IBM",
                   sd=dt.datetime(2009, 1, 1),
                   ed=dt.datetime(2010, 1, 1),
                   sv=100000):

        prices_df = self._get_prices(symbol, sd, ed)
        price = prices_df[symbol]

        feats = self._build_indicators(price)
        prices = price.reindex(feats.index).dropna()
        feats = feats.reindex(prices.index)

        if prices.empty:
            return pd.DataFrame(0, index=price.index, columns=["Trades"])

        idx = prices.index
        trades = pd.Series(0, index=idx, dtype=int)

        old_rar = self.learner.rar
        self.learner.rar = 0.0

        holdings = 0

        s0 = self._row_to_state(feats.iloc[0])
        a0 = self._best_action(s0)
        trade, holdings = self._apply_action(a0, holdings)
        trades.iloc[0] = trade

        for i in range(1, len(idx)):
            s = self._row_to_state(feats.iloc[i])
            a = self._best_action(s)
            trade, holdings = self._apply_action(a, holdings)
            trades.iloc[i] = trade

        self.learner.rar = old_rar

        full = pd.Series(0, index=price.index, dtype=int)
        full.loc[idx] = trades.values

        holdings_series = full.cumsum().clip(-self.MAX_HOLDINGS, self.MAX_HOLDINGS)
        full = holdings_series.diff().fillna(holdings_series).astype(int)
        full = full.clip(-2 * self.MAX_HOLDINGS, 2 * self.MAX_HOLDINGS)

        return pd.DataFrame(full, columns=["Trades"])


if __name__ == "__main__":
    print("StrategyLearner with 5 indicators")
