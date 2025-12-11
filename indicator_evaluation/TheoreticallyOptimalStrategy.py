import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data

def author():
    return "crailton3"

def study_group():
    return []

def testPolicy(symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates)[[symbol]].dropna().rename(columns={symbol: "price"})
    trades = pd.DataFrame(index=prices.index, data={"Trades": 0.0})
    holdings = 0

    for i in range(len(prices) - 1):
        today = prices.index[i]
        tomorrow = prices.index[i + 1]
        if prices.at[tomorrow, "price"] > prices.at[today, "price"]:
            target = 1000
        elif prices.at[tomorrow, "price"] < prices.at[today, "price"]:
            target = -1000
        else:
            target = holdings
        trade = target - holdings
        if trade != 0:
            trades.at[today, "Trades"] = trade
            holdings = target

    if holdings != 0:
        trades.at[prices.index[-1], "Trades"] = -holdings

    return trades
