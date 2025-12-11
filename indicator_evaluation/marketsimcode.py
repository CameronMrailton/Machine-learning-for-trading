import pandas as pd
from util import get_data

def author():
    return "crailton3"

def study_group():
    return []

def compute_portvals_from_trades(trades, symbol, start_val=100000, commission=0.0, impact=0.0):
    trades = trades.copy()
    trades.columns = ["Trades"]

    prices = get_data([symbol], trades.index)[[symbol]].dropna().rename(columns={symbol: "price"})
    trades = trades.reindex(prices.index).fillna(0.0)

    exec_price = prices["price"].where(trades["Trades"] <= 0, prices["price"] * (1 + impact))
    exec_price = exec_price.where(trades["Trades"] >= 0, prices["price"] * (1 - impact))

    cash_flow = -(exec_price * trades["Trades"]) - commission * (trades["Trades"] != 0).astype(float)
    ledger = pd.DataFrame(index=prices.index, data={"Cash": cash_flow, "Shares": trades["Trades"]})
    ledger.iloc[0, ledger.columns.get_loc("Cash")] += float(start_val)

    holdings = ledger.cumsum()
    portvals = holdings["Cash"] + holdings["Shares"] * prices["price"]

    return pd.DataFrame(portvals, columns=["portval"])
