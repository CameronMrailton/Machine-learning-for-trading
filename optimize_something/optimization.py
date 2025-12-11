""""""
"""MC1-P2: Optimize a portfolio.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	 	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	 	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 		  		  		    	 		 		   		 		  
or edited.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	 	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Student Name: Cameron Railton (replace with your name)  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT User ID: crailton3 (replace with your User ID)  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT ID: 904071082 (replace with your GT ID)  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt  		  	   		 	 	 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		 	 	 		  		  		    	 		 		   		 		  
from util import get_data, plot_data
from scipy.optimize import minimize

  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
# This is the function that will be tested by the autograder  		  	   		 	 	 		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality
def author():
    return "crailton3"  # Replace with your GT username

def study_group():
    return []
def optimize_portfolio(  		  	   		 	 	 		  		  		    	 		 		   		 		  
    sd=dt.datetime(2008, 6, 1),
    ed=dt.datetime(2009, 9, 1),
    syms=["IBM", "X", "GLD", "JPM"],
    gen_plot=True,
):  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		 	 	 		  		  		    	 		 		   		 		  
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		 	 	 		  		  		    	 		 		   		 		  
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		 	 	 		  		  		    	 		 		   		 		  
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		 	 	 		  		  		    	 		 		   		 		  
    statistics.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type ed: datetime  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		 	 	 		  		  		    	 		 		   		 		  
        symbol in the data directory)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type syms: list  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		 	 	 		  		  		    	 		 		   		 		  
        code with gen_plot = False.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type gen_plot: bool  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		 	 	 		  		  		    	 		 		   		 		  
        standard deviation of daily returns, and Sharpe ratio  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # Read in adjusted closing prices for given symbols, date range  		  	   		 	 	 		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		 	 	 		  		  		    	 		 		   		 		  
    prices = prices_all[syms]  # only portfolio symbols  		  	   		 	 	 		  		  		    	 		 		   		 		  
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # find the allocations for the optimal portfolio  		  	   		 	 	 		  		  		    	 		 		   		 		  

    normalized_prices = prices / prices.iloc[0]

    def neg_sharpe_ratio(allocations):
        weighted_prices = normalized_prices * allocations
        portfolio_value = weighted_prices.sum(axis=1)
        daily_returns = portfolio_value.pct_change().dropna()
        avg_daily_ret = daily_returns.mean()
        std_daily_ret = daily_returns.std()
        sharpe_ratio = (avg_daily_ret / std_daily_ret) * np.sqrt(252)
        return sharpe_ratio * -1


    num_stocks = len(syms)
    initial_allocation = np.ones(num_stocks) / num_stocks

    bounds = tuple((0, 1) for _ in range(num_stocks))
    constraints = {'type': 'eq', 'fun': lambda initial_allocation: np.sum(initial_allocation) - 1}

    result = minimize(neg_sharpe_ratio, initial_allocation,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    allocations = result.x

    weighted_prices = normalized_prices * allocations

    portfolio_value = weighted_prices.sum(axis=1)
    daily_returns = portfolio_value.pct_change().dropna()

    cr = (portfolio_value[-1] / portfolio_value[0]) - 1
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    sr = (adr / sddr) * np.sqrt(252)

    # Compare daily portfolio value with SPY using a normalized plot  		  	   		 	 	 		  		  		    	 		 		   		 		  
    if gen_plot:  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # add code to plot here  		  	   		 	 	 		  		  		    	 		 		   		 		  
        normalized_portfolio = portfolio_value / portfolio_value.iloc[0]
        normalized_SPY = prices_SPY / prices_SPY.iloc[0]

        df_plot = pd.concat([normalized_portfolio, normalized_SPY], axis=1)
        df_plot.columns = ["Portfolio", "SPY"]
        df_plot.plot(title="Daily Portfolio Value and SPY")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.savefig("images/plot.png")
        plt.close()
        pass  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    return allocations, cr, adr, sddr, sr
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def test_code():  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2009, 1, 1)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    end_date = dt.datetime(2010, 1, 1)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM"]  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # Assess the portfolio  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # allocations, cr, adr, sddr, sr = optimize_portfolio(
    #     sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    # )
    allocations, cr, adr, sddr, sr = optimize_portfolio(
         gen_plot=True
    )
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # Print statistics  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # print(f"Start Date: {start_date}")
    # print(f"End Date: {end_date}")
    # print(f"Symbols: {symbols}")
    # print(f"Allocations:{allocations}")
    # print(f"Sharpe Ratio: {sr}")
    # print(f"Volatility (stdev of daily returns): {sddr}")
    # print(f"Average Daily Return: {adr}")
    # print(f"Cumulative Return: {cr}")
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # This code WILL NOT be called by the auto grader  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # Do not assume that it will be called  		  	   		 	 	 		  		  		    	 		 		   		 		  
    test_code()  		  	   		 	 	 		  		  		    	 		 		   		 		  
