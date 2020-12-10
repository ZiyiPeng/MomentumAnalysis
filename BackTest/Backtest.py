from Analyzer import Analyzer
from get_all_tickers import get_tickers as gt
import pandas_datareader.data as web
from scipy.stats import ttest_rel
from scipy.stats import linregress

from BackTest.MomentumVolumeStrategy import MomentumVolumeStrategy
from Stock import Stock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm
def rolling_sharpe_ratio(obj, factors):
    obj.df['excess_return'] = obj.df['daily_return'] - factors['Mkt-RF']
    std = (np.exp(obj.df['excess_return'])-1).rolling(252).std()
    mean = (np.exp(obj.df['excess_return'])-1).rolling(252).mean()
    sharpe = mean / std * np.sqrt(252)
    sharpe.plot()
    plt.show()

def rolling_sharpe_ratio(obj, bench, factors):
    if 'sharpe' not in bench.df.columns:
        sharpe_ratio(bench, factors)
    if 'sharpe' not in obj.df.columns:
        sharpe_ratio(obj, factors)
    obj.df['benchmark'] = bench.df['sharpe']
    obj.df[['sharpe', 'benchmark']].plot()
    plt.title("Rolling 12 Month Sharpe Ratio Portfolio's vs Benchmark's")
    plt.show()


def sharpe_ratio(obj, factors):
    obj.df['excess_return'] = obj.df['daily_return'] - factors['Mkt-RF']
    std = (np.exp(obj.df['excess_return']) - 1).rolling(252).std()
    mean = (np.exp(obj.df['excess_return']) - 1).rolling(252).mean()
    obj.df['sharpe'] = mean / std * np.sqrt(252)
    return obj.df['sharpe'].mean()

def sharpe_ratio_correlation(obj, bench):
    df = pd.DataFrame(columns=['port_sharpe', 'bench_sharpe'], index=obj.df.index)
    df['port_sharpe'] = obj.df['sharpe']
    df['bench_sharpe'] = bench.df['sharpe']
    df = df.dropna()
    x = df['bench_sharpe'].tolist()
    y = df['port_sharpe'].tolist()
    x = sm.add_constant(x)
    #regr = linregress(df['port_sharpe'], df['bench_sharpe'])
    regr = sm.OLS(y, x).fit()
    #print(regr.summary())
    return {'coef': regr.params[1], 'pvalue': regr.pvalues[1]}

def daily_return_correlation(obj, bench):
    df = pd.DataFrame(columns=['port_daily_return', 'bench_daily_return'], index=obj.df.index)
    df['port_daily_return'] = obj.df['daily_return']
    df['bench_daily_return'] = bench.df['daily_return']
    df=df.dropna()
    x = df['port_daily_return'].tolist()
    y = df['bench_daily_return'].tolist()
    x = sm.add_constant(x)
    # regr = linregress(df['port_sharpe'], df['bench_sharpe'])
    regr = sm.OLS(y, x).fit()
    #print(regr.summary())
    return {'coef': regr.params[1], 'pvalue': regr.pvalues[1]}


if __name__ == '__main__':
    start = '2010-01-04'
    # get daily risk-free rate from fama french
    factors = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start=start)[0]
    factors.index = pd.to_datetime(factors.index, format="%Y%m%d")
    factors = factors[['Mkt-RF']]
    factors['Mkt-RF'] = np.log(1+factors['Mkt-RF']/100)

    # use SPY as benchmark
    SPY = Stock('SPY')
    SPY.populate_df(start)
    SPY.df['annual_return'] = SPY.df['daily_return'].rolling(252).sum()
    SPY.df['excess_return'] = SPY.df['daily_return'] - factors['Mkt-RF']
    sharpe_ratio(SPY, factors)
    #rolling_sharpee_ratio(SPY, factors)

    tickers = gt.get_biggest_n_tickers(40)
    analyzer = Analyzer(tickers, start=start)
    initial_investment = 10000.0
    period = 10
    # list of strategy with different period
    lo_strt = []
    lo_period = [21, 21*3, 21*6, 21*9, 252]
    for p in lo_period:
        strt = MomentumVolumeStrategy(analyzer, initial_investment, p, 5)
        strt.apply(start)
        strt.backtrace()
        lo_strt.append(strt)
    lo_sharpes = [sharpe_ratio(strt, factors) for strt in lo_strt]
    plt.plot(['1M', '3M', '6M', '9M', '12M'], lo_sharpes)
    plt.title('Sharpe ratio vs period length')
    plt.show()
    max_idx = lo_sharpes.index(max(lo_sharpes))
    optimal_period = lo_period[max_idx]
    optimal_period = 21*3

    #fix period length, change number of winner/losers
    lo_strt_diff_n = []
    for n in range(2, 15):
        strt = MomentumVolumeStrategy(analyzer, initial_investment, optimal_period, n)
        strt.apply(start)
        strt.backtrace()
        lo_strt_diff_n.append(strt)
    lo_sharpes_diff_n = [sharpe_ratio(strt, factors) for strt in lo_strt_diff_n]
    plt.plot(list(range(2, 15)), lo_sharpes_diff_n)
    plt.title('Sharpe ratio vs Number of Winners/losers')
    plt.show()
    max_idx = lo_sharpes_diff_n.index(max(lo_sharpes_diff_n))
    optimal_n = list(range(2, 15))[max_idx]

    # analyze the optimal strategy
    #optimal_strt = lo_strt_diff_n[max_idx]
    optimal_strt = MomentumVolumeStrategy(analyzer, initial_investment, 63, 11)
    optimal_strt.apply(start)
    optimal_strt.backtrace()
    optimal_strt.plot_values(SPY)
    optimal_strt.plot_daily_return_density(SPY)
    optimal_strt.plot_annual_return(SPY)
    optimal_strt.plot_annual_return_density(SPY)
    print('reltest result (pvalue)', optimal_strt.t_test(SPY))
    print('strategy mean annual return:', optimal_strt.df['annual_return'].mean(skipna=True))
    print('SPY mean annual return:', SPY.df['annual_return'].mean(skipna=True))
    print('strategy mean daily return:', optimal_strt.df['daily_return'].mean(skipna=True))
    print('SPY mean daily return:', SPY.df['daily_return'].mean(skipna=True))
    rolling_sharpe_ratio(optimal_strt, SPY, factors)
    print('daily return correlation:', daily_return_correlation(optimal_strt, SPY))
    print('sharpe ratio correlation:', sharpe_ratio_correlation(optimal_strt, SPY))
    data = optimal_strt.df['annual_return'].dropna().tolist()
    interval = st.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=st.sem(data))
    print('95% confidence interval for portfolio annual return', interval)
    print('min_annual_return:', np.min(data))
    print('max_annual_return:', np.max(data))
    print('total_return:', optimal_strt.calc_total_return())
    print('sharpe ratio', optimal_strt.df['sharpe'].mean(), SPY.df['sharpe'].mean())
    print('a')
