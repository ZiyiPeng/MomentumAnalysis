from collections import defaultdict

import pandas as pd
from itertools import accumulate
from scipy import stats
import numpy as np
from BackTest.TradingRecord import TradingRecord
import matplotlib.pyplot as plt

class TradingStrategy:
    def __init__(self, analyzer, init_value, period, n=5):
        """
        :param analyzer: Analyzer
        :param init_value: initial investment (float)
        :param start_date: date to start using this strategy (str)
        :param period: int
        :param n: number of winners/losers
        """
        self.analyzer = analyzer
        self.initial_value = init_value
        self.period = period
        self.records = defaultdict(list)
        # TODO: clean this part
        t = list(analyzer.stocks.keys())[0]
        self.df = pd.DataFrame(index=analyzer.stocks[t].df.index, columns=['value', 'daily_return'])
        self.df = self.df.fillna(0)
        self.n = n

    def trading_record(self, ticker, date, total_value, is_long):
        """
        :param ticker: ticker (str)
        :param date: (timestamp)
        :param total_value: amount of money to invest in this stock (float)
        :param is_long: whether to long this stock or short it (bool)
        :return: a dict which contains trading records and change in purchasing capacity when the position is closed
        """
        end_date_idx = self.df.index.get_loc(date) + self.period
        end_date = self.df.index[end_date_idx]
        if ticker == 'cash':
            record1 = TradingRecord(ticker, total_value if is_long else -total_value, date, 1)
            record2 = TradingRecord(ticker, -total_value if is_long else total_value, end_date, 1)
            return {'records': [record1, record2], 'capacity_delta': 0}
        enter_price = self.analyzer.stock_price(ticker, date)
        exit_price = self.analyzer.stock_price(ticker, end_date)
        position = total_value / enter_price if is_long else -total_value / enter_price
        record1 = TradingRecord(ticker, position, date, enter_price)
        record2 = TradingRecord(ticker, -position, end_date, exit_price)
        capacity_delta = (exit_price - enter_price) * position
        return {'records': [record1, record2], 'capacity_delta': capacity_delta}

    # apply this trading strategy and fill out trading_records
    def apply(self, start_date):
        pass

    # return the portfolio's value on date (exclude cash)
    def portfolio_value(self, date):
        return self.df['value'][date]

    # fill out path of this portfolio's historical value, store it as a column in self.df
    # after this method is called, self.df should contain 2 columns 'value' & 'daily_return'
    def backtrace(self):
        tickers = self.records.keys()
        stocks_prices = {t:self.analyzer.get_stock(t).get_prices() for t in tickers if t != 'cash'}
        # calculate portfolio's value everyday
        for t in tickers:
            # determine accumulated position of the stock on every single day
            self.df[t+'_pos'] = np.NaN
            self.df[t+'_pos'].iloc[0] = 0
            dates = [r.date for r in self.records[t]]
            accumulated_pos = list(accumulate([r.position for r in self.records[t]]))
            for idx, d in enumerate(dates):
                self.df[t+'_pos'][d] = accumulated_pos[idx]
            self.df[t + '_pos'] = self.df[t+'_pos'].fillna(method='ffill')
            # update portfolio's value by including this stock's contribution to portfolio's value
            self.df['value'] += (self.df[t + '_pos'] * stocks_prices[t]).fillna(0) if t != 'cash' else self.df[t + '_pos']
        self.df['daily_return'] = np.log(self.df['value'].pct_change().fillna(0) + 1).replace([np.inf, -np.inf], np.nan)
        self.df['annual_return'] = self.df['daily_return'].rolling(252).sum()

    def calc_total_return(self):
        return np.sum(self.df['daily_return'])

    # plot the value of this portfolio on top of benchmark
    def plot_values(self, benchmark):
        """
        :param benchmark: a populated stock instance (SPY)
        """
        df = pd.DataFrame(columns=['port_value', 'bench_value'], index=benchmark.df.index)
        benchmark_value = benchmark.df['daily_return'].cumsum(skipna=True)
        benchmark_value = self.initial_value * np.exp(benchmark_value)
        df['bench_value'] = benchmark_value
        df['port_value'] = self.df['value']
        df = df.replace([0, np.inf, -np.inf], np.nan).dropna()
        df.plot()
        plt.title("Strategy's Value vs Benchmark's")
        plt.show()

    #plot daily return of this portfolio on top of benchmark's daily return
    def plot_daily_return(self, benchmark):
        """
        :param benchmark: a populated stock instance (SPY)
        """
        df = pd.DataFrame(columns=['portfolio_daily_return', 'benchmark_daily_return'], index=benchmark.df.index)
        df['benchmark_daily_return'] = benchmark.df['daily_return']
        df['portfolio_daily_return'] = self.df['daily_return']
        df = df.replace([0, np.inf, -np.inf], np.nan).dropna()
        df.plot()
        plt.title("Strategy's Daily Return vs Benchmark's")
        plt.show()

    def plot_daily_return_density(self, benchmark):
        """
        :param benchmark: a populated stock instance (SPY)
        """
        df = pd.DataFrame(columns=['portfolio_daily_return', 'benchmark_daily_return'], index=benchmark.df.index)
        df['benchmark_daily_return'] = benchmark.df['daily_return']
        df['portfolio_daily_return'] = self.df['daily_return']
        df = df.replace([0, np.inf, -np.inf], np.nan).dropna()
        df.plot(kind='density')
        plt.title("Strategy's Daily Return Density vs Benchmark's")
        plt.show()

    #plot annual return of this portfolio on top of benchmark's annual return
    def plot_annual_return(self, benchmark):
        df = pd.DataFrame(columns=['portfolio_annual_return', 'benchmark_annual_return'], index=benchmark.df.index)
        df['benchmark_annual_return'] = benchmark.df['daily_return'].rolling(252).sum()
        df['portfolio_annual_return'] = self.df['annual_return']
        df = df.replace([0, np.inf, -np.inf], np.nan).dropna()
        df.plot()
        plt.title("Strategy's Annual Return vs Benchmark's")
        plt.show()

    def plot_annual_return_density(self, benchmark):
        """
        :param benchmark: a populated stock instance (SPY)
        """
        df = pd.DataFrame(columns=['benchmark_annual_return', 'portfolio_annual_return'], index=benchmark.df.index)
        df['benchmark_annual_return'] = benchmark.df['daily_return'].rolling(252).sum()
        df['portfolio_annual_return'] = self.df['annual_return']
        df = df.replace([0, np.inf, -np.inf], np.nan).dropna()
        df.plot(kind='density')
        plt.title("Strategy's Annual Return Density vs Benchmark's")
        plt.show()

    def t_test(self, benchmark):
        df = pd.DataFrame(columns=['port_return', 'bench_return'], index=benchmark.df.index)
        df['port_return'] = self.df['daily_return'][self.period+1:]
        df['bench_return'] = benchmark.df['daily_return']
        df = df.replace([0, np.inf, -np.inf], np.nan).dropna()
        return stats.ttest_rel(df['port_return'], df['bench_return'], nan_policy='omit').pvalue

    def average_daily_return(self):
        return self.df['daily_return'].mean(skipna=True)

    def average_annual_return(self):
        return self.df['annual_return'].mean(skipna=True)

    def excess_return(self, ff_df):
        """
        :param ff_df: a dataframe which contains fama french 3 factors daily data
        """
        self.df['excess_return'] = self.df['daily_return'] - ff_df['Mkt-RF']

    # TODO: add & implement methods for analyzation
    # 1) compare the strategy's daily return to SP500's
    # 2) plot portfolio's value on each day (on top of SP500's)
    # 3) plot portfolio's daily/annual return (on top of SP500's)
    # 4) check if the return beats SP500's using p-test