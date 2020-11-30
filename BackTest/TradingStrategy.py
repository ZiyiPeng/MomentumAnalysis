
import pandas as pd
from itertools import accumulate

import numpy as np

from BackTest.TradingRecord import TradingRecord


class TradingStrategy:
    def __init__(self, analyzer, init_value, period):
        """
        :param analyzer: Analyzer
        :param init_value: initial investment (float)
        :param start_date: date to start using this strategy (str)
        :param period: int
        """
        self.analyzer = analyzer
        self.initial_value = init_value
        self.period = period
        self.records = {}
        # TODO: clean this part
        t = list(analyzer.stocks.keys())[0]
        self.df = pd.DataFrame(index=analyzer.stocks[t].df.index, columns=['value', 'daily_return'])
        self.df = self.df.fillna(0)

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
        stocks_prices = {t:self.analyzer.get_stock(t).get_prices() for t in tickers}
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
            self.df['value'] += self.df[t + '_pos'] * stocks_prices[t]
        self.df['daily_return'] = np.log(self.df['value'].pct_change().fillna(0) + 1)

    def calc_total_return(self):
        return np.sum(self.df['daily_return'])

    def calc_annual_return(self):
        self.df['annual_return'] = self.df['daily_return'].rolling(252).sum()
        return np.mean(self.df['annual_return'])

    # TODO: add & implement methods for analyzation
    # 1) compare the strategy's daily return to SP500's
    # 2) plot portfolio's value on each day (on top of SP500's)
    # 3) plot portfolio's daily/annual return (on top of SP500's)
    # 4) check if the return beats SP500's using p-test