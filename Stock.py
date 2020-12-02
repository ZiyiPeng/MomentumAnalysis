from pandas import np

import helper
class Stock:

    def __init__(self, ticker):
        self.ticker = ticker
        self.df = None
        self.momentum = None
        self.returns = None

    # collect historical data for this stock and initialize df
    # df contains 3 columns, 'Adj Close','Volume','daily_return'
    def populate_df(self, start, end=None):
        """
        :param start: start date of the historical data (ex: '2010-01-01')
        :param end: end date of the historical data
        """
        self.df = helper.get_historical_data(self.ticker, start, end)
        self.df['daily_return'] = np.log(self.df['Adj Close'].pct_change() + 1)

    def get_prices(self):
        return self.df['Adj Close']

    # TODO: add other methods if necessary