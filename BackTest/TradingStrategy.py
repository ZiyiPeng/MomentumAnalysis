import pandas as pd
class TradingStrategy:
    def __init__(self, analyzer, init_value, ranking_period):
        """
        :param analyzer: Analyzer
        :param init_value: initial investment (float)
        :param start_date: date to start using this strategy (str)
        :param ranking_period: int
        """
        self.analyzer = analyzer
        self.initial_value = init_value
        self.period = ranking_period
        self.records = []
        # index: date, columns: value, daily_return
        self.df = pd.DataFrame()

    # apply this trading strategy and fill out trading_records
    # re-adjust positions at the end of each holding period
    def apply(self, start_date):
        pass

    # fill out path of this portfolio's historical value, store it as a column in self.df
    # after this method is called, self.df should contain 2 columns 'value' & 'daily_return'
    def backtrace(self):
        pass

    def calc_total_return(self):
        pass

    def calc_annual_return(self):
        pass