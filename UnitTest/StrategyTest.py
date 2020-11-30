import unittest
from datetime import datetime, timedelta

from Analyzer import Analyzer
from BackTest.TradingRecord import TradingRecord
from BackTest.TradingStrategy import TradingStrategy


class UnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tickers = ['AAPL', 'MSFT']
        cls.analyzer = Analyzer(cls.tickers, start='2010-01-01')
        cls.ts = TradingStrategy(cls.analyzer, 0, 10)
        cls.aapl_long_3 = TradingRecord('AAPL', 3, '2010-01-05', cls.analyzer.stock_price('AAPL', '2010-01-05'))
        cls.aapl_short_3 = TradingRecord('AAPL', -3, '2010-01-15', cls.analyzer.stock_price('AAPL', '2010-01-15'))
        cls.msft_long_3 = TradingRecord('MSFT', 3, '2010-01-05', cls.analyzer.stock_price('MSFT', '2010-01-05'))
        cls.msft_short_3 = TradingRecord('AAPL', -3, '2010-01-15', cls.analyzer.stock_price('MSFT', '2010-01-15'))
        cls.records = {'AAPL':[cls.aapl_long_3, cls.aapl_short_3],
                       'MSFT':[cls.msft_long_3, cls.msft_short_3]}

# TODO: test all methods in class TradingStrategy
    def test_inital_value(self):
        """Strategy's total value (exclude cash) should be zero if no stock is traded"""
        self.assertEqual(self.ts.portfolio_value('2010-01-04'), 0)

    def test_trading_record(self):
        """trading_record() returns correct records and change in purchasing capacity"""
        start = datetime.strptime('2010-01-04', "%Y-%m-%d")
        tr = self.ts.trading_record('AAPL', start, 100, True)
        records = tr['records']
        dcap = tr['capacity_delta']
        self.assertEqual(len(records), 2)
        self.assertGreater(dcap, 0)

    def test_backtrace_position_on_trading_days(self):
        """portfolio's position in a stock should equal to 0 at the end of holding period"""
        self.ts.records = self.records
        self.ts.backtrace()
        start_pos = self.ts.df['AAPL_pos']['2010-01-05']
        end_pos = self.ts.df['AAPL_pos']['2010-01-15']
        self.assertEqual(start_pos, 3)
        self.assertEqual(end_pos, 0)

    def test_backtrace_position_between_trading_days(self):
        """portfolio's position in a stock should equal to initial position during holding period"""
        self.ts.records = self.records
        self.ts.backtrace()
        pos = self.ts.df['AAPL_pos']['2010-01-05':'2010-01-14']
        self.assertListEqual(pos.tolist(), [3.0]*len(pos))

    def test_backtrace_position_non_overlapping_records(self):
        """portfolio's position in a stock should be zero throughout the period in which that stock is not traded"""
        long = TradingRecord('AAPL', 3, '2010-01-20', self.analyzer.stock_price('AAPL', '2010-01-20'))
        short = TradingRecord('AAPL', -3, '2010-02-01', self.analyzer.stock_price('AAPL', '2010-02-01'))
        self.records['AAPL'].extend([long, short])
        self.ts.records = self.records
        self.ts.backtrace()
        pos = self.ts.df['AAPL_pos']['2010-01-15':'2010-01-19']
        self.assertListEqual(pos.tolist(), [0.0] * len(pos))



