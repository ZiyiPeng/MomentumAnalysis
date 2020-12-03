import unittest
from Analyzer import Analyzer
from Stock import Stock
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import helper
from unittest.mock import MagicMock
from get_all_tickers import get_tickers as gt

class UnitTest(unittest.TestCase):
    
    def test_date_before_start_keyerror_should_appear(self):
        start = "2010-01-01"
        date = '2009-05-28'
        tickers = list(gt.get_biggest_n_tickers(40))
        b = Analyzer(tickers, start)
        self.assertRaises(KeyError, lambda: b.winners(date, 25, 5))
        
    def test_date_before_oneperiod(self):
        start = "2010-01-01"
        date = '2010-01-28'
        tickers = list(gt.get_biggest_n_tickers(40))
        b = Analyzer(tickers, start)
        self.assertEqual(b.winners(date, 25, 5), [])
        
    def test_ticker_not_momentum_before_list(self):
        start_date = "2010-01-01"
        ranking_period = 25
        n= 5
        listeddate ="2010-06-28" 
        s ='TSLA'
        tickers = list(gt.get_biggest_n_tickers(40))
        b = Analyzer(tickers, start_date)
        self.assertEqual(b.appeartimes( start_date, ranking_period, n, listeddate, s, volume_filter=False), 0,0)

if __name__ == '__main__':
    unittest.main()
