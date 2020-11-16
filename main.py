from Analyzer import Analyzer
from get_all_tickers import get_tickers as gt

if __name__ == '__main__':
    tickers = gt.get_biggest_n_tickers(40)
    analyzer = Analyzer(tickers, start='2010-01-01')
    print(tickers)
