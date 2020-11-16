from BackTest.TradingStrategy import TradingStrategy

class SimpleMomentumStrategy(TradingStrategy):
    def __init__(self, analyzer, init_value, ranking_period):
        super.__init__(analyzer, init_value, ranking_period)

    # override apply method in parent class, fill out trading records
    def apply(self):
        pass
