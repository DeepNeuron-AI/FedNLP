from flwr.server.strategy import Strategy

class StrategyWrapper:

    def __init__(self, strategy: Strategy):
        self.strategy = strategy