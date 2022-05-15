import flwr as fl

from flwr.server.strategy import FedAvg
from attack_strategy import CuriousServerStrategy

fed_strategy = FedAvg()
strategy = CuriousServerStrategy(attack=None, fed_strategy=fed_strategy)

# Start Flower server
fl.server.start_server(
    server_address="[::]:8080",
    config={"num_rounds": 1},
    strategy=strategy
)
