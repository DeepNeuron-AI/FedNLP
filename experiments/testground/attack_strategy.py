from breaching.attacks.base_attack import _BaseAttacker

from flwr.server.strategy import Strategy
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional

class BaseCuriousServerStrategy(Strategy, ABC):
    """
    Acts like a normal Federated Learning strategy,
    except expose some of the clients' parameters during aggregation
    to a handler task
    """

    def __init__(self, attack: _BaseAttacker, fed_strategy: Strategy):
        self.attack = attack
        self.strategy = fed_strategy

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Implement a strategy for selecting and retrieving some of the client data
        """
        server_payload = self.get_server_payload(rnd, results, failures)
        shared_data = self.get_shared_data(rnd, results, failures)
        self.signal_attack_node(server_payload, shared_data)
        return self.strategy.aggregate_fit(rnd, results, failures)

    @abstractmethod
    def get_server_payload(
        self, 
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> dict:
        pass

    @abstractmethod
    def get_shared_data(
        self, 
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> dict:
        pass

    @abstractmethod
    def signal_attack_node(
        self,
        server_payload: dict, shared_data: dict
    ):
        pass

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        return self.strategy.configure_fit(rnd, parameters, client_manager)

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return self.strategy.configure_evaluate(rnd, parameters, client_manager)

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return self.strategy.aggregate_evaluate(rnd, results, failures)

    def evaluate(
        self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return self.strategy.evaluate(parameters)