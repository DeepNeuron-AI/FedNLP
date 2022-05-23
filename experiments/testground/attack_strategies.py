import random

from base_attack_strategy import BaseCuriousServerStrategy

from flwr.server.strategy import Strategy
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from typing import List, Tuple

class AttackOneClient(BaseCuriousServerStrategy):

    def get_server_payload(self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]) -> dict:
        """Randomly select one client to attack"""
        if not results:
            return None, {}

        return [fit_res for _, fit_res in results][random.randint(0, len(random))]


