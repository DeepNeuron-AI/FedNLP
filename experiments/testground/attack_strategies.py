import random
import multiprocessing

from base_attack_strategy import BaseCuriousServerStrategy
from omegaconf import DictConfig

from logging import INFO, DEBUG
from flwr.common.logger import log
from flwr.common import FitRes
from flwr.server.client_proxy import ClientProxy

from typing import List, Tuple

class AttackOneClient(BaseCuriousServerStrategy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_process = None
        self.attack_result_queue = None

    def get_shared_data(self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]) -> dict:
        """Randomly select one client to attack"""
        if (len(results) == 0) or (self.attack_process is not None):
            return None

        attacked_result = random.choice(results)

        return {
            "gradients": attacked_result[1].parameters,
            "buffers": None,
            "metadata": DictConfig({
                'num_data_points': attacked_result[1].num_examples,
                'labels': None,
                'local_hyperparams': None
            }
        )}

    def get_server_payload(self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]) -> dict:
        return {
            'parameters': self.server_params,
            'metadata': self.metadata
        }

    def signal_attack_node(self, server_payload: dict, shared_data: dict):
        if self.attack_process is not None:
            return

        def reconstruct_wrapper(queue: multiprocessing.Queue, *args, **kwargs):
            result = self.attack.reconstruct(*args, **kwargs)
            queue.put(result)

        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=reconstruct_wrapper, args=(queue, [server_payload], [shared_data]))
        self.attack_process = process
        self.attack_result_queue = queue
        log(INFO, 'start attack')
        process.start()
        return
