import torch
import flwr as fl
import breaching

from flwr.common import FitRes, Scalar, Parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from attack_strategies import AttackOneClient 
from typing import List, Tuple, Dict, Optional

class FedSGD(FedAvg):

    def __init__(self, *args, learning_rate: float=1e-5, **kwargs):
        super(self, FedSGD).__init__(*args, **kwargs)
        self.server_params = None
        self.learning_rate = learning_rate
    
    def aggregate_fit(self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_gradients = super().aggregate_fit(rnd, results, failures)
        with torch.no_grad():
            self.server_params = self.server_params - self.learning_rate * aggregated_gradients
        return self.server_params

# code copied from breaching example (Inverting Gradients - Optimization-based Attack - Large Batch CIFAR-100)
cfg = breaching.get_config(overrides=["case=6_large_batch_cifar"])

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark

cfg.case.data.partition="balanced" # 100 unique CIFAR-100 images
cfg.case.user.user_idx = 0
cfg.case.user.provide_labels = False
cfg.attack.label_strategy = "yin" # also works here, as labels are unique
cfg.attack.regularization.total_variation.scale = 5e-4 # Total variation regularization needs to be smaller on CIFAR-10:

# options for improvement: do not generate server and make clear where the shared data is, etc.
# ideally, we want to only construct the model and loss_fn etc, but the attack library doesn't have
# that modularity built-in
_, _, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg.attack, setup)
fed_strategy = FedSGD(learning_rate=1e-5)
strategy = AttackOneClient(attack=attacker, fed_strategy=fed_strategy, metadata=cfg.case.data)

# Start Flower server
fl.server.start_server(
    server_address="[::]:8080",
    config={"num_rounds": 1},
    strategy=strategy
)
