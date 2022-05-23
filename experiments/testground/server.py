import torch
import flwr as fl
import breaching

from flwr.server.strategy import FedAvg
from experiments.testground.attack_strategies import AttackOneClient 

model = torchvision.models.resnet152(pretrained=True)
model.eval()
loss_fn = torch.nn.CrossEntropyLoss()

setup = dict(device=torch.device("cpu"), dtype=torch.float)
cfg_attack = breaching.get_attack_config("invertinggradients")
attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg_attack, setup)
shared_data = {}

fed_strategy = FedAvg()
strategy = AttackOneClient(attack=None, fed_strategy=fed_strategy, shared_data=shared_data})

# Start Flower server
fl.server.start_server(
    server_address="[::]:8080",
    config={"num_rounds": 1},
    strategy=strategy
)
