import warnings
import breaching
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    i = 0
    threshold = 1
    for _ in range(epochs):
        for images, labels in tqdm(trainloader, total=threshold):
            i += 1
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            if i == threshold:
                break

def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    i = 0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            i += 1
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            if i == 2:
                break
    return loss / len(testloader.dataset), correct / total


def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=4, shuffle=True), DataLoader(testset)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
# options for improvement: use config to populate the same model on server, attack node, and clients instead of running breaching?
# instead of running the same piece of code
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
_, _, net, loss_fn = breaching.cases.construct_case(cfg.case, setup)
net : nn.Module = net.to(DEVICE)

trainloader, testloader = load_data()

# Define Flower client
class FlowerGradientClient(fl.client.NumPyClient):
    def get_parameters(self):
        """Return initial parameters"""
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return [v.grad.cpu().numpy() for v in net.parameters()], len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        # option for improvement: does this return batch size or size of dataset
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client("[::]:8080", client=FlowerGradientClient())
