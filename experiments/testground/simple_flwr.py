# %%
import flwr
import torch

from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

from typing import Dict


# %% [markdown]
# # Standard pytorch, black-box to flower
# Simple text model according to: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

# %%
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

# %%
tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# %%
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

# %%
from torch.utils.data import DataLoader, Dataset
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def collate_batch(batch):
    text_list, offsets, label_list = [], [0], []
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

train_iter = list(AG_NEWS(split='train'))
valid_iter = list(AG_NEWS(split='test'))
train_dataloaders = [
    DataLoader(train_iter[i:i+len(train_iter)//10], batch_size=8, shuffle=False, collate_fn=collate_batch)
    for i in range(0, len(train_iter), len(train_iter)//10)
]
valid_dataloaders = [
    DataLoader(valid_iter[i:i+len(valid_iter)//10], batch_size=8, shuffle=False, collate_fn=collate_batch)
    for i in range(0, len(valid_iter), len(valid_iter)//10)
]

# %%
from torch import nn

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# %%
import time

def train(dataloader: DataLoader, model: nn.Module, epochs: int):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=1.0)
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for epoch in range(epochs):
        for idx, (label, text, offsets) in enumerate(dataloader):
            optimizer.zero_grad()
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                    '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                                total_acc/total_count))
                total_acc, total_count = 0, 0
                start_time = time.time()

def evaluate(dataloader: DataLoader, model: nn.Module):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

# %% [markdown]
# # Test centralised

# %%
trainloader = train_dataloaders[0]
valloader = valid_dataloaders[0]

vocab_size = len(vocab)
emsize = 64
num_class = len(set([label for (label, text) in train_iter]))
net = TextClassificationModel(vocab_size=vocab_size, embed_dim=emsize, num_class=num_class).to(device)

for epoch in range(5):
    train(trainloader, net, epochs=1)
    accuracy = evaluate(valloader, net)
    print(f"Epoch {epoch+1}: validation accuracy {accuracy}")

accuracy = evaluate(valloader, net)
print(f"Final test set performance:\n\taccuracy {accuracy}")


# %% [markdown]
# # Federated Learning wrapper
# Where we modify the training and evaluation loop

# %%
from collections import OrderedDict

class FlowerClient(flwr.client.NumPyClient):
    def __init__(self, net: nn.Module, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
        train(self.trainloader, self.net,  epochs=1)
        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
        accuracy = evaluate(self.valloader, self.net)
        return 0, len(self.valloader), {"accuracy": float(accuracy)}

# %%
def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    # define stuff
    vocab_size = len(vocab)
    emsize = 64
    num_class = len(set([label for (label, text) in train_iter]))

    # Load model
    net = TextClassificationModel(
        vocab_size=vocab_size,
        embed_dim=emsize,
        num_class=num_class
    ).to(device)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = train_dataloaders[int(cid)]
    valloader = valid_dataloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader)

# %%
# Create FedAvg strategy
strategy = flwr.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_eval=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=10,  # Never sample less than 10 clients for training
        min_eval_clients=5,  # Never sample less than 5 clients for evaluation
        min_available_clients=10,  # Wait until all 10 clients are available
)

# Start simulation
flwr.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=len(train_dataloaders),
    num_rounds=5,
    strategy=strategy,
)

# %%



