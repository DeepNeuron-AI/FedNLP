from torch import nn

import torch
import torch.optim as optim
import numpy as np

class LogisticRegression(nn.Module):

    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(100, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.fc(x))

net = LogisticRegression()

X, y = torch.Tensor(np.random.randn(10, 100)), torch.Tensor(np.random.randint(0, 2, (10,)))

loss_fn = nn.BCELoss()
optimiser = optim.SGD(net.parameters(), lr=0.01)
optimiser.zero_grad()
net.train()
output = net(X).flatten()
loss = loss_fn(output, y)
loss.backward()

print([v.grad for v in net.parameters()])



