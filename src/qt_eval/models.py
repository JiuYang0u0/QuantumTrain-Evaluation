"""The same 610-weight MLP for direct and generated-weight training."""
import torch
from torch import nn
from torch.nn import functional as F

TARGET_PARAMETERS = 610


class ClassicalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 8)
        self.fc2 = nn.Linear(8, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def functional_mlp(x, weights):
    if weights.ndim != 1 or weights.numel() != TARGET_PARAMETERS:
        raise ValueError('Expected a flat vector of 610 weights')
    hidden = F.relu(F.linear(x, weights[:512].reshape(8,64), weights[512:520]))
    return F.linear(hidden, weights[520:600].reshape(10,8), weights[600:610])


def make_model(config):
    if config['model'] == 'classical':
        return ClassicalMLP().double()
    from .quantum import QuantumTrain
    return QuantumTrain(config['q_depth'], config['q_delta'])
