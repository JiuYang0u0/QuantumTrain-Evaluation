"""The same 610-weight MLP for direct and generated-weight training."""
import torch
from torch import nn
from torch.nn import functional as F

TARGET_PARAMETERS = 610


class ClassicalMLP(nn.Module):
    def __init__(self,target_activation='relu'):
        super().__init__()
        self.target_activation=target_activation
        self.fc1 = nn.Linear(64, 8)
        self.fc2 = nn.Linear(8, 10)

    def forward(self, x):
        return self.fc2(activate_hidden(self.fc1(x),self.target_activation))


def activate_hidden(x,kind='relu'):
    if kind=='relu':
        return F.relu(x)
    if kind=='leaky_relu':
        return F.leaky_relu(x,negative_slope=0.01)
    raise ValueError('Unsupported target activation')


def functional_mlp(x, weights,target_activation='relu'):
    if weights.ndim != 1 or weights.numel() != TARGET_PARAMETERS:
        raise ValueError('Expected a flat vector of 610 weights')
    hidden = activate_hidden(F.linear(x, weights[:512].reshape(8,64), weights[512:520]),target_activation)
    return F.linear(hidden, weights[520:600].reshape(10,8), weights[600:610])


def make_model(config):
    if config['model'] == 'classical':
        return ClassicalMLP(config.get('target_activation','relu')).double()
    from .quantum import QuantumTrain
    return QuantumTrain(config['q_depth'], config['q_delta'],config.get('mapping_activation','affine'),config.get('target_activation','relu'),config.get('weight_transform','none'),config.get('feature_scale',1.))
