"""Low-rank direct training: a standard classical compression baseline.

The target CNN weight matrices are represented by rank-r factors and trained
directly, with no quantum circuit and no weight-generating hypernetwork. This
tests whether the quantum generator's accuracy is special relative to a
classical low-rank compression of the same model.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .paper_mnist import TargetCNN, LAYOUT, unpack

# Weight matrix shapes, flattened as (out_features, in_features).
MATRICES = {
    'conv1.weight': (8, 25),    # (8, 1, 5, 5)
    'conv2.weight': (12, 200),  # (12, 8, 5, 5)
    'fc1.weight': (20, 192),
    'fc2.weight': (10, 20),
}
BIAS_SIZES = {'conv1.bias': 8, 'conv2.bias': 12, 'fc1.bias': 20, 'fc2.bias': 10}
BIAS_PARAMETERS = sum(BIAS_SIZES.values())  # 50
MAX_RANK = max(min(shape) for shape in MATRICES.values())  # 20


def effective_rank(rank, shape):
    return min(rank, min(shape))


def parameters_for_rank(rank):
    if type(rank) is not int or rank < 1 or rank > MAX_RANK:
        raise ValueError(f'rank must be an integer in 1..{MAX_RANK}')
    return sum(effective_rank(rank, MATRICES[n]) * (MATRICES[n][0] + MATRICES[n][1])
               for n in MATRICES) + BIAS_PARAMETERS


def _svd_init(weight, rank):
    """Balanced rank-r factors taken from the target model's own initialization."""
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    rank = min(rank, s.numel())
    root = s[:rank].sqrt()
    return (u[:, :rank] * root).contiguous(), (root[:, None] * vh[:rank]).contiguous()


class LowRankCNN(nn.Module):
    def __init__(self, rank, target=None):
        super().__init__()
        if type(rank) is not int or rank < 1 or rank > MAX_RANK:
            raise ValueError(f'rank must be an integer in 1..{MAX_RANK}')
        self.rank = rank
        if target is None:
            with torch.random.fork_rng(devices=[]):
                target = TargetCNN().double()
        state = {k: v.detach().double() for k, v in target.state_dict().items()}
        self.factors = nn.ParameterDict()
        self.effective_ranks = {}
        for name, (out, n_in) in MATRICES.items():
            weight = state[name].reshape(out, n_in)
            r = effective_rank(rank, (out, n_in))
            self.effective_ranks[name] = r
            u, v = _svd_init(weight, r)
            key = name.split('.')[0]
            self.factors[f'{key}_u'] = nn.Parameter(u)
            self.factors[f'{key}_v'] = nn.Parameter(v)
        self.biases = nn.ParameterDict({
            name.split('.')[0]: nn.Parameter(state[name].clone()) for name in BIAS_SIZES})

    def weight_tensor(self):
        pieces = []
        for name, _ in LAYOUT:
            if name in MATRICES:
                out, n_in = MATRICES[name]
                key = name.split('.')[0]
                pieces.append((self.factors[f'{key}_u'] @ self.factors[f'{key}_v']).reshape(out, n_in).flatten())
            else:
                pieces.append(self.biases[name.split('.')[0]].flatten())
        return torch.cat(pieces)

    def _forward(self, x, weights):
        w = unpack(weights)
        x = F.max_pool2d(F.conv2d(x, w['conv1.weight'], w['conv1.bias']), 2)
        x = F.max_pool2d(F.conv2d(x, w['conv2.weight'], w['conv2.bias']), 2)
        x = F.linear(x.flatten(1), w['fc1.weight'], w['fc1.bias'])
        return F.linear(x, w['fc2.weight'], w['fc2.bias'])

    def forward(self, x):
        return self._forward(x, self.weight_tensor())

    def export_classical(self):
        with torch.random.fork_rng(devices=[]):
            model = TargetCNN().double()
        model.load_state_dict(unpack(self.weight_tensor().detach()))
        return model.eval()
