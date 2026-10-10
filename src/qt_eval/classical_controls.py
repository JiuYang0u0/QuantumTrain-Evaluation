"""Classical feature-source control for the same r3 compression setup.

Swap only the quantum circuit for a fixed random projection + softmax so the
trainable budget (latent + shared mapping) matches QT16/QT106 exactly.
"""
import math
import torch
from torch import nn
from .paper_mnist import TARGET_PARAMETERS, unpack, functional_cnn, TargetCNN

N_QUBITS = 13
PROB_DIM = 2 ** N_QUBITS
MAPPING_WIDTHS = [14, 4, 20, 4, 1]


def build_mapping():
    """Same mapping as PaperQuantumTrain: one Tanh after the first linear."""
    return nn.Sequential(
        nn.Linear(14, 4, dtype=torch.float64), nn.Tanh(),
        nn.Linear(4, 20, dtype=torch.float64),
        nn.Linear(20, 4, dtype=torch.float64),
        nn.Linear(4, 1, dtype=torch.float64),
    )


def bit_codes():
    indices = torch.arange(TARGET_PARAMETERS)[:, None]
    return ((indices >> torch.arange(12, -1, -1)) & 1).double()


class ClassicalRandomFeatureTrain(nn.Module):
    def __init__(self, latent_size, projection_seed=2026, global_center=True,
                 probability_processing='upstream'):
        super().__init__()
        if type(latent_size) is not int or latent_size < 1:
            raise ValueError('latent_size must be a positive integer')
        if probability_processing not in ('raw', 'upstream'):
            raise ValueError('Unknown probability processing')
        if type(global_center) is not bool:
            raise ValueError('global_center must be bool')
        self.latent_size = latent_size
        self.global_center = global_center
        self.probability_processing = probability_processing
        # Match PaperQuantumTrain theta init: uniform(-pi, pi) over the same count.
        self.z = nn.Parameter(torch.empty(latent_size, dtype=torch.float64).uniform_(-math.pi, math.pi))
        self.mapping = build_mapping()
        self.register_buffer('codes', bit_codes())
        generator = torch.Generator().manual_seed(projection_seed)
        projection = torch.randn(PROB_DIM, latent_size, generator=generator, dtype=torch.float64)
        self.register_buffer('projection', projection / math.sqrt(latent_size))

    def probabilities(self):
        return torch.softmax(self.projection @ self.z, dim=0)

    def mapping_inputs(self, probabilities):
        feature = probabilities[:TARGET_PARAMETERS]
        if self.probability_processing == 'upstream':
            feature = (0.8 * torch.tanh(0.1 * 4096 * feature)).pow(0.3)
            feature = feature - feature.mean()
        return torch.cat([self.codes, feature[:, None]], dim=1)

    def feature_values(self):
        return self.mapping_inputs(self.probabilities())[:, -1]

    def generated_weights(self):
        weights = self.mapping(self.mapping_inputs(self.probabilities())).flatten()
        return weights - weights.mean() if self.global_center else weights

    def forward(self, x):
        return functional_cnn(x, self.generated_weights())

    def export_classical(self):
        with torch.random.fork_rng(devices=[]):
            model = TargetCNN().double()
        model.load_state_dict({k: v.detach().clone() for k, v in unpack(self.generated_weights()).items()})
        return model.eval()
