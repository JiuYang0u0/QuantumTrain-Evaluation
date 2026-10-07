"""Scale-matched classical feature controls; shared mapping initialization."""
import copy
import torch
from torch import nn
from .quantum import QuantumTrain,center_weight_blocks
from .models import functional_mlp


class ClassicalFeature(nn.Module):
    def __init__(self,reference,basis,coefficients,gain,trainable=False,zero=False):
        super().__init__()
        self.mapping=copy.deepcopy(reference.mapping)
        self.register_buffer('codes',reference.codes.clone())
        self.register_buffer('basis',basis.clone())
        self.register_buffer('gain',gain.clone())
        self.coefficients=nn.Parameter(coefficients.clone(),requires_grad=trainable)
        self.zero=zero

    def feature_values(self):
        if self.zero:
            return torch.zeros(610,dtype=self.codes.dtype)
        return (self.basis@self.coefficients)*self.gain

    def generated_weights(self):
        features=torch.cat([self.codes,self.feature_values()[:,None]],dim=1)
        return center_weight_blocks(self.mapping(features).flatten())

    def forward(self,x):
        return functional_mlp(x,self.generated_weights(),'leaky_relu')


def make_controls(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        qt=QuantumTrain(3,mapping_activation='tanh',target_activation='leaky_relu',weight_transform='center_weights',feature_scale=1000.)
        frozen=copy.deepcopy(qt);frozen.q_params.requires_grad_(False)
        generator=torch.Generator().manual_seed(2026)
        basis=torch.randn(610,30,generator=generator,dtype=torch.float64)
        basis=basis-basis.mean(0,keepdim=True)
        basis=basis/basis.std(0,unbiased=False,keepdim=True)
        coeff=qt.q_params.detach().flatten().clone()
        raw=basis@coeff
        # Gain is fitted to initial quantum feature std, then held fixed throughout training.
        gain=qt.feature_values().detach().std(unbiased=False)/raw.std(unbiased=False)
        return dict(quantum_trainable=qt,quantum_frozen=frozen,
            zero=ClassicalFeature(qt,basis,coeff,gain,zero=True),
            random_fixed=ClassicalFeature(qt,basis,coeff,gain),
            classical_trainable=ClassicalFeature(qt,basis,coeff,gain,trainable=True))
