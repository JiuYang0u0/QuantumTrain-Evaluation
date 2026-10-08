"""Paper-guided MNIST reconstruction; explicit assumptions in docs, not exact reproduction."""
import math
import torch
from torch import nn
from torch.nn import functional as F
import pennylane as qml

LAYOUT=[('conv1.weight',(8,1,5,5)),('conv1.bias',(8,)),
        ('conv2.weight',(12,8,5,5)),('conv2.bias',(12,)),
        ('fc1.weight',(20,192)),('fc1.bias',(20,)),('fc2.weight',(10,20)),('fc2.bias',(10,))]
TARGET_PARAMETERS=sum(math.prod(shape) for _,shape in LAYOUT)


def unpack(weights):
    if weights.ndim!=1 or weights.numel()!=TARGET_PARAMETERS:
        raise ValueError('Expected 6690 generated weights')
    result={};start=0
    for name,shape in LAYOUT:
        size=math.prod(shape);result[name]=weights[start:start+size].reshape(shape);start+=size
    return result


def functional_cnn(x,weights):
    w=unpack(weights)
    x=F.max_pool2d(F.conv2d(x,w['conv1.weight'],w['conv1.bias']),2)
    x=F.max_pool2d(F.conv2d(x,w['conv2.weight'],w['conv2.bias']),2)
    x=F.linear(x.flatten(1),w['fc1.weight'],w['fc1.bias'])
    return F.linear(x,w['fc2.weight'],w['fc2.bias'])


class TargetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,8,5)
        self.conv2=nn.Conv2d(8,12,5)
        self.fc1=nn.Linear(192,20)
        self.fc2=nn.Linear(20,10)

    def forward(self,x):
        x=F.max_pool2d(self.conv1(x),2)
        x=F.max_pool2d(self.conv2(x),2)
        return self.fc2(self.fc1(x.flatten(1)))


def ry_chain(n_qubits):
    @qml.qnode(qml.device('default.qubit',wires=n_qubits),interface='torch',diff_method='backprop')
    def circuit(theta):
        for layer in range(theta.shape[0]):
            for wire in range(n_qubits):qml.RY(theta[layer,wire],wires=wire)
            for wire in range(n_qubits-1):qml.CNOT(wires=[wire,wire+1])
        return qml.probs(wires=range(n_qubits))
    return circuit


class PaperQuantumTrain(nn.Module):
    def __init__(self,blocks=16,global_center=False):
        super().__init__()
        if type(global_center)is not bool:raise ValueError('global_center must be bool')
        self.global_center=global_center
        if type(blocks)is not int or blocks<1:raise ValueError('Positive blocks required')
        # Uniform angles are a declared reconstruction assumption, not recovered author seeds.
        self.theta=nn.Parameter(torch.empty(blocks,13,dtype=torch.float64).uniform_(-math.pi,math.pi))
        self.mapping=nn.Sequential(nn.Linear(14,4,dtype=torch.float64),nn.Tanh(),
            nn.Linear(4,20,dtype=torch.float64),nn.Linear(20,4,dtype=torch.float64),nn.Linear(4,1,dtype=torch.float64))
        indices=torch.arange(6690)[:,None]
        self.register_buffer('codes',((indices>>torch.arange(12,-1,-1))&1).double())
        self.qnode=ry_chain(13)

    def mapping_inputs(self,probabilities):
        return torch.cat([self.codes,probabilities[:6690,None]],dim=1)

    def generated_weights(self):
        weights=self.mapping(self.mapping_inputs(self.qnode(self.theta))).flatten()
        return weights-weights.mean() if self.global_center else weights

    def forward(self,x):return functional_cnn(x,self.generated_weights())

    def export_classical(self):
        with torch.random.fork_rng(devices=[]):
            model=TargetCNN().double()
        model.load_state_dict({k:v.detach().clone() for k,v in unpack(self.generated_weights()).items()})
        return model.eval()
