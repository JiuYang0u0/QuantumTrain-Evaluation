"""Independent small-task implementation of the cited QuantumTrain method."""
import pennylane as qml
import torch
from torch import nn
from .models import TARGET_PARAMETERS, functional_mlp


def make_qnode(n_qubits):
    device = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(device, interface='torch', diff_method='backprop')
    def probabilities(theta):
        for wire in range(n_qubits):
            qml.Hadamard(wires=wire)
        for layer in range(theta.shape[0]):
            rotation = qml.RY if layer % 2 == 0 else qml.RZ
            for wire in range(n_qubits):
                rotation(theta[layer, wire], wires=wire)
            for wire in range(n_qubits-1):
                qml.CZ(wires=[wire, wire+1])
        return qml.probs(wires=range(n_qubits))
    return probabilities


def basis_codes(n_qubits):
    index = torch.arange(2**n_qubits).unsqueeze(1)
    shifts = torch.arange(n_qubits-1, -1, -1)
    return (((index >> shifts) & 1)*2-1).double()


def probability_features(p, n_qubits):
    transformed = (0.8*torch.tanh(0.1*2**(n_qubits-1)*p.square())).pow(0.3)
    return transformed - transformed.mean()


class QuantumTrain(nn.Module):
    def __init__(self, depth=3, delta=0.001, mapping_activation='affine'):
        super().__init__()
        if mapping_activation not in ('affine','tanh'):
            raise ValueError('Unsupported mapping activation')
        self.n_qubits = 10
        self.q_params = nn.Parameter(delta*torch.randn(depth,10,dtype=torch.float64))
        widths = [11,40,200,40,1]
        layers=[]
        for index,(a,b) in enumerate(zip(widths[:-1],widths[1:])):
            layers.append(nn.Linear(a,b,dtype=torch.float64))
            if mapping_activation=='tanh' and index < len(widths)-2:
                layers.append(nn.Tanh())
        self.mapping = nn.Sequential(*layers)
        self.register_buffer('codes', basis_codes(10)[:TARGET_PARAMETERS])
        self.qnode = make_qnode(10)
        self.detach_quantum = False

    def generated_weights(self):
        probabilities = self.qnode(self.q_params)[:TARGET_PARAMETERS]
        if self.detach_quantum:
            probabilities = probabilities.detach()
        feature = probability_features(probabilities, self.n_qubits)
        inputs = torch.cat([self.codes, feature[:,None]], dim=1)
        return self.mapping(inputs).flatten()

    def forward(self, x):
        return functional_mlp(x, self.generated_weights())
