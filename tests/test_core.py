import numpy as np
import pytest
import torch


def test_splits_are_reproducible_disjoint_and_stratified():
    from qt_eval.data import load_digits_splits
    a, b = load_digits_splits(), load_digits_splits()
    sets = [set(a['indices'][k]) for k in ('train', 'validation', 'test')]
    assert len(set.union(*sets)) == 1797
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    assert a['indices'] == b['indices']
    small = load_digits_splits(smoke=True)
    for key in ('train', 'validation', 'test'):
        assert len(small[key]) == 40
        assert len(torch.unique(small[key].tensors[1])) == 10


def test_functional_model_matches_module_and_keeps_gradient():
    from qt_eval.models import ClassicalMLP, functional_mlp
    m = ClassicalMLP().double()
    x = torch.randn(3, 64, dtype=torch.float64)
    flat = torch.cat([p.flatten() for p in m.parameters()])
    assert flat.numel() == 610
    torch.testing.assert_close(m(x), functional_mlp(x, flat), atol=1e-9, rtol=1e-9)
    functional_mlp(x, flat).sum().backward()
    assert all(p.grad is not None for p in m.parameters())
    with pytest.raises(ValueError):
        functional_mlp(x, flat[:-1])


def test_qnode_gradient_matches_finite_difference_and_terminal_phase():
    from qt_eval.quantum import make_qnode, basis_codes
    circuit = make_qnode(2)
    theta = torch.tensor([[0.3, -0.7]], dtype=torch.float64, requires_grad=True)
    weights = torch.arange(4, dtype=torch.float64)
    p = circuit(theta)
    assert p.shape == (4,) and torch.all(p >= 0)
    assert float(p.sum().detach()) == pytest.approx(1)
    grad = torch.autograd.grad((p * weights).sum(), theta)[0]
    for i in range(2):
        plus, minus = theta.detach().clone(), theta.detach().clone()
        plus[0, i] += 1e-6
        minus[0, i] -= 1e-6
        numeric = ((circuit(plus)-circuit(minus))*weights).sum()/2e-6
        assert float(grad[0,i]) == pytest.approx(float(numeric), abs=1e-5, rel=1e-3)
    phase = torch.tensor([[0.3, -0.7], [0.9, 0.4]], dtype=torch.float64)
    torch.testing.assert_close(p, circuit(phase))
    assert basis_codes(2).tolist() == [[-1,-1],[-1,1],[1,-1],[1,1]]


def test_diagnostic_detects_detach_without_mutating_model_or_rng():
    from qt_eval.quantum import QuantumTrain
    from qt_eval.diagnostics import check_training_step
    torch.manual_seed(11)
    m = QuantumTrain(3)
    x = torch.randn(10,64,dtype=torch.float64)
    y = torch.arange(10)
    before = {k:v.clone() for k,v in m.state_dict().items()}
    rng = torch.get_rng_state().clone()
    good = check_training_step(m,x,y)
    assert good['status'] == 'passed'
    assert good['groups']['quantum']['max_delta'] > 0
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
    m.detach_quantum = True
    bad = check_training_step(m,x,y)
    assert bad['status'] == 'failed'
    assert not bad['groups']['quantum']['has_grad']
    assert bad['groups']['mapping']['has_grad']


@pytest.mark.parametrize('change', [{'epochs':0},{'shots':10},{'model':'unknown'},{'q_depth':2},{'learning_rate':float('nan')}])
def test_invalid_config(change):
    from qt_eval.config import validate_config
    with pytest.raises(ValueError):
        validate_config(change)


def test_wire_zero_is_most_significant_probability_bit():
    import pennylane as qml
    from qt_eval.quantum import basis_codes
    @qml.qnode(qml.device('default.qubit',wires=2))
    def known_state():
        qml.PauliX(0)
        return qml.probs(wires=[0,1])
    index=int(np.argmax(known_state()))
    assert index==2
    assert basis_codes(2)[index].tolist()==[1,-1]


def test_nonfinite_gradient_diagnostic_is_json_serializable():
    import json
    from qt_eval.diagnostics import check_training_step
    class BadBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x):
            return x.clone()
        @staticmethod
        def backward(ctx,gradient):
            return torch.full_like(gradient,float('nan'))
    class BadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.ones(64,10,dtype=torch.float64))
        def forward(self,x):
            return x@BadBackward.apply(self.weight)
    result=check_training_step(BadModel(),torch.ones(3,64,dtype=torch.float64),torch.tensor([0,1,2]))
    assert result['status']=='failed'
    json.dumps(result,allow_nan=False)
