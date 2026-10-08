import pytest
import torch


def test_upstream_probability_formula_uses_p_not_p_squared():
    from qt_eval.paper_mnist import PaperQuantumTrain
    m=PaperQuantumTrain(global_center=True,probability_processing='upstream')
    p=torch.linspace(1e-8,.002,8192,dtype=torch.float64,requires_grad=True)
    f=m.mapping_inputs(p)[:,-1]
    expected=(.8*torch.tanh(.1*4096*p[:6690])).pow(.3)
    torch.testing.assert_close(f,expected-expected.mean())
    assert abs(float(f.mean().detach()))<1e-12
    (f*torch.arange(6690)).sum().backward()
    assert torch.isfinite(p.grad).all()


def test_r3_same_initial_parameters_gradients_and_export():
    from qt_eval.paper_mnist import PaperQuantumTrain
    torch.manual_seed(11);a=PaperQuantumTrain(global_center=True)
    torch.manual_seed(11);b=PaperQuantumTrain(global_center=True,probability_processing='upstream')
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q)
    assert sum(p.numel() for p in b.parameters())==457
    x=torch.randn(2,1,28,28,dtype=torch.float64)
    loss=torch.nn.functional.cross_entropy(b(x),torch.tensor([0,1]));loss.backward()
    assert b.theta.grad.norm()>0 and torch.isfinite(b.theta.grad).all()
    torch.testing.assert_close(b(x),b.export_classical()(x),atol=1e-10,rtol=1e-9)
    with pytest.raises(ValueError):PaperQuantumTrain(probability_processing='unknown')
