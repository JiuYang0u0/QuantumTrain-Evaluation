import torch
import pytest


def test_counts_and_export_are_exact_without_quantum_inference():
    from qt_eval.paper_mnist import TargetCNN,PaperQuantumTrain
    target=TargetCNN().double()
    model=PaperQuantumTrain()
    assert sum(p.numel() for p in target.parameters())==6690
    assert sum(p.numel() for p in model.mapping.parameters())==249
    assert model.theta.numel()==208
    assert sum(p.numel() for p in model.parameters())==457
    torch.manual_seed(11)
    x=torch.randn(2,1,28,28,dtype=torch.float64)
    expected=model(x).detach()
    exported=model.export_classical()
    def forbidden(*args):raise AssertionError('QNode invoked during classical inference')
    model.qnode=forbidden
    torch.testing.assert_close(exported(x),expected,atol=1e-10,rtol=1e-9)


def test_cnn_functional_matches_module_and_gradient_reaches_generator():
    from qt_eval.paper_mnist import TargetCNN,PaperQuantumTrain,functional_cnn
    torch.manual_seed(11)
    x=torch.randn(3,1,28,28,dtype=torch.float64)
    target=TargetCNN().double()
    flat=torch.cat([p.flatten() for p in target.parameters()])
    torch.testing.assert_close(target(x),functional_cnn(x,flat))
    model=PaperQuantumTrain()
    loss=torch.nn.functional.cross_entropy(model(x),torch.tensor([0,1,2]))
    loss.backward()
    assert model.theta.grad is not None and torch.isfinite(model.theta.grad).all()
    assert model.theta.grad.norm()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.mapping.parameters())


def test_ry_chain_value_and_gradient():
    from qt_eval.paper_mnist import ry_chain
    q=ry_chain(2)
    theta=torch.tensor([[.3,.7]],dtype=torch.float64,requires_grad=True)
    p=q(theta)
    expected=torch.tensor([.93879128*.88242109,.93879128*.11757891,
                           .06120872*.11757891,.06120872*.88242109],dtype=torch.float64)
    # Use analytic probabilities, independent of circuit simulator.
    a=torch.cos(theta.detach()[0,0]/2).square();b=torch.cos(theta.detach()[0,1]/2).square()
    torch.testing.assert_close(p,torch.stack([a*b,a*(1-b),(1-a)*(1-b),(1-a)*b]))
    coefficients=torch.arange(4,dtype=torch.float64)
    g=torch.autograd.grad((p*coefficients).sum(),theta)[0]
    for i in range(2):
        plus=theta.detach().clone();minus=plus.clone()
        plus[0,i]+=1e-6;minus[0,i]-=1e-6
        fd=((q(plus)-q(minus))*coefficients).sum()/2e-6
        assert float(g[0,i])==pytest.approx(float(fd),abs=1e-6,rel=1e-4)


def test_mapping_uses_first_tanh_only_and_raw_probabilities():
    from qt_eval.paper_mnist import PaperQuantumTrain
    m=PaperQuantumTrain()
    assert sum(isinstance(layer,torch.nn.Tanh) for layer in m.mapping)==1
    p=m.qnode(m.theta)
    features=m.mapping_inputs(p)
    torch.testing.assert_close(features[:,-1],p[:6690])
    assert set(features[:,:-1].detach().unique().tolist())=={0.,1.}
