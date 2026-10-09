import torch
from qt_eval.paper_mnist import PaperQuantumTrain


def test_qt106_count_gradient_export():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    m=PaperQuantumTrain(blocks=106,global_center=True,probability_processing='upstream')
    assert m.theta.numel()==1378
    assert sum(p.numel() for p in m.parameters())==1627
    x=torch.randn(2,1,28,28,dtype=torch.float64)
    loss=torch.nn.functional.cross_entropy(m(x),torch.tensor([0,1]));loss.backward()
    assert m.theta.grad.norm()>0 and torch.isfinite(m.theta.grad).all()
    with torch.no_grad():torch.testing.assert_close(m(x),m.export_classical()(x),atol=1e-10,rtol=1e-9)
