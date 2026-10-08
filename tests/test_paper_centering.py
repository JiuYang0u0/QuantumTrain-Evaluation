import torch


def test_global_center_is_exact_and_export_keeps_semantics():
    from qt_eval.paper_mnist import PaperQuantumTrain
    torch.manual_seed(11);raw=PaperQuantumTrain()
    torch.manual_seed(11);centered=PaperQuantumTrain(global_center=True)
    w=raw.generated_weights()
    torch.testing.assert_close(centered.generated_weights(),w-w.mean())
    x=torch.randn(2,1,28,28,dtype=torch.float64)
    torch.testing.assert_close(centered(x),centered.export_classical()(x))
    assert sum(p.numel() for p in centered.parameters())==457
    loss=torch.nn.functional.cross_entropy(centered(x),torch.tensor([0,1]))
    loss.backward()
    assert centered.theta.grad.norm()>0 and torch.isfinite(centered.theta.grad).all()
