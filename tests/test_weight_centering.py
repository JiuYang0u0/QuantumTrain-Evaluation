import torch


def test_centering_only_changes_weight_means_and_preserves_biases():
    from qt_eval.quantum import center_weight_blocks
    w=torch.randn(610,dtype=torch.float64,requires_grad=True)
    centered=center_weight_blocks(w)
    for start,end in [(0,512),(520,600)]:
        assert abs(float(centered[start:end].mean().detach()))<1e-12
        torch.testing.assert_close(centered[start:end].std(),w[start:end].std())
    torch.testing.assert_close(centered[512:520],w[512:520])
    torch.testing.assert_close(centered[600:],w[600:])
    (centered*torch.arange(610)).sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()


def test_centering_model_and_diagnostic_protocol(tmp_path):
    from qt_eval.quantum import QuantumTrain
    from qt_eval.training_diagnostics import run_diagnostic
    from qt_eval.data import load_digits_splits
    m=QuantumTrain(3,mapping_activation='tanh',weight_transform='center_weights')
    assert abs(float(m.generated_weights()[:512].mean().detach()))<1e-12
    data=load_digits_splits(smoke=True)
    d=run_diagnostic('tanh',11,{k:data[k] for k in ('train','validation')},tmp_path,
                     epochs=2,tiny=True,target_activation='leaky_relu',weight_transform='center_weights')
    assert d['weight_transform']=='center_weights'
    assert 'blocks' in d['initial']
