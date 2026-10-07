import math
import torch
import pytest


def test_fanin_sets_weight_scale_preserves_bias_and_gradient():
    from qt_eval.quantum import fanin_weight_blocks
    w=torch.randn(610,dtype=torch.float64,requires_grad=True)
    out=fanin_weight_blocks(w)
    for start,end,fan in [(0,512,64),(520,600,8)]:
        assert float(out[start:end].mean().detach())==pytest.approx(0,abs=1e-12)
        assert float(out[start:end].std(unbiased=False).detach())==pytest.approx(1/math.sqrt(3*fan))
    torch.testing.assert_close(out[512:520],w[512:520])
    torch.testing.assert_close(out[600:],w[600:])
    (out*torch.arange(610)).sum().backward()
    assert torch.isfinite(w.grad).all() and w.grad[:512].abs().sum()>0


def test_fanin_constant_block_has_finite_backward():
    from qt_eval.quantum import fanin_weight_blocks
    w=torch.ones(610,dtype=torch.float64,requires_grad=True)
    out=fanin_weight_blocks(w)
    assert torch.isfinite(out).all()
    out.square().sum().backward()
    assert torch.isfinite(w.grad).all()


def test_fanin_diagnostic_runs(tmp_path):
    from qt_eval.data import load_digits_splits
    from qt_eval.training_diagnostics import run_diagnostic
    data=load_digits_splits(smoke=True)
    d=run_diagnostic('tanh',11,{k:data[k] for k in ('train','validation')},tmp_path,
        epochs=2,tiny=True,target_activation='leaky_relu',weight_transform='center_fanin')
    assert d['weight_transform']=='center_fanin'
    assert d['initial']['blocks']['W1']['std']==pytest.approx(1/math.sqrt(192))
