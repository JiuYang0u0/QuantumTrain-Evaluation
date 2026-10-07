import torch
import pytest


def test_leaky_target_preserves_negative_gradient():
    from qt_eval.models import functional_mlp
    w=torch.zeros(610,dtype=torch.float64,requires_grad=True)
    with torch.no_grad():
        w[512:520]=-1
        w[520:600]=torch.arange(80,dtype=torch.float64)/80
    x=torch.ones(2,64,dtype=torch.float64)
    functional_mlp(x,w,target_activation='leaky_relu').sum().backward()
    assert w.grad[:512].abs().sum()>0
    w.grad=None
    functional_mlp(x,w,target_activation='relu').sum().backward()
    assert w.grad[:512].abs().sum()==0


def test_snapshot_matches_actual_target_activation():
    from qt_eval.quantum import QuantumTrain
    from qt_eval.training_diagnostics import snapshot
    torch.manual_seed(11)
    model=QuantumTrain(3,target_activation='leaky_relu')
    x=torch.rand(20,64,dtype=torch.float64)
    report=snapshot(model,x)
    logits=model(x).detach()
    assert report['logits']['mean']==pytest.approx(float(logits.mean()))
    assert report['hidden_zero_fraction']<1


def test_leaky_diagnostic_saves_protocol(tmp_path):
    from qt_eval.training_diagnostics import run_diagnostic
    from qt_eval.data import load_digits_splits
    import json
    data=load_digits_splits(smoke=True)
    result=run_diagnostic('affine',11,{k:data[k] for k in ['train','validation']},tmp_path,
                          epochs=2,tiny=True,target_activation='leaky_relu')
    config=json.loads(next(tmp_path.glob('*/config.json')).read_text())
    assert config['target_activation']=='leaky_relu'
    assert result['target_activation']=='leaky_relu'
