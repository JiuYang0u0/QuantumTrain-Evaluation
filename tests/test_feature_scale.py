import pytest
import torch


def test_feature_scale_changes_only_feature_not_initial_parameters():
    from qt_eval.quantum import QuantumTrain
    torch.manual_seed(11)
    a=QuantumTrain(3,mapping_activation='tanh',feature_scale=1.)
    torch.manual_seed(11)
    b=QuantumTrain(3,mapping_activation='tanh',feature_scale=1000.)
    for p,q in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(p,q)
    assert not torch.allclose(a.generated_weights(),b.generated_weights())
    b.generated_weights().square().sum().backward()
    assert torch.isfinite(b.q_params.grad).all()


def test_scale_is_applied_to_sensitivity_and_snapshot():
    from qt_eval.quantum import QuantumTrain
    from qt_eval.sensitivity import measure_sensitivity
    from qt_eval.training_diagnostics import snapshot
    torch.manual_seed(11)
    m=QuantumTrain(3,mapping_activation='tanh',feature_scale=1000.)
    x=torch.rand(20,64,dtype=torch.float64);y=torch.arange(20)%10
    s=measure_sensitivity(m,x,y)
    assert s['scaled_feature_std']==pytest.approx(s['feature_std']*1000)
    observation=snapshot(m,x)
    assert len(observation['mapping_tanh_saturation'])==3
    assert all(0<=p<=1 for p in observation['mapping_tanh_saturation'])


def test_scale_config_and_saved_diagnostic(tmp_path):
    from qt_eval.config import validate_config
    from qt_eval.data import load_digits_splits
    from qt_eval.training_diagnostics import run_diagnostic
    for scale in [0,-1,float('nan'),float('inf')]:
        with pytest.raises(ValueError):
            validate_config({'feature_scale':scale})
    data=load_digits_splits(smoke=True)
    d=run_diagnostic('tanh',11,{k:data[k] for k in ('train','validation')},tmp_path,
       epochs=2,tiny=True,target_activation='leaky_relu',weight_transform='center_weights',
       train_quantum=False,feature_scale=1000.)
    assert d['feature_scale']==1000.
    assert all(h['quantum_max_delta']==0 for h in d['history'])
