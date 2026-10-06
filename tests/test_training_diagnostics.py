import torch


def test_mapping_activation_preserves_initial_linear_weights():
    from qt_eval.quantum import QuantumTrain
    torch.manual_seed(11)
    a=QuantumTrain(3,mapping_activation='affine')
    torch.manual_seed(11)
    b=QuantumTrain(3,mapping_activation='tanh')
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    for p,q in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(p,q)
    assert not torch.allclose(a.generated_weights(),b.generated_weights())


def test_snapshot_preserves_rng_parameters_and_reports_shapes():
    from qt_eval.quantum import QuantumTrain
    from qt_eval.training_diagnostics import snapshot
    model=QuantumTrain(3)
    x=torch.rand(20,64,dtype=torch.float64)
    state=torch.get_rng_state().clone()
    before=[p.detach().clone() for p in model.parameters()]
    result=snapshot(model,x)
    assert sum(result['prediction_counts'])==20
    assert len(result['hidden_active_fraction'])==8
    assert 0<=result['relu_zero_fraction']<=1
    assert result['weights']['count']==610
    assert torch.equal(state,torch.get_rng_state())
    assert all(torch.equal(p,q) for p,q in zip(before,model.parameters()))


def test_diagnostic_run_never_uses_test_split(tmp_path):
    from qt_eval.training_diagnostics import run_diagnostic
    from qt_eval.data import load_digits_splits
    data=load_digits_splits(smoke=True)
    limited={k:data[k] for k in ('train','validation')}
    a=run_diagnostic('affine',11,limited,tmp_path,epochs=2,tiny=True)
    assert a['train_samples']==20
    assert 'test' not in a
    assert len(a['history'])==2
    assert a['history'][0]['quantum_grad_norm']>0
    assert a['history'][0]['quantum_max_delta']>0
    assert (tmp_path/'affine-seed11-tiny'/'diagnostic.json').exists()


def test_new_config_activation_is_validated():
    import pytest
    from qt_eval.config import validate_config
    assert validate_config({'mapping_activation':'tanh'})['mapping_activation']=='tanh'
    with pytest.raises(ValueError):
        validate_config({'mapping_activation':'relu'})
