import torch


def test_frozen_quantum_stays_fixed_while_mapping_learns(tmp_path):
    from qt_eval.training_diagnostics import run_diagnostic
    from qt_eval.data import load_digits_splits
    data=load_digits_splits(smoke=True)
    d=run_diagnostic('tanh',11,{k:data[k] for k in ('train','validation')},tmp_path,
        epochs=2,tiny=True,target_activation='leaky_relu',weight_transform='center_weights',train_quantum=False)
    assert d['train_quantum'] is False
    assert all(h['quantum_grad_norm'] is None and h['quantum_max_delta']==0 for h in d['history'])
    assert any(h['mapping_max_delta']>0 for h in d['history'])


def test_sensitivity_preserves_model_and_rng():
    from qt_eval.sensitivity import measure_sensitivity
    from qt_eval.quantum import QuantumTrain
    torch.manual_seed(11)
    m=QuantumTrain(3,mapping_activation='tanh',target_activation='leaky_relu',weight_transform='center_weights')
    x=torch.rand(3,64,dtype=torch.float64);y=torch.tensor([0,1,2])
    before={k:v.clone() for k,v in m.state_dict().items()}
    rng=torch.get_rng_state().clone()
    result=measure_sensitivity(m,x,y)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
    assert result['perturbations'][0]['probability_l2_delta']>0
    assert result['perturbations'][0]['weight_l2_delta']>0
    assert result['quantum_loss_grad_norm']>0
