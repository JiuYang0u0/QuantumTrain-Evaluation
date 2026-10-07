import importlib.util
import json
from pathlib import Path
import torch


def test_full_runner_accepts_only_train_validation_and_preserves_frozen(tmp_path):
    from qt_eval.full_validation import run_condition
    from qt_eval.data import load_digits_splits
    from qt_eval.feature_controls import make_controls
    data=load_digits_splits(smoke=True)
    model=make_controls(11)['quantum_frozen']
    initial=model.q_params.detach().clone()
    result=run_condition(model,'quantum_frozen',11,
        {k:data[k] for k in ('train','validation')},tmp_path,epochs=2,smoke=True)
    assert result['status']=='completed'
    assert result['train_samples']==40 and result['validation_samples']==40
    assert result['frozen_parameter_delta']==0
    assert torch.equal(initial,model.q_params)
    assert 'test' not in result
    assert (tmp_path/'best.pt').exists()


def test_direct_and_generated_models_use_same_target_activation():
    from qt_eval.full_validation import models_for_seed
    models=models_for_seed(11)
    assert set(models)=={'direct_mlp','classical_trainable','quantum_trainable','quantum_frozen'}
    assert models['direct_mlp'].target_activation=='leaky_relu'
    assert sum(p.numel() for p in models['direct_mlp'].parameters())==610
