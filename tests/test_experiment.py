import json
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


def test_evaluation_weights_unequal_batches():
    from qt_eval.experiment import evaluate
    logits = torch.tensor([[4.,0.],[0.,1.],[2.,0.]], dtype=torch.float64)
    targets = torch.tensor([0,0,1])
    result = evaluate(torch.nn.Identity(), DataLoader(TensorDataset(logits,targets),batch_size=2))
    assert result['loss'] == pytest.approx(float(torch.nn.functional.cross_entropy(logits,targets)))
    assert result['accuracy'] == pytest.approx(1/3)
    assert result['sample_count'] == 3


def test_fit_does_not_access_test_and_checkpoint_reloads(tmp_path):
    from qt_eval.experiment import fit, evaluate
    from qt_eval.config import validate_config
    from qt_eval.data import load_digits_splits
    from qt_eval.models import make_model
    c = validate_config({'model':'classical'})
    torch.manual_seed(c['seed'])
    m = make_model(c)
    splits = load_digits_splits(smoke=True)
    limited = {k:splits[k] for k in ('train','validation')}
    history = fit(m, limited, c, tmp_path)
    saved = torch.load(tmp_path/'best.pt',weights_only=True)
    assert saved['epoch'] == min(history, key=lambda row:row['val_loss'])['epoch']
    m.load_state_dict(saved['model_state'])
    other = make_model(c)
    other.load_state_dict(saved['model_state'])
    x = splits['test'].tensors[0]
    torch.testing.assert_close(m(x),other(x))


def test_smoke_runs_are_unique_complete_and_reproducible(tmp_path):
    from qt_eval.experiment import run_experiment
    from qt_eval.data import load_digits_splits
    from qt_eval.models import make_model
    from qt_eval.experiment import evaluate
    runs = [run_experiment({'model':'classical'}, tmp_path) for _ in range(2)]
    assert runs[0] != runs[1]
    results = [json.loads((p/'result.json').read_text()) for p in runs]
    assert all(r['status']=='completed' for r in results)
    assert results[0]['test'] == results[1]['test']
    for p in runs:
        for file in ('config.json','environment.json','split_indices.json','diagnostics.json','metrics.csv','best.pt'):
            assert (p/file).exists()
    c = json.loads((runs[0]/'config.json').read_text())
    m = make_model(c)
    m.load_state_dict(torch.load(runs[0]/'best.pt',weights_only=True)['model_state'])
    test = load_digits_splits(smoke=True)['test']
    assert evaluate(m,DataLoader(test,batch_size=64)) == results[0]['test']


def test_failure_is_preserved_not_silently_completed(tmp_path,monkeypatch):
    import qt_eval.experiment as module
    class NonFinite(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(float('nan')))
        def forward(self,x):
            return torch.ones(len(x),10,dtype=x.dtype)*self.weight
    monkeypatch.setattr(module,'make_model',lambda c:NonFinite())
    with pytest.raises(RuntimeError):
        module.run_experiment({'model':'classical'},tmp_path)
    run = next(tmp_path.iterdir())
    assert json.loads((run/'result.json').read_text())['status']=='failed'
    assert 'non-finite' in (run/'error.txt').read_text()


def test_quantum_smoke_completes(tmp_path):
    from qt_eval.experiment import run_experiment
    p = run_experiment({'model':'quantum','q_depth':3},tmp_path)
    result = json.loads((p/'result.json').read_text())
    assert result['status']=='completed'
    assert result['parameters']['mapping']==16761
    assert result['parameters']['quantum']==30


def test_nonfinite_gradient_aborts_fit(tmp_path):
    from qt_eval.experiment import fit
    from qt_eval.models import ClassicalMLP
    from qt_eval.config import validate_config
    from qt_eval.data import load_digits_splits
    m=ClassicalMLP().double()
    next(m.parameters()).register_hook(lambda g:torch.full_like(g,float('nan')))
    with pytest.raises(RuntimeError,match='gradient'):
        fit(m,load_digits_splits(smoke=True),validate_config({'model':'classical'}),tmp_path)
