import json
import pytest


def test_diagnostic_learning_rate_controls_optimizer_and_artifact(tmp_path,monkeypatch):
    import qt_eval.training_diagnostics as d
    from qt_eval.data import load_digits_splits
    original=d.torch.optim.Adam
    observed=[]
    def capture(params,**kwargs):
        observed.append(kwargs['lr'])
        return original(params,**kwargs)
    monkeypatch.setattr(d.torch.optim,'Adam',capture)
    data=load_digits_splits(smoke=True)
    result=d.run_diagnostic('tanh',11,{k:data[k] for k in ('train','validation')},tmp_path,
                            epochs=2,tiny=True,target_activation='leaky_relu',learning_rate=.0003)
    assert observed==[.0003]
    assert result['learning_rate']==.0003
    config=json.loads(next(tmp_path.glob('*/config.json')).read_text())
    assert config['learning_rate']==.0003


def test_invalid_rate_creates_no_run(tmp_path):
    from qt_eval.training_diagnostics import run_diagnostic
    with pytest.raises(ValueError):
        run_diagnostic('tanh',11,{},tmp_path,learning_rate=float('nan'))
    assert not list(tmp_path.iterdir())
