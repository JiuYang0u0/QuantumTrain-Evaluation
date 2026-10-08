import torch


def test_fit_observer_records_epochs_and_preserves_training(tmp_path):
    from qt_eval.experiment import fit
    from qt_eval.data import load_digits_splits
    from qt_eval.models import ClassicalMLP
    from qt_eval.config import validate_config
    data=load_digits_splits(smoke=True)
    splits={k:data[k] for k in ('train','validation')}
    config=validate_config({'model':'classical','epochs':2})
    seen=[]
    for name,observer in [('plain',None),('observed',lambda m,e:seen.append((e,all(p.grad is not None for p in m.parameters()))))]:
        folder=tmp_path/name;folder.mkdir()
        torch.manual_seed(11);model=ClassicalMLP().double()
        fit(model,splits,config,folder,epoch_observer=observer)
    a=torch.load(tmp_path/'plain/best.pt',weights_only=True)
    b=torch.load(tmp_path/'observed/best.pt',weights_only=True)
    assert seen==[(1,True),(2,True)]
    for key in a['model_state']:torch.testing.assert_close(a['model_state'][key],b['model_state'][key])
