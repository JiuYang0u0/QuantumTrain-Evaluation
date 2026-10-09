import copy
import pytest
import torch
from torch.utils.data import TensorDataset


def test_interrupted_resume_matches_uninterrupted_adam_and_sampling(tmp_path,monkeypatch):
    import qt_eval.update_budget as module
    torch.manual_seed(5)
    data=TensorDataset(torch.randn(13,3,dtype=torch.float64),torch.arange(13)%2)
    splits={'train':data,'validation':data}
    base=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Dropout(.2),torch.nn.Linear(5,2)).double()
    torch.manual_seed(19)
    reference=module.train_updates(copy.deepcopy(base),splits,tmp_path/'whole',updates=8,batch_size=4,eval_every=2)
    evaluate=module.evaluate
    calls=0
    def interrupt(*args):
        nonlocal calls
        calls+=1
        if calls==2:raise KeyboardInterrupt('simulated stop')
        return evaluate(*args)
    monkeypatch.setattr(module,'evaluate',interrupt)
    torch.manual_seed(19)
    with pytest.raises(KeyboardInterrupt):
        module.train_updates(copy.deepcopy(base),splits,tmp_path/'resume',updates=8,batch_size=4,eval_every=2)
    saved=torch.load(tmp_path/'resume/latest.pt',weights_only=True)
    assert saved['step']==2
    monkeypatch.setattr(module,'evaluate',evaluate)
    torch.manual_seed(999)
    resumed=module.train_updates(copy.deepcopy(base),splits,tmp_path/'resume',updates=8,batch_size=4,eval_every=2,resume=True)
    for key in ('history','final','best','best_update','unique_examples_seen'):
        assert resumed[key]==reference[key]
    a=torch.load(tmp_path/'whole/latest.pt',weights_only=True)
    b=torch.load(tmp_path/'resume/latest.pt',weights_only=True)
    for name in a['model_state']:assert torch.equal(a['model_state'][name],b['model_state'][name])
    assert torch.equal(a['sampler_rng'],b['sampler_rng'])
    for i,state in a['optimizer_state']['state'].items():
        for key,value in state.items():assert torch.equal(value,b['optimizer_state']['state'][i][key])


def test_resume_rejects_changed_data_and_budget_without_overwriting(tmp_path):
    from qt_eval.update_budget import train_updates
    d=TensorDataset(torch.ones(8,3,dtype=torch.float64),torch.arange(8)%2)
    m=torch.nn.Linear(3,2).double()
    train_updates(m,{'train':d,'validation':d},tmp_path,updates=2,batch_size=4,eval_every=1)
    before=(tmp_path/'latest.pt').read_bytes()
    with pytest.raises(ValueError,match='protocol'):
        train_updates(m,{'train':d,'validation':d},tmp_path,updates=3,batch_size=4,eval_every=1,resume=True)
    changed=TensorDataset(torch.zeros(8,3,dtype=torch.float64),d.tensors[1])
    with pytest.raises(ValueError,match='protocol'):
        train_updates(m,{'train':changed,'validation':d},tmp_path,updates=2,batch_size=4,eval_every=1,resume=True)
    assert (tmp_path/'latest.pt').read_bytes()==before
    with pytest.raises(FileExistsError):
        train_updates(m,{'train':d,'validation':d},tmp_path,updates=2,batch_size=4,eval_every=1)
