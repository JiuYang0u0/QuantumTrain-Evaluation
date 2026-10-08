import torch
from torch.utils.data import TensorDataset


def test_exact_updates_full_batches_and_fixed_evaluation_schedule(tmp_path):
    from qt_eval.update_budget import train_updates
    torch.manual_seed(11)
    model=torch.nn.Linear(3,2).double()
    data=TensorDataset(torch.randn(10,3,dtype=torch.float64),torch.arange(10)%2)
    r=train_updates(model,{'train':data,'validation':data},tmp_path,updates=5,batch_size=4,eval_every=2,milestones=[2,5])
    assert r['updates']==5 and r['examples_seen']==20
    assert [x['update'] for x in r['history']]==[2,4,5]
    assert r['unique_examples_seen']<=10
    assert (tmp_path/'best.pt').exists()
    assert (tmp_path/'update-2.pt').exists() and (tmp_path/'update-5.pt').exists()
