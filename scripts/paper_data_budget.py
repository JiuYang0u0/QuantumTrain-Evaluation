"""Dataset-size comparison with exactly matched update/sample budgets."""
import argparse
import uuid
from pathlib import Path
import torch
from qt_eval.mnist_data import load_mnist_pilot
from qt_eval.paper_mnist import TargetCNN,PaperQuantumTrain
from qt_eval.update_budget import train_updates
from qt_eval.artifacts import write_json,environment
from qt_eval.config import validate_config


def run(output='results',smoke=False):
    root=Path(output)/('paper-data-budget-'+uuid.uuid4().hex[:10]);root.mkdir(parents=True)
    protocol=dict(smoke=smoke,seeds=[11] if smoke else [11,22,33],sizes=[1000,10000],
        conditions=['cnn','qt16_r3'],updates=4 if smoke else 4000,batch_size=128,
        eval_every=2 if smoke else 100,sampling='uniform with replacement',test_access=False,
        comparison='same initialization, LR1e-4, r3 architecture, total updates and 128 examples per update',status='running')
    write_json(root/'protocol.json',protocol)
    datasets={n:load_mnist_pilot('data/mnist',train_size=n) for n in protocol['sizes']}
    assert datasets[1000]['indices']['validation']==datasets[10000]['indices']['validation']
    overlap=len(set(datasets[1000]['indices']['train'])&set(datasets[10000]['indices']['train']))
    protocol['train_subset_overlap']=overlap
    protocol['nested_subsets']=overlap==1000
    write_json(root/'protocol.json',protocol)
    for n,d in datasets.items():write_json(root/f'indices-{n}.json',d['indices'])
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        for seed in protocol['seeds']:
            for n in protocol['sizes']:
                for condition in protocol['conditions']:
                    print(condition,n,seed,flush=True)
                    folder=root/f'{condition}-n{n}-seed{seed}';folder.mkdir()
                    write_json(folder/'result.json',{'status':'running'})
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(seed)
                        model=TargetCNN().double() if condition=='cnn' else PaperQuantumTrain(global_center=True,probability_processing='upstream')
                        write_json(folder/'config.json',protocol|{'seed':seed,'train_size':n,'condition':condition})
                        write_json(folder/'environment.json',environment(validate_config({'seed':seed})))
                        result=train_updates(model,{k:datasets[n][k] for k in ['train','validation']},folder,
                            updates=protocol['updates'],eval_every=protocol['eval_every'],seed=seed)
                        delta=None
                        if condition!='cnn':
                            exported=model.export_classical();x=datasets[n]['train'].tensors[0][:16]
                            with torch.no_grad():
                                torch.testing.assert_close(model(x),exported(x),atol=1e-10,rtol=1e-9)
                                delta=float((model(x)-exported(x)).abs().max())
                            torch.save(exported.state_dict(),folder/'classical-export.pt')
                        write_json(folder/'result.json',dict(status='completed',seed=seed,condition=condition,train_size=n,
                            parameters=sum(p.numel() for p in model.parameters()),export_delta=delta,**result))
        protocol['status']='completed';write_json(root/'protocol.json',protocol)
    except BaseException as e:
        import traceback
        (root/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        protocol.update(status='failed',error=str(e));write_json(root/'protocol.json',protocol);raise
    finally:torch.set_num_threads(old)
    print(root,flush=True);return root


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');p.add_argument('--output',default='results')
    a=p.parse_args();run(a.output,a.smoke)
