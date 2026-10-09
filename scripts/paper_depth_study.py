"""QT106 vs preserved QT16 at the same dataset and update budget."""
import argparse
import json
import time
import uuid
from pathlib import Path
import torch
import pandas as pd
from qt_eval.paper_mnist import PaperQuantumTrain
from qt_eval.mnist_data import load_mnist_pilot
from qt_eval.update_budget import train_updates
from qt_eval.artifacts import write_json,environment
from qt_eval.config import validate_config

SOURCE=Path('results/paper-data-extended-d206b2e56a')


def run(output='results',smoke=False,resume_root=None):
    data=load_mnist_pilot('data/mnist',train_size=10000)
    if json.loads((SOURCE/'indices-10000.json').read_text())!=data['indices']:raise ValueError('Control data mismatch')
    protocol=dict(seeds=[11] if smoke else [11,22,33],blocks=106,updates=4 if smoke else 12000,
        eval_every=2 if smoke else 100,milestones=[4] if smoke else [4000,8000,12000],smoke=smoke,
        train_size=10000,validation_size=200,batch_size=128,lr=.0001,global_center=True,
        probability_processing='upstream',source_control=str(SOURCE),test_access=False,
        initialization='same RNG seed; different theta size consumes different draws; mapping NOT paired')
    root=Path(resume_root) if resume_root else Path(output)/('paper-depth-'+uuid.uuid4().hex[:10])
    if resume_root:
        if json.loads((root/'protocol.json').read_text())!=protocol:raise ValueError('Resume study mismatch')
    else:
        root.mkdir(parents=True);write_json(root/'protocol.json',protocol)
        write_json(root/'indices.json',data['indices'])
    write_json(root/'status.json',{'status':'running'})
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        for seed in protocol['seeds']:
            folder=root/f'qt106-seed{seed}';folder.mkdir(exist_ok=True)
            if (folder/'result.json').exists() and json.loads((folder/'result.json').read_text()).get('status')=='completed':continue
            print('QT106 seed',seed,flush=True)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model=PaperQuantumTrain(106,global_center=True,probability_processing='upstream')
                resume=(folder/'latest.pt').exists()
                write_json(folder/('resume-environment.json' if resume else 'environment.json'),environment(validate_config({'seed':seed})))
                write_json(folder/'config.json',protocol|{'seed':seed})
                result=train_updates(model,{k:data[k] for k in ['train','validation']},folder,updates=protocol['updates'],
                    eval_every=protocol['eval_every'],seed=seed,milestones=protocol['milestones'],resume=resume)
                with torch.no_grad():
                    exported=model.export_classical();x=data['train'].tensors[0][:16]
                    torch.testing.assert_close(model(x),exported(x),atol=1e-10,rtol=1e-9)
                    delta=float((model(x)-exported(x)).abs().max())
                torch.save(exported.state_dict(),folder/'classical-export.pt')
                write_json(folder/'result.json',dict(status='completed',seed=seed,blocks=106,parameters=1627,export_delta=delta,**result))
        write_json(root/'status.json',{'status':'completed'})
    except BaseException as e:
        import traceback
        (root/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write_json(root/'status.json',dict(status='failed',error=str(e)));raise
    finally:torch.set_num_threads(old)
    print(root,flush=True);return root


def benchmark(output):
    data=load_mnist_pilot('data/mnist',train_size=10000);x,y=data['train'].tensors;x,y=x[:128],y[:128]
    old=torch.get_num_threads();torch.set_num_threads(1);rows=[]
    try:
        for blocks in [16,106]:
            torch.manual_seed(11);m=PaperQuantumTrain(blocks,True,'upstream')
            opt=torch.optim.Adam(m.parameters(),lr=.0001);times=[]
            for step in range(4):
                t=time.perf_counter();opt.zero_grad(set_to_none=True)
                loss=torch.nn.functional.cross_entropy(m(x),y);loss.backward();opt.step()
                if not torch.isfinite(loss) or not torch.isfinite(m.theta.grad).all():raise RuntimeError('Non-finite pilot')
                if step:times.append(time.perf_counter()-t)
            rows.append(dict(blocks=blocks,parameters=sum(p.numel() for p in m.parameters()),
                step_seconds=times,median_seconds=float(pd.Series(times).median()),
                estimated_three_seed_train_hours=float(pd.Series(times).median())*12000*3/3600))
        write_json(Path(output),rows);print(rows)
    finally:torch.set_num_threads(old)


def report(root,output):
    root,output=Path(root),Path(output);protocol=json.loads((root/'protocol.json').read_text())
    if protocol['smoke']:raise ValueError('Smoke QT106 is not comparable to full historical controls')
    records=[];rows=[]
    for seed in protocol['seeds']:
        for label,folder,params in [('CNN',SOURCE/f'cnn-n10000-seed{seed}',6690),
                ('QT16',SOURCE/f'qt16_r3-n10000-seed{seed}',457),('QT106',root/f'qt106-seed{seed}',1627)]:
            r=json.loads((folder/'result.json').read_text())
            if r['status']!='completed' or r['updates']!=12000 or r['seed']!=seed or r['parameters']!=params:raise ValueError('Incomplete control/study')
            records.append(dict(label=label,source=str(folder),result=r,environment=json.loads((folder/'environment.json').read_text())))
            rows.append(dict(model=label,seed=seed,parameters=params,ratio=params/6690,best_update=r['best_update'],
                train_accuracy=r['best']['train']['accuracy'],val_accuracy=r['best']['validation']['accuracy'],val_loss=r['best']['validation']['loss'],
                final_val_accuracy=r['final']['validation']['accuracy'],train_seconds=r['train_seconds']))
    output.mkdir(parents=True,exist_ok=True)
    f=pd.DataFrame(rows);f.to_csv(output/'runs.csv',index=False)
    write_json(output/'manifest.json',dict(protocol=protocol,runs=records))
    a=f.groupby('model').agg(n=('seed','count'),parameters=('parameters','first'),ratio=('ratio','first'),
        val_mean=('val_accuracy','mean'),val_std=('val_accuracy','std'),val_loss=('val_loss','mean'))
    a.to_csv(output/'summary.csv')
    text='# MNIST壓縮深度對照\n\n'+a.to_markdown()+'\n\n'+f.to_markdown(index=False)
    text+='\n\nCNN/QT16引用既有完成控制，QT106為新run；同資料/updates/抽樣/評估協定，非同耗時或相同mapping初值。更多blocks同時增加参数與計算成本。Validation已用於開發，未使用official test；此為r3重建，不是論文精確復現。\n'
    (output/'study.md').write_text(text,encoding='utf-8');print(a.to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');p.add_argument('--resume-root')
    p.add_argument('--report');p.add_argument('--benchmark');p.add_argument('--output',default='results');a=p.parse_args()
    if a.benchmark:benchmark(a.benchmark)
    elif a.report:report(a.report,a.output)
    else:run(a.output,a.smoke,a.resume_root)
