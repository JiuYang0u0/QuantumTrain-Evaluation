"""Fixed r2, longer small-data pilot. No official test access."""
import argparse
import json
import uuid
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from qt_eval.paper_mnist import TargetCNN,PaperQuantumTrain
from qt_eval.mnist_data import load_mnist_pilot
from qt_eval.config import validate_config
from qt_eval.artifacts import write_json,environment
from qt_eval.experiment import fit,evaluate


def run(output='results',data_root='data/mnist',smoke=False,extended_r3=False):
    prefix='paper-r3-budget-' if extended_r3 else 'paper-r2-learning-'
    root=Path(output)/(prefix+uuid.uuid4().hex[:10]);root.mkdir(parents=True)
    qt_condition='qt16_r3' if extended_r3 else 'qt16_r2'
    protocol=dict(evidence='paper-guided r2 small-data learning pilot, not paper reproduction',
        seeds=[11] if smoke else [11,22,33],conditions=['classical_cnn',qt_condition],smoke=smoke,
        train_samples=1000,validation_samples=200,epochs=2 if smoke else (500 if extended_r3 else 50),batch_size=128,
        learning_rate=.0001,qt_parameters=457,target_parameters=6690,test_access=False,status='running',
        assumptions=['docs/paper-mnist-protocol.md','docs/paper-centering-protocol.md','docs/paper-r2-learning-protocol.md'])
    if extended_r3:
        protocol.update(evidence='paper-guided r3 fixed-data update-budget study, not paper reproduction',
            probability_processing='upstream',milestones=[2] if smoke else [50,100,200,500],
            updates_per_epoch=8,total_updates=16 if smoke else 4000)
        protocol['assumptions'].append('docs/paper-r3-budget-protocol.md')
    write_json(root/'protocol.json',protocol)
    data=load_mnist_pilot(data_root);splits={k:data[k] for k in ('train','validation')}
    write_json(root/'split_indices.json',data['indices']);write_json(root/'dataset_hashes.json',data['hashes'])
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        for seed in protocol['seeds']:
            for condition in protocol['conditions']:
                print(condition,seed,flush=True)
                folder=root/f'{condition}-{seed}';folder.mkdir()
                write_json(folder/'result.json',{'status':'running'})
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    model=TargetCNN().double() if condition=='classical_cnn' else PaperQuantumTrain(global_center=True,probability_processing='upstream' if extended_r3 else 'raw')
                    config=validate_config(dict(seed=seed,epochs=protocol['epochs'],batch_size=128,learning_rate=.0001))
                    write_json(folder/'config.json',protocol|{'seed':seed,'condition':condition})
                    write_json(folder/'environment.json',environment(config))
                    previous={k:p.detach().clone() for k,p in model.named_parameters()}
                    observations=[]
                    milestones=[]
                    def observe(m,epoch):
                        groups={}
                        for name,p in m.named_parameters():
                            group='quantum' if name=='theta' else ('mapping' if condition!= 'classical_cnn' else 'classical')
                            g=groups.setdefault(group,dict(last_batch_grad_norm_squared=0.,epoch_max_delta=0.))
                            if p.grad is not None:g['last_batch_grad_norm_squared']+=float(p.grad.norm())**2
                            g['epoch_max_delta']=max(g['epoch_max_delta'],float((p.detach()-previous[name]).abs().max()))
                            previous[name]=p.detach().clone()
                        for g in groups.values():g['last_batch_grad_norm']=g.pop('last_batch_grad_norm_squared')**.5
                        observations.append(dict(epoch=epoch,groups=groups))
                        write_json(folder/'updates.json',observations)
                        if extended_r3 and epoch in protocol['milestones']:
                            # DataLoader iteration consumes Torch RNG even without shuffle.
                            # Keep milestone observation independent from the training stream.
                            with torch.random.fork_rng(devices=[]):
                                metrics={k:evaluate(m,DataLoader(v,batch_size=128)) for k,v in splits.items()}
                            milestones.append(dict(epoch=epoch,updates=epoch*8,**metrics))
                            write_json(folder/'milestones.json',milestones)
                            torch.save(dict(model_state=m.state_dict(),epoch=epoch),folder/f'epoch-{epoch}.pt')
                            print(condition,seed,'epoch',epoch,'val',metrics['validation'],flush=True)
                    history=fit(model,splits,config,folder,epoch_observer=observe)
                    loaders={k:DataLoader(splits[k],batch_size=128) for k in splits}
                    final={k:evaluate(model,loader) for k,loader in loaders.items()}
                    checkpoint=torch.load(folder/'best.pt',weights_only=True);model.load_state_dict(checkpoint['model_state'])
                    best={k:evaluate(model,loader) for k,loader in loaders.items()}
                    export_delta=None
                    if condition!='classical_cnn':
                        exported=model.export_classical();x=splits['train'].tensors[0][:32]
                        with torch.no_grad():
                            torch.testing.assert_close(model(x),exported(x),atol=1e-10,rtol=1e-9)
                            export_delta=float((model(x)-exported(x)).abs().max())
                        torch.save(exported.state_dict(),folder/'classical-export.pt')
                    write_json(folder/'result.json',dict(status='completed',condition=condition,seed=seed,
                        parameters=sum(p.numel() for p in model.parameters()),parameter_ratio=sum(p.numel() for p in model.parameters())/6690,
                        best_epoch=checkpoint['epoch'],best=best,final=final,export_delta=export_delta,
                        train_seconds=sum(h['train_seconds'] for h in history)))
        protocol['status']='completed';write_json(root/'protocol.json',protocol)
    except BaseException as e:
        import traceback
        (root/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        protocol['status']='failed';protocol['error']=str(e);write_json(root/'protocol.json',protocol);raise
    finally:torch.set_num_threads(old)
    print(root,flush=True);return root


def report(root,output):
    import matplotlib.pyplot as plt
    root,output=Path(root),Path(output);p=json.loads((root/'protocol.json').read_text())
    records=[];rows=[]
    for seed in p['seeds']:
        for condition in p['conditions']:
            folder=root/f'{condition}-{seed}'
            record={k:json.loads((folder/f'{k}.json').read_text()) for k in ('config','environment','result','updates')}
            r=record['result'];h=pd.read_csv(folder/'metrics.csv')
            if r['status']!='completed' or r['condition']!=condition or r['seed']!=seed or len(h)!=p['epochs']:
                raise ValueError('Incomplete/inconsistent pilot')
            record['metrics']=h.to_dict('records');records.append(record)
            milestone_file=folder/'milestones.json'
            if p.get('probability_processing')=='upstream':
                record['milestones']=json.loads(milestone_file.read_text())
                if [m['epoch'] for m in record['milestones']]!=p['milestones']:
                    raise ValueError('Missing milestone observations')
            rows.append(dict(condition=condition,seed=seed,parameters=r['parameters'],parameter_ratio=r['parameter_ratio'],
                best_epoch=r['best_epoch'],best_train_accuracy=r['best']['train']['accuracy'],
                best_val_accuracy=r['best']['validation']['accuracy'],best_val_loss=r['best']['validation']['loss'],
                final_train_accuracy=r['final']['train']['accuracy'],final_val_accuracy=r['final']['validation']['accuracy'],
                train_seconds=r['train_seconds'],export_delta=r['export_delta']))
    output.mkdir(parents=True,exist_ok=True)
    f=pd.DataFrame(rows);f.to_csv(output/'runs.csv',index=False)
    a=f.groupby('condition').agg(n=('seed','count'),parameters=('parameters','first'),ratio=('parameter_ratio','first'),
        train_mean=('best_train_accuracy','mean'),val_mean=('best_val_accuracy','mean'),val_std=('best_val_accuracy','std'),
        val_loss_mean=('best_val_loss','mean'),final_val_mean=('final_val_accuracy','mean'),train_seconds=('train_seconds','mean'))
    a.to_csv(output/'summary.csv');write_json(output/'manifest.json',dict(protocol=p,runs=records))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for record in records:
        h=pd.DataFrame(record['metrics'])
        for ax,key in zip(axes,['train_loss','val_accuracy']):
            ax.plot(h.epoch,h[key],label=f"{record['result']['condition']}/{record['result']['seed']}");ax.set(xlabel='Epoch',ylabel=key)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(output/'curves.png',dpi=130);plt.close(fig)
    title='MNIST r3更新預算研究' if p.get('probability_processing')=='upstream' else 'MNIST r2較長學習pilot'
    text='# '+title+'\n\nStudy: `'+root.name+'`\n\n'+('SMOKE，僅工程驗證。\n\n' if p['smoke'] else '')
    text+=a.fillna('N/A').to_markdown()+'\n\n'+f.fillna('N/A').to_markdown(index=False)
    text+='\n\n![curves](curves.png)\n\n每線一個condition/seed，1000train/200validation。Best train/validation同checkpoint；final另存。論文MNIST60000train、10000test，本pilot不讀official test，不能將本validation accuracy稱重現論文test。參數比例6.831%是架構計數，不證明accuracy維持。updates.json記錄最後batch梯度norm與epoch首尾delta，不是每batch平均。計時僅本機train-only，不代表QPU加速。\n'
    if p.get('milestones'):
        milestone_rows=[dict(condition=r['result']['condition'],seed=r['result']['seed'],epoch=m['epoch'],updates=m['updates'],
            train_accuracy=m['train']['accuracy'],val_accuracy=m['validation']['accuracy'],val_loss=m['validation']['loss'])
            for r in records for m in r['milestones']]
        mf=pd.DataFrame(milestone_rows);mf.to_csv(output/'milestones.csv',index=False)
        text+='\n## 固定預算節點\n'+mf.to_markdown(index=False)+'\n\n節點是當下epoch模型，不是事後挑選。500epochs=4000updates，只重複1000筆，不等於60000筆完整MNIST。\n'
    (output/'study.md').write_text(text,encoding='utf-8');print(a.to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');p.add_argument('--report')
    p.add_argument('--output',default='results');p.add_argument('--data',default='data/mnist');a=p.parse_args()
    if a.report:report(a.report,a.output)
    else:run(a.output,a.data,a.smoke)
