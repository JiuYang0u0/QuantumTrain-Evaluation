"""Weight-only block centering: one transformation, unchanged biases/std/LR."""
import argparse
import json
import uuid
from pathlib import Path
import pandas as pd
import torch
from qt_eval.artifacts import write_json
from qt_eval.data import load_digits_splits
from qt_eval.quantum import QuantumTrain
from qt_eval.models import ClassicalMLP
from qt_eval.training_diagnostics import run_diagnostic


def initialization_evidence(x):
    rows=[]
    for seed in [11,22,33]:
        for kind in ['classical','none','center_weights']:
            torch.manual_seed(seed)
            if kind=='classical':
                m=ClassicalMLP('leaky_relu').double()
                w=torch.cat([p.flatten() for p in m.parameters()]).detach()
            else:
                m=QuantumTrain(3,mapping_activation='tanh',target_activation='leaky_relu',weight_transform=kind)
                w=m.generated_weights().detach()
            blocks={k:dict(mean=float(v.mean()),std=float(v.std(unbiased=False))) for k,v in
                    [('W1',w[:512]),('b1',w[512:520]),('W2',w[520:600]),('b2',w[600:])]}
            z=x@w[:512].reshape(8,64).T+w[512:520]
            rows.append(dict(seed=seed,kind=kind,blocks=blocks,preactivation_mean=float(z.mean()),
                             preactivation_std=float(z.std(unbiased=False)),negative_fraction=float((z<=0).double().mean())))
    return rows


def run(output,smoke=False):
    root=Path(output)/('centering-'+uuid.uuid4().hex[:10]);root.mkdir(parents=True)
    data=load_digits_splits(smoke=smoke)
    splits={k:data[k] for k in ['train','validation']}
    x,y=splits['train'].tensors
    tiny_index=torch.cat([torch.where(y==i)[0][:2] for i in range(10)])
    protocol=dict(smoke=smoke,seeds=[11] if smoke else [11,22,33],epochs=2 if smoke else 200,
                  transforms=['none','center_weights'],mapping='tanh',target='leaky_relu',negative_slope=.01,
                  depth=3,learning_rate=.001,q_delta=.001,test_access=False,status='running',
                  definition='Subtract each weight matrix mean (W1 and W2); leave biases and std unchanged')
    write_json(root/'protocol.json',protocol)
    write_json(root/'split_indices.json',{k:data['indices'][k] for k in splits})
    write_json(root/'tiny_indices.json',[data['indices']['train'][i] for i in tiny_index.tolist()])
    write_json(root/'initialization.json',initialization_evidence(x[tiny_index]))
    for seed in protocol['seeds']:
        for transform in protocol['transforms']:
            print(f'{transform} seed={seed}',flush=True)
            run_diagnostic('tanh',seed,splits,root/transform,epochs=protocol['epochs'],tiny=True,
                           target_activation='leaky_relu',weight_transform=transform)
    protocol['status']='completed';write_json(root/'protocol.json',protocol)
    print(root,flush=True)
    return root


def report(root,output):
    import matplotlib.pyplot as plt
    root,output=Path(root),Path(output)
    p=json.loads((root/'protocol.json').read_text())
    rows=[];records=[]
    for seed in p['seeds']:
        for transform in p['transforms']:
            folder=root/transform/f'tanh-seed{seed}-tiny'
            record={k:json.loads((folder/f'{k}.json').read_text()) for k in ['config','environment','result','diagnostic']}
            if record['result']['status']!='completed' or record['config']['weight_transform']!=transform:
                raise ValueError('Incomplete or inconsistent run')
            record['run_id']=str(folder.relative_to(root));records.append(record)
            d=record['diagnostic'];h=d['history']
            rows.append(dict(transform=transform,seed=seed,maximum_train_accuracy=max(r['train_accuracy'] for r in h),
                final_train_accuracy=h[-1]['train_accuracy'],final_train_loss=h[-1]['train_loss'],
                best_epoch=d['best_epoch'],best_train_accuracy=d['best_train']['accuracy'],
                best_val_accuracy=d['best_validation']['accuracy'],best_val_loss=d['best_validation']['loss']))
    output.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows);frame.to_csv(output/'runs.csv',index=False)
    initial=json.loads((root/'initialization.json').read_text())
    write_json(output/'manifest.json',dict(protocol=p,initialization=initial,runs=records))
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for record in records:
        h=pd.DataFrame(record['diagnostic']['history'])
        for ax,key in zip(axes,['train_accuracy','train_loss','val_loss']):
            ax.plot(h.epoch,h[key],label=record['run_id'].split('/')[0]+'/'+str(record['config']['seed']))
            ax.set(xlabel='Epoch',ylabel=key)
    axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(output/'curves.png',dpi=130);plt.close(fig)
    text='# 生成權重中心化對照\n\nStudy: `'+root.name+'`\n\n'+('SMOKE，不作研究結論。\n\n' if p['smoke'] else '')
    text+=frame.to_markdown(index=False)+'\n\n![curves](curves.png)\n\n固定20筆train；各線一個處理/seed。沒有test；best checkpoint依validation loss。中心化僅W1/W2各自減平均，不改bias、不除以std、不做fan-in scaling。同seed具有相同原始初始化。最高train accuracy僅為擬合診斷。\n\n設定、初始化層級統計和逐epoch診斷見manifest.json。\n'
    (output/'study.md').write_text(text,encoding='utf-8');print(frame.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true')
    p.add_argument('--report');p.add_argument('--output',default='results');a=p.parse_args()
    if a.report: report(a.report,a.output)
    else: run(a.output,a.smoke)
