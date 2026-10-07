"""Quantum sensitivity followed by paired frozen/trainable quantum runs."""
import argparse
import json
import uuid
from pathlib import Path
import pandas as pd
import torch
from qt_eval.artifacts import write_json
from qt_eval.data import load_digits_splits
from qt_eval.quantum import QuantumTrain
from qt_eval.sensitivity import measure_sensitivity
from qt_eval.training_diagnostics import run_diagnostic


def run(output,smoke=False):
    root=Path(output)/('quantum-ablation-'+uuid.uuid4().hex[:10]);root.mkdir(parents=True)
    data=load_digits_splits(smoke=smoke)
    splits={k:data[k] for k in ('train','validation')}
    x,y=splits['train'].tensors
    idx=torch.cat([torch.where(y==i)[0][:2] for i in range(10)]);x,y=x[idx],y[idx]
    protocol=dict(smoke=smoke,seeds=[11] if smoke else [11,22,33],epochs=2 if smoke else 200,
        conditions=['trainable','frozen'],mapping='tanh',target='leaky_relu',negative_slope=.01,
        depth=3,q_delta=.001,learning_rate=.001,weight_transform='center_weights',test_access=False,status='running',
        sensitivity_epsilons=[.001,.01,.1],direction_seed=2026)
    write_json(root/'protocol.json',protocol)
    write_json(root/'split_indices.json',{k:data['indices'][k] for k in splits})
    write_json(root/'tiny_indices.json',[data['indices']['train'][i] for i in idx.tolist()])
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        for seed in protocol['seeds']:
            torch.manual_seed(seed)
            m=QuantumTrain(3,mapping_activation='tanh',target_activation='leaky_relu',weight_transform='center_weights')
            write_json(root/f'initial-sensitivity-{seed}.json',measure_sensitivity(m,x,y))
            for condition in protocol['conditions']:
                print(f'{condition} seed={seed}',flush=True)
                run_diagnostic('tanh',seed,splits,root/condition,epochs=protocol['epochs'],tiny=True,
                    target_activation='leaky_relu',weight_transform='center_weights',train_quantum=condition=='trainable')
                folder=root/condition/f'tanh-seed{seed}-tiny'
                m.load_state_dict(torch.load(folder/'best.pt',weights_only=True)['model_state'])
                write_json(folder/'sensitivity.json',measure_sensitivity(m,x,y))
        protocol['status']='completed';write_json(root/'protocol.json',protocol)
    finally:
        torch.set_num_threads(previous)
    print(root,flush=True)
    return root


def report(root,output):
    import matplotlib.pyplot as plt
    root,output=Path(root),Path(output)
    protocol=json.loads((root/'protocol.json').read_text())
    records=[];rows=[]
    for seed in protocol['seeds']:
        for condition in protocol['conditions']:
            folder=root/condition/f'tanh-seed{seed}-tiny'
            record={k:json.loads((folder/f'{k}.json').read_text()) for k in ['config','environment','result','diagnostic','sensitivity']}
            if record['result']['status']!='completed' or record['config']['train_quantum']!=(condition=='trainable'):
                raise ValueError('Incomplete/inconsistent run')
            d=record['diagnostic'];h=d['history']
            if condition=='frozen' and any(r['quantum_max_delta']!=0 for r in h):
                raise ValueError('Frozen parameter changed')
            record['condition']=condition;records.append(record)
            rows.append(dict(condition=condition,seed=seed,max_train_accuracy=max(r['train_accuracy'] for r in h),
                final_train_accuracy=h[-1]['train_accuracy'],best_val_accuracy=d['best_validation']['accuracy'],
                best_val_loss=d['best_validation']['loss'],best_epoch=d['best_epoch'],
                zero_feature_weight_relative_delta=record['sensitivity']['zero_feature_weight_relative_delta']))
    output.mkdir(parents=True,exist_ok=True)
    initial={str(seed):json.loads((root/f'initial-sensitivity-{seed}.json').read_text()) for seed in protocol['seeds']}
    write_json(output/'manifest.json',dict(protocol=protocol,initial_sensitivity=initial,runs=records))
    frame=pd.DataFrame(rows);frame.to_csv(output/'runs.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for r in records:
        h=pd.DataFrame(r['diagnostic']['history'])
        for ax,key in zip(axes,['train_accuracy','val_loss']):
            ax.plot(h.epoch,h[key],label=f"{r['condition']}-{r['config']['seed']}")
            ax.set(xlabel='Epoch',ylabel=key)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(output/'curves.png',dpi=130);plt.close(fig)
    text='# 量子敏感度與固定參數消融\n\nStudy: `'+root.name+'`\n\n'+('SMOKE：僅工程檢查。\n\n' if protocol['smoke'] else '')
    text+=frame.to_markdown(index=False)+'\n\n![curves](curves.png)\n\n相同20筆資料、同seed原始模型參數，只有量子參數是否更新不同。Frozen仍使用初始量子機率，不是無量子模型。Null量子梯度是預期固定，不是錯誤。敏感度使用一個固定方向，不能代表所有方向；feature置零是未重訓干預，不是零特徵訓練baseline。Best依validation loss選擇，未使用test。當前低表現設定的消融不能推論一般量子優勢或無效。完整設定、環境、逐epoch資料與敏感度見manifest.json。\n'
    (output/'study.md').write_text(text,encoding='utf-8');print(frame.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true')
    p.add_argument('--report');p.add_argument('--output',default='results');a=p.parse_args()
    if a.report: report(a.report,a.output)
    else: run(a.output,a.smoke)
