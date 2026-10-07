"""Frozen train/validation-only LR study, preserving a complete run manifest."""
import argparse
import json
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from qt_eval.artifacts import write_json
from qt_eval.data import load_digits_splits
from qt_eval.training_diagnostics import run_diagnostic


def summarize(d,rate,seed):
    history=d['history']
    accuracy=[r['train_accuracy'] for r in history]
    loss=[r['train_loss'] for r in history]
    return dict(learning_rate=rate,seed=seed,maximum_train_accuracy=max(accuracy),
                final_train_accuracy=accuracy[-1],peak_to_final_drop=max(accuracy)-accuracy[-1],
                tail_accuracy_std=float(np.std(accuracy[-20:])),
                tail_loss_std=float(np.std(loss[-20:])),final_train_loss=loss[-1],
                best_epoch=d['best_epoch'],best_val_accuracy=d['best_validation']['accuracy'],
                best_val_loss=d['best_validation']['loss'])


def run(output,smoke=False):
    root=Path(output)/('learning-rate-'+uuid.uuid4().hex[:10])
    root.mkdir(parents=True)
    data=load_digits_splits(smoke=smoke)
    splits={k:data[k] for k in ('train','validation')}
    protocol=dict(smoke=smoke,rates=[.001,.0003,.0001],seeds=[11] if smoke else [11,22,33],
                  epochs=2 if smoke else 200,mapping='tanh',target='leaky_relu',negative_slope=.01,
                  depth=3,q_delta=.001,train_samples=20,split_seed=2026,test_access=False,
                  tail_epochs=20,status='running')
    write_json(root/'protocol.json',protocol)
    write_json(root/'split_indices.json',{k:data['indices'][k] for k in splits})
    rows=[]
    for seed in protocol['seeds']:
        for rate in protocol['rates']:
            print(f'lr={rate} seed={seed}',flush=True)
            d=run_diagnostic('tanh',seed,splits,root/f'lr-{rate:g}',epochs=protocol['epochs'],
                             tiny=True,target_activation='leaky_relu',learning_rate=rate)
            rows.append(summarize(d,rate,seed))
            pd.DataFrame(rows).to_csv(root/'summary.csv',index=False)
    protocol['status']='completed'
    write_json(root/'protocol.json',protocol)
    print(root,flush=True)
    return root


def report(root,output):
    import matplotlib.pyplot as plt
    root,output=Path(root),Path(output)
    protocol=json.loads((root/'protocol.json').read_text())
    records=[]
    rows=[]
    for seed in protocol['seeds']:
        for rate in protocol['rates']:
            p=root/f'lr-{rate:g}'/f'tanh-seed{seed}-tiny'
            if not (p/'diagnostic.json').exists():
                raise ValueError(f'Incomplete matrix: {p}')
            record=dict(run_id=str(p.relative_to(root)))
            for key in ('config','environment','result','diagnostic'):
                record[key]=json.loads((p/f'{key}.json').read_text())
            if record['result']['status']!='completed' or record['config']['learning_rate']!=rate:
                raise ValueError(f'Incomplete/inconsistent run: {p}')
            records.append(record)
            rows.append(summarize(record['diagnostic'],rate,seed))
    output.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'runs.csv',index=False)
    aggregate=frame.groupby('learning_rate').agg(n=('seed','count'),
        max_train_mean=('maximum_train_accuracy','mean'),final_train_mean=('final_train_accuracy','mean'),
        final_train_std=('final_train_accuracy','std'),drop_mean=('peak_to_final_drop','mean'),
        tail_loss_std_mean=('tail_loss_std','mean'),val_mean=('best_val_accuracy','mean'),val_std=('best_val_accuracy','std'))
    aggregate.to_csv(output/'summary.csv')
    write_json(output/'manifest.json',dict(protocol=protocol,runs=records))
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for r in records:
        h=pd.DataFrame(r['diagnostic']['history'])
        for ax,metric in zip(axes,['train_accuracy','train_loss','val_loss']):
            ax.plot(h.epoch,h[metric],label=f"{r['config']['learning_rate']:g}/{r['config']['seed']}")
            ax.set(xlabel='Epoch',ylabel=metric)
    axes[0].legend(fontsize=6)
    fig.tight_layout();fig.savefig(output/'curves.png',dpi=130);plt.close(fig)
    text='# Learning rate 單變因研究\n\nStudy: `'+root.name+'`\n\n'
    text+=('SMOKE工程檢查，不作研究結論。\n\n' if protocol['smoke'] else '只使用train/validation，不讀test。\n\n')
    text+=aggregate.fillna('N/A').to_markdown()+'\n\n## 逐run\n'+frame.to_markdown(index=False)
    text+='\n\n![curves](curves.png)\n\n每線一個lr/seed；固定20筆，200epochs（smoke只有2epochs）。drop為最高train accuracy減最後accuracy；tail loss std為最後最多20epochs loss的母體標準差，描述時間波動，不是独立抽樣不確定度。val指validation loss最佳checkpoint的accuracy。較低波動不代表學得更好，需同時看loss、accuracy與收斂進度。同epochs不等於各LR都收斂。\n\n## 協定\n```json\n'+json.dumps(protocol,indent=2)+'\n```\n'
    (output/'study.md').write_text(text,encoding='utf-8')
    print(aggregate.to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--report')
    p.add_argument('--output',default='results')
    a=p.parse_args()
    if a.report: report(a.report,a.output)
    else: run(a.output,a.smoke)
