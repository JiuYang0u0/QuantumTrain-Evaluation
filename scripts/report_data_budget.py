"""Export only complete paired sample-coverage matrices."""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from qt_eval.artifacts import write_json


def report(root,output):
    root,output=Path(root),Path(output);protocol=json.loads((root/'protocol.json').read_text())
    rows=[];records=[]
    indices={str(n):json.loads((root/f'indices-{n}.json').read_text()) for n in protocol['sizes']}
    reference=None
    for d in indices.values():
        assert not set(d['train'])&set(d['validation'])
        if reference is not None:assert reference==d['validation']
        reference=d['validation']
    for seed in protocol['seeds']:
        for n in protocol['sizes']:
            for condition in protocol['conditions']:
                folder=root/f'{condition}-n{n}-seed{seed}'
                record={k:json.loads((folder/f'{k}.json').read_text()) for k in ['config','environment','result']}
                r=record['result']
                if (r['status']!='completed' or r['updates']!=protocol['updates'] or r['examples_seen']!=protocol['updates']*128
                    or r['train_size']!=n or r['condition']!=condition or r['seed']!=seed):raise ValueError('Incomplete/inconsistent budget study')
                records.append(record)
                rows.append(dict(condition=condition,train_size=n,seed=seed,parameters=r['parameters'],
                    updates=r['updates'],examples_seen=r['examples_seen'],unique_examples_seen=r['unique_examples_seen'],
                    best_update=r['best_update'],train_accuracy=r['best']['train']['accuracy'],val_accuracy=r['best']['validation']['accuracy'],
                    val_loss=r['best']['validation']['loss'],final_val_accuracy=r['final']['validation']['accuracy'],export_delta=r['export_delta']))
    output.mkdir(parents=True,exist_ok=True)
    f=pd.DataFrame(rows);f.to_csv(output/'runs.csv',index=False)
    a=f.groupby(['condition','train_size']).agg(n=('seed','count'),train_mean=('train_accuracy','mean'),val_mean=('val_accuracy','mean'),val_std=('val_accuracy','std'),val_loss_mean=('val_loss','mean'))
    a.to_csv(output/'summary.csv');write_json(output/'manifest.json',dict(protocol=protocol,indices=indices,runs=records))
    fig,ax=plt.subplots(figsize=(10,5))
    for r in records:
        h=r['result']['history'];result=r['result']
        ax.plot([x['update'] for x in h],[x['validation']['accuracy'] for x in h],label=f"{result['condition']}/{result['train_size']}/{result['seed']}")
    ax.set(xlabel='Optimizer updates',ylabel='Validation accuracy');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(output/'curves.png',dpi=130);plt.close(fig)
    text='# MNIST同更新預算資料覆蓋研究\n\nStudy: `'+root.name+'`\n\n'+('SMOKE：只驗證流程。\n\n' if protocol['smoke'] else '')
    text+=a.fillna('N/A').to_markdown()+'\n\n'+f.fillna('N/A').to_markdown(index=False)
    text+='\n\n![curves](curves.png)\n\n每線一個模型/資料量/seed；橫軸updates。兩組同updates、每batch128、相同validation及評估次數。使用有放回抽樣，不能直接與舊epoch shuffle研究混比。集合交集與nested狀態見manifest；best/final分開，validation曾參與開發，非盲測。未使用official test，不是完整論文MNIST重現。\n'
    (output/'study.md').write_text(text,encoding='utf-8');print(a.to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study');p.add_argument('--output',default='reports/paper-data-budget')
    a=p.parse_args();report(a.study,a.output)
