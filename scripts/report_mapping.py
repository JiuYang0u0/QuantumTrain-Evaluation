"""Export a mapping study, including failed/incomplete matrix evidence."""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def report(root,output):
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((root/'protocol.json').read_text())
    records=[]
    for tiny in (True,False):
        for seed in protocol['seeds']:
            for activation in ('affine','tanh'):
                path=root/(f'{activation}-seed{seed}-'+('tiny' if tiny else 'full'))
                item=dict(run_id=path.name,tiny=tiny,seed=seed,activation=activation)
                for name in ('config','environment','result','diagnostic'):
                    file=path/f'{name}.json'
                    item[name]=json.loads(file.read_text()) if file.exists() else None
                records.append(item)
    good=[r for r in records if r['diagnostic'] and r['result']['status']=='completed']
    rows=[]
    for r in good:
        d=r['diagnostic']
        rows.append(dict(activation=r['activation'],tiny=r['tiny'],seed=r['seed'],
                         best_val_accuracy=d['best_validation']['accuracy'],
                         best_train_accuracy=d['best_train']['accuracy'],
                         maximum_train_accuracy=max(h['train_accuracy'] for h in d['history']),
                         train_seconds=sum(h['train_seconds'] for h in d['history'])))
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'runs.csv',index=False)
    (output/'manifest.json').write_text(json.dumps(dict(protocol=protocol,runs=records),ensure_ascii=False,indent=2),encoding='utf-8')
    grouped=frame.groupby(['tiny','activation']).agg(n=('seed','count'),val_mean=('best_val_accuracy','mean'),
                        val_std=('best_val_accuracy','std'),train_max_mean=('maximum_train_accuracy','mean'),
                        seconds_mean=('train_seconds','mean')).reset_index()
    grouped.to_csv(output/'summary.csv',index=False)
    for tiny in (True,False):
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        for r in good:
            if r['tiny']!=tiny:
                continue
            h=pd.DataFrame(r['diagnostic']['history'])
            label=f"{r['activation']} seed={r['seed']}"
            axes[0].plot(h.epoch,h.train_accuracy,label=label)
            axes[1].plot(h.epoch,h.quantum_grad_norm,label=label)
            axes[2].plot(h.epoch,[o['relu_zero_fraction'] for o in h.observation],label=label)
        for ax,ylabel in zip(axes,['Train accuracy','Mean quantum gradient norm','Train probe ReLU zero fraction']):
            ax.set(xlabel='Epoch',ylabel=ylabel)
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(output/('tiny.png' if tiny else 'full.png'),dpi=130)
        plt.close(fig)
    text='# Mapping 診斷與延伸報告\n\n'
    text+=f"來源 study：`{root.name}`；完成 {len(good)}/{len(records)} runs。Mode：{'SMOKE，不作研究結論' if protocol['smoke'] else 'validation-only exploratory study'}。\n\n"
    text+=grouped.fillna('N/A').to_markdown(index=False)
    text+='\n\ntrain_max_mean 是各run整段訓練最高accuracy的平均，僅供過擬合診斷；val_mean由validation loss最佳checkpoint計算，不是最高validation accuracy。秒數只含訓練batch。\n\n'
    text+='![Tiny](tiny.png)\n\n20筆訓練集：各線為一個seed；左右依序為訓練accuracy、量子梯度norm、固定train probe的ReLU零比例。\n\n![Full](full.png)\n\n完整train情境（smoke則為40筆）：相同軸定義。\n\n'
    text+='## 限制\n固定depth3、q_delta=.001、Adam lr=.001；兩組初始參數相同，Tanh不增加參數。此輪沒有使用test。非線性mapping的效果不能證明量子學習的增量價值；仍需固定量子參數消融。三seeds僅供描述，validation也不能當反覆調參後的獨立最終評估。詳細config、逐epoch分布、梯度、更新量與環境見manifest.json。\n'
    (output/'mapping-study.md').write_text(text,encoding='utf-8')
    print(grouped.to_string(index=False))
    if len(good)!=len(records):
        raise RuntimeError('Incomplete mapping study; see manifest')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('study')
    p.add_argument('--output',default='reports/mapping')
    args=p.parse_args()
    report(args.study,args.output)
