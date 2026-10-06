"""Protocol-aware summaries. Incomplete and failed runs remain visible."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import validate_config


def read_runs(run_dirs):
    records=[]
    for path in run_dirs:
        path=Path(path)
        config=validate_config(json.loads((path/'config.json').read_text(encoding='utf-8')))
        result=json.loads((path/'result.json').read_text(encoding='utf-8'))
        group='classical' if config['model']=='classical' else f"quantum-depth{config['q_depth']}"
        records.append(dict(path=path,config=config,result=result,group=group))
    return records


def summarize_runs(run_dirs, allow_smoke=False, expected_groups=None):
    records=read_runs(run_dirs)
    if not records:
        raise ValueError('No runs supplied')
    protocol=None
    seen=set()
    for r in records:
        c=r['config']
        if c['mode']=='smoke' and not allow_smoke:
            raise ValueError('Smoke runs require allow_smoke=True; not scientific evidence')
        current={k:v for k,v in c.items() if k not in ('model','q_depth','seed')}
        if protocol is not None and current != protocol:
            raise ValueError('Incompatible experiment protocol')
        protocol=current
        key=(r['group'],c['seed'])
        if key in seen:
            raise ValueError(f'Duplicate group/seed: {key}')
        seen.add(key)
    expected=[11] if protocol['mode']=='smoke' else [11,22,33]
    if any(r['config']['seed'] not in expected for r in records):
        raise ValueError('Unexpected seed for the declared study protocol')
    groups=sorted(set(expected_groups or []) | {r['group'] for r in records})
    rows=[]
    for group in groups:
        subset=[r for r in records if r['group']==group]
        good=[r for r in subset if r['result']['status']=='completed']
        for r in good:
            test=r['result']['test']
            if not np.isfinite(test['accuracy']) or not 0 <= test['accuracy'] <= 1 or not np.isfinite(test['loss']):
                raise ValueError('Invalid completed result metrics')
        missing=sorted(set(expected)-{r['config']['seed'] for r in good})
        accuracy=[r['result']['test']['accuracy'] for r in good]
        rows.append(dict(group=group,mode=protocol['mode'],expected=len(expected),completed=len(good),
                         failed=sum(r['result']['status']=='failed' for r in subset),
                         missing_seeds=missing,status='complete' if not missing else 'incomplete',
                         accuracy_mean=float(np.mean(accuracy)) if accuracy else np.nan,
                         accuracy_std=float(np.std(accuracy,ddof=1)) if len(accuracy)>1 else np.nan,
                         train_seconds_mean=float(np.mean([r['result']['train_seconds'] for r in good])) if good else np.nan,
                         parameters=good[0]['result']['parameters']['total'] if good else np.nan))
    return pd.DataFrame(rows)


def write_report(run_dirs, output_dir, allow_smoke=False, expected_groups=None):
    run_dirs=list(run_dirs)
    frame=summarize_runs(run_dirs,allow_smoke,expected_groups)
    records=read_runs(run_dirs)
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output/'summary.csv',index=False)
    figures=[]
    for metric in ('train_loss','val_loss','val_accuracy'):
        fig,ax=plt.subplots(figsize=(8,4))
        for r in records:
            path=r['path']/'metrics.csv'
            if path.exists():
                history=pd.read_csv(path)
                ax.plot(history.epoch,history[metric],label=f"{r['group']} seed={r['config']['seed']}")
        ax.set(xlabel='Epoch',ylabel=metric,title=metric+' (individual runs)')
        if ax.lines:
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output/f'{metric}.png',dpi=130)
        plt.close(fig)
        figures.append(f'![{metric}]({metric}.png)\n\n橫軸 epoch；縱軸 {metric}，每線一個 run；loss 為樣本平均交叉熵，accuracy 為 0–1。曲線不代表量子優勢。')
    for metric,label in [('accuracy_mean','Test accuracy (0–1)'),('train_seconds_mean','Training time (seconds)')]:
        fig,ax=plt.subplots(figsize=(8,4))
        available=frame[frame.completed>0]
        ax.bar(available.group,available[metric])
        if metric=='accuracy_mean':
            multi=available[available.completed>1]
            ax.errorbar(multi.group,multi[metric],yerr=multi.accuracy_std,fmt='none',color='black',capsize=4)
        ax.set(ylabel=label)
        fig.tight_layout()
        fig.savefig(output/f'{metric}.png',dpi=130)
        plt.close(fig)
        figures.append(f'![{label}]({metric}.png)\n\n組別比較：{label}；accuracy 誤差棒為樣本標準差，單次不畫誤差棒。不完整組只描述已完成的 runs。')
    display=frame.copy()
    for c in ('accuracy_mean','accuracy_std','train_seconds_mean'):
        display[c]=display[c].map(lambda x:'N/A' if pd.isna(x) else f'{x:.6f}')
    title='SMOKE 工程檢查（不可用於研究結論）' if frame.iloc[0]['mode']=='smoke' else 'QuantumTrain 電路深度適配實驗'
    failures='\n'.join(f"- {r['path'].name}: {r['result']['status']} {r['result'].get('error','')}" for r in records)
    manifest=[{'run_id':r['path'].name,'config':r['config'],'result':r['result']} for r in records]
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    text=f'# {title}\n\n研究問題：同 epoch 預算下，1/3/5 層電路如何影響精度與成本？\n\n'
    text+='來源：[QuantumTrain](https://github.com/Hon-Hai-Quantum-Computing/QuantumTrain)，參照 commit d5b654116565ae9550e540e021ab0d4160a15c86。\n\n'
    text+=display.to_markdown(index=False)+'\n\n## Run 狀態\n'+failures+'\n\n## 圖表\n'+'\n\n'.join(figures)
    text+='\n\n## 協定與限制\n完整設定、逐次結果與 run IDs 見 manifest.json；原始 metrics/checkpoint 見本機 results。以 validation loss 選 checkpoint，test 只作 final evaluation。三 seeds 是描述統計；同 epochs 不等於同成本；深度同時改變參數量。10-qubit 理想模擬 backprop 不代表 QPU 成本。Mapping 比目標 MLP 大，不能宣稱壓縮。失敗與缺 seed 不可當作成功排除。\n'
    report=output/'depth-study.md'
    report.write_text(text,encoding='utf-8')
    return report
