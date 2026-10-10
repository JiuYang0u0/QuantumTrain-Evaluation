"""Low-rank direct training vs QT16/QT106 at comparable trainable budgets.

A stronger classical compression baseline than frozen random features: the
target CNN weights are trained directly as rank-r factors, with no generator.
All data/optimizer settings follow the r3 study.
"""
import argparse
import json
import uuid
from pathlib import Path
import pandas as pd
import torch
from qt_eval.lowrank import LowRankCNN, parameters_for_rank
from qt_eval.mnist_data import load_mnist_pilot
from qt_eval.update_budget import train_updates
from qt_eval.artifacts import write_json, environment
from qt_eval.config import validate_config

QT_EXTENDED = Path('results/paper-data-extended-d206b2e56a')
QT_DEPTH = Path('results/paper-depth-3f729b7257')
RANKS = [1, 2, 3, 4, 6]


def run(output='results', smoke=False, resume_root=None):
    data = load_mnist_pilot('data/mnist', train_size=10000)
    if data['indices'] != json.loads((QT_EXTENDED / 'indices-10000.json').read_text()):
        raise ValueError('MNIST indices differ from the QT control study')
    ranks = RANKS[:1] if smoke else RANKS
    protocol = dict(seeds=[11] if smoke else [11, 22, 33], ranks=ranks,
        parameters={str(r): parameters_for_rank(r) for r in ranks},
        updates=4 if smoke else 12000, eval_every=2 if smoke else 100,
        milestones=[4] if smoke else [4000, 8000, 12000], smoke=smoke, train_size=10000,
        validation_size=200, batch_size=128, lr=.0001, test_access=False,
        method='direct rank-r factorization of target weights; no generator', status='running')
    if resume_root:
        root = Path(resume_root)
        existing = json.loads((root / 'protocol.json').read_text())
        for key in ('seeds', 'ranks', 'updates', 'eval_every', 'milestones', 'smoke', 'batch_size'):
            if existing.get(key) != protocol.get(key):
                raise ValueError(f'Resume protocol mismatch: {key}')
        if json.loads((root / 'indices.json').read_text()) != data['indices']:
            raise ValueError('Resume data indices mismatch')
        protocol = existing
        print('Resuming study', root, flush=True)
    else:
        root = Path(output) / ('paper-lowrank-' + uuid.uuid4().hex[:10])
        root.mkdir(parents=True)
        write_json(root / 'protocol.json', protocol)
        write_json(root / 'indices.json', data['indices'])
        write_json(root / 'status.json', {'status': 'running'})
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for seed in protocol['seeds']:
            for rank in ranks:
                folder = root / f'rank{rank}-seed{seed}'
                if (folder / 'result.json').exists() and json.loads(
                        (folder / 'result.json').read_text()).get('status') == 'completed':
                    print('skip completed', rank, seed, flush=True)
                    continue
                folder.mkdir(exist_ok=True)
                resume = (folder / 'latest.pt').exists()
                print('rank', rank, 'seed', seed, flush=True)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    model = LowRankCNN(rank)
                    write_json(folder / ('resume-environment.json' if resume else 'environment.json'),
                               environment(validate_config({'seed': seed})))
                    write_json(folder / 'config.json', protocol | {'seed': seed, 'rank': rank})
                    result = train_updates(model, {k: data[k] for k in ['train', 'validation']}, folder,
                        updates=protocol['updates'], eval_every=protocol['eval_every'], seed=seed,
                        milestones=protocol['milestones'], resume=resume)
                    exported = model.export_classical()
                    x = data['train'].tensors[0][:16]
                    with torch.no_grad():
                        torch.testing.assert_close(model(x), exported(x), atol=1e-10, rtol=1e-9)
                        delta = float((model(x) - exported(x)).abs().max())
                    torch.save(exported.state_dict(), folder / 'classical-export.pt')
                    write_json(folder / 'result.json', dict(status='completed', seed=seed, rank=rank,
                        parameters=sum(p.numel() for p in model.parameters()), export_delta=delta, **result))
        write_json(root / 'status.json', {'status': 'completed'})
    except BaseException as exc:
        import traceback
        (root / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        write_json(root / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise
    finally:
        torch.set_num_threads(old)
    print(root, flush=True)
    return root


def _env(folder):
    for name in ('environment.json', 'resume-environment.json'):
        if (folder / name).exists():
            return json.loads((folder / name).read_text())
    return None


def report(root, output):
    root, output = Path(root), Path(output)
    protocol = json.loads((root / 'protocol.json').read_text())
    if protocol['smoke']:
        raise ValueError('Smoke run is not comparable to full controls')
    records, rows = [], []
    controls = [('CNN', QT_EXTENDED / 'cnn-n10000-seed{}', 6690),
                ('QT16', QT_EXTENDED / 'qt16_r3-n10000-seed{}', 457),
                ('QT106', QT_DEPTH / 'qt106-seed{}', 1627)]
    for seed in protocol['seeds']:
        for label, pattern, params in controls:
            folder = Path(str(pattern).format(seed))
            result = json.loads((folder / 'result.json').read_text())
            if result['status'] != 'completed' or result['parameters'] != params:
                raise ValueError(f'Bad control {label} {seed}')
            records.append({'label': label, 'seed': seed, 'parameters': params,
                            'source': str(folder), 'result': result, 'environment': _env(folder)})
            rows.append(dict(model=label, rank=None, seed=seed, parameters=params,
                val_accuracy=result['best']['validation']['accuracy'],
                val_loss=result['best']['validation']['loss']))
        for rank in protocol['ranks']:
            folder = root / f'rank{rank}-seed{seed}'
            result = json.loads((folder / 'result.json').read_text())
            if result['status'] != 'completed' or result['updates'] != 12000:
                raise ValueError(f'Incomplete low-rank run rank{rank} seed{seed}')
            records.append({'label': f'lowrank{rank}', 'seed': seed, 'parameters': result['parameters'],
                            'source': str(folder), 'result': result, 'environment': _env(folder)})
            rows.append(dict(model=f'lowrank{rank}', rank=rank, seed=seed, parameters=result['parameters'],
                val_accuracy=result['best']['validation']['accuracy'],
                val_loss=result['best']['validation']['loss']))
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'runs.csv', index=False)
    write_json(output / 'manifest.json', dict(protocol=protocol, runs=records))
    summary = frame.groupby('model').agg(n=('seed', 'count'), parameters=('parameters', 'first'),
        ratio=('parameters', lambda s: s.iloc[0] / 6690),
        val_mean=('val_accuracy', 'mean'), val_std=('val_accuracy', 'std'), val_loss=('val_loss', 'mean'))
    summary.to_csv(output / 'summary.csv')
    text = '# 低秩直接訓練 vs QT16/QT106\n\n' + summary.to_markdown() + '\n\n' + frame.to_markdown(index=False)
    text += ('\n\n低秩直接訓練為古典壓縮基準：直接訓練目標CNN權重的rank-r因子，沒有生成器、'
             '沒有量子電路。資料/最佳化/評估沿用r3。參數含每層(r因子)與bias 50。'
             '未使用official test；200筆開發validation；不構成量子優勢或無效的結論。\n')
    (output / 'study.md').write_text(text, encoding='utf-8')
    print(summary.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--report')
    parser.add_argument('--resume-root')
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    if args.report:
        report(args.report, args.output)
    else:
        run(args.output, args.smoke, args.resume_root)
