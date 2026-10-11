"""Full MNIST training-set coverage pilot at a fixed update budget.

Splits the official 60000-image training file into ~54000 train and a
train-carved validation; the official 10000-image test set is never read.
Same update budget as the 10000-image study so data coverage is the main change.
"""
import argparse
import json
import uuid
from pathlib import Path
import pandas as pd
import torch
from qt_eval.mnist_data import load_mnist_full
from qt_eval.paper_mnist import TargetCNN, PaperQuantumTrain
from qt_eval.lowrank import LowRankCNN
from qt_eval.update_budget import train_updates
from qt_eval.artifacts import write_json, environment
from qt_eval.config import validate_config

CONDITIONS = ['cnn', 'lowrank3', 'lowrank4', 'qt16', 'qt106']
PARAMETERS = {'cnn': 6690, 'lowrank3': 1511, 'lowrank4': 1998, 'qt16': 457, 'qt106': 1627}
TEN_K = Path('results/paper-data-extended-d206b2e56a')
LOWRANK = Path('results/paper-lowrank-42eadd6099')


def build(condition, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if condition == 'cnn':
            return TargetCNN().double()
        if condition == 'qt16':
            return PaperQuantumTrain(16, global_center=True, probability_processing='upstream')
        if condition == 'qt106':
            return PaperQuantumTrain(106, global_center=True, probability_processing='upstream')
        if condition.startswith('lowrank'):
            return LowRankCNN(int(condition[len('lowrank'):]))
    raise ValueError('Unknown condition')


def run(output='results', smoke=False, resume_root=None, updates=None, lowranks=(3,4)):
    data = load_mnist_full('data/mnist', val_size=64 if smoke else 6000)
    conditions = ['cnn'] + [f'lowrank{r}' for r in lowranks] + ['qt16', 'qt106']
    parameters = dict(PARAMETERS)
    for r in lowranks:
        parameters[f'lowrank{r}'] = 487 * r + 50
    total_updates = updates if updates is not None else (4 if smoke else 12000)
    protocol = dict(seeds=[11] if smoke else [11], conditions=conditions, parameters=parameters,
        updates=total_updates, eval_every=2 if smoke else 500,
        milestones=[4] if smoke else [total_updates // 2, total_updates], smoke=smoke,
        train_size=len(data['train']),
        validation_size=len(data['validation']), batch_size=128, lr=.0001, test_access=False,
        train_eval_size=64 if smoke else 2000,
        official_test_read=False, source='train-carved validation; official test untouched',
        comparison='same 12000-update budget as the 10000-image coverage result', status='running')
    if resume_root:
        root = Path(resume_root)
        existing = json.loads((root / 'protocol.json').read_text())
        for key in ('seeds', 'conditions', 'updates', 'eval_every', 'milestones', 'smoke', 'batch_size'):
            if existing.get(key) != protocol.get(key):
                raise ValueError(f'Resume protocol mismatch: {key}')
        if json.loads((root / 'indices.json').read_text()) != data['indices']:
            raise ValueError('Resume data indices mismatch')
        protocol = existing
        print('Resuming', root, flush=True)
    else:
        root = Path(output) / ('paper-full-' + uuid.uuid4().hex[:10])
        root.mkdir(parents=True)
        write_json(root / 'protocol.json', protocol)
        write_json(root / 'indices.json', data['indices'])
        write_json(root / 'status.json', {'status': 'running'})
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for seed in protocol['seeds']:
            for condition in conditions:
                folder = root / f'{condition}-seed{seed}'
                if (folder / 'result.json').exists() and json.loads(
                        (folder / 'result.json').read_text()).get('status') == 'completed':
                    print('skip', condition, seed, flush=True)
                    continue
                folder.mkdir(exist_ok=True)
                resume = (folder / 'latest.pt').exists()
                print(condition, seed, flush=True)
                with torch.random.fork_rng(devices=[]):
                    model = build(condition, seed)
                    write_json(folder / ('resume-environment.json' if resume else 'environment.json'),
                               environment(validate_config({'seed': seed})))
                    write_json(folder / 'config.json', protocol | {'seed': seed, 'condition': condition})
                    result = train_updates(model, {k: data[k] for k in ['train', 'validation']}, folder,
                        updates=protocol['updates'], eval_every=protocol['eval_every'], seed=seed,
                        milestones=protocol['milestones'], resume=resume,
                        train_eval_size=protocol['train_eval_size'])
                    delta = None
                    if condition != 'cnn':
                        exported = model.export_classical()
                        x = data['train'].tensors[0][:16]
                        with torch.no_grad():
                            torch.testing.assert_close(model(x), exported(x), atol=1e-10, rtol=1e-9)
                            delta = float((model(x) - exported(x)).abs().max())
                        torch.save(exported.state_dict(), folder / 'classical-export.pt')
                    write_json(folder / 'result.json', dict(status='completed', seed=seed, condition=condition,
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
        raise ValueError('Smoke run is not reportable')
    rows, records = [], []
    for seed in protocol['seeds']:
        for condition in protocol['conditions']:
            folder = root / f'{condition}-seed{seed}'
            result = json.loads((folder / 'result.json').read_text())
            if result['status'] != 'completed' or result['updates'] != protocol['updates']:
                raise ValueError(f'Incomplete full-data run: {folder}')
            records.append({'condition': condition, 'seed': seed, 'result': result, 'environment': _env(folder)})
            rows.append(dict(condition=condition, seed=seed, parameters=result['parameters'],
                ratio=result['parameters'] / 6690, best_update=result['best_update'],
                train_accuracy=result['best']['train']['accuracy'],
                val_accuracy=result['best']['validation']['accuracy'],
                val_loss=result['best']['validation']['loss'], train_seconds=result['train_seconds']))
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'runs.csv', index=False)
    write_json(output / 'manifest.json', dict(protocol=protocol, runs=records))
    tenk = {'cnn': 0.963333, 'qt16': 0.76, 'qt106': 0.90, 'lowrank3': 0.853333, 'lowrank4': 0.90}
    comparison = frame[['condition', 'parameters', 'val_accuracy']].copy()
    comparison['val_accuracy_10k_data'] = comparison['condition'].map(tenk)
    text = '# 完整MNIST訓練集覆蓋 pilot\n\nStudy: `' + root.name + '`\n\n單seed(11)、54000train／6000validation（由官方train切分，官方test未讀）、'
    text += f"{protocol['updates']}updates、batch128。與10000筆同更新預算比較：\n\n"
    text += frame.to_markdown(index=False) + '\n\n' + comparison.to_markdown(index=False)
    text += ('\n\n這是資料覆蓋的單seed pilot，不是論文50epochs(約23450updates)、也不是官方test結果。'
             '10k欄為既有三seed結果，僅供方向比較。未使用official test。\n')
    (output / 'study.md').write_text(text, encoding='utf-8')
    print(frame.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume-root')
    parser.add_argument('--report')
    parser.add_argument('--updates', type=int)
    parser.add_argument('--lowranks', default='3,4')
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    ranks = tuple(int(x) for x in args.lowranks.split(',') if x)
    if args.report:
        report(args.report, args.output)
    else:
        run(args.output, args.smoke, args.resume_root, args.updates, ranks)
