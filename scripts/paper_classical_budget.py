"""Matched-budget classical random-feature control vs QT16/QT106.

Only the feature source changes: a fixed random projection + softmax replaces the
quantum circuit. Trainable parameter count and mapping are identical to the QT
counterparts, and all data/optimizer settings follow the r3 study.
"""
import argparse
import json
import uuid
from pathlib import Path
import pandas as pd
import torch
from qt_eval.classical_controls import ClassicalRandomFeatureTrain
from qt_eval.mnist_data import load_mnist_pilot
from qt_eval.update_budget import train_updates
from qt_eval.artifacts import write_json, environment
from qt_eval.config import validate_config

QT_EXTENDED = Path('results/paper-data-extended-d206b2e56a')
QT_DEPTH = Path('results/paper-depth-3f729b7257')
CONDITIONS = {'classical_rf208': 208, 'classical_rf1378': 1378}


def run(output='results', smoke=False):
    data = load_mnist_pilot('data/mnist', train_size=10000)
    expected = json.loads((QT_EXTENDED / 'indices-10000.json').read_text())
    if data['indices'] != expected:
        raise ValueError('MNIST indices differ from the QT control study')
    protocol = dict(seeds=[11] if smoke else [11, 22, 33], conditions=list(CONDITIONS),
        latent_sizes=CONDITIONS, updates=4 if smoke else 12000, eval_every=2 if smoke else 100,
        milestones=[4] if smoke else [4000, 8000, 12000], smoke=smoke, train_size=10000,
        validation_size=200, batch_size=128, lr=.0001, global_center=True,
        probability_processing='upstream', projection_seed=2026, test_access=False,
        feature_source='fixed random projection + softmax replacing the quantum circuit',
        comparators={'QT16': str(QT_EXTENDED), 'QT106': str(QT_DEPTH)},
        status='running')
    root = Path(output) / ('paper-classical-' + uuid.uuid4().hex[:10])
    root.mkdir(parents=True)
    write_json(root / 'protocol.json', protocol)
    write_json(root / 'indices.json', data['indices'])
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for seed in protocol['seeds']:
            for condition, latent in CONDITIONS.items():
                folder = root / f'{condition}-seed{seed}'
                folder.mkdir(exist_ok=True)
                resume = (folder / 'latest.pt').exists()
                print(condition, seed, flush=True)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    model = ClassicalRandomFeatureTrain(latent)
                    write_json(folder / ('resume-environment.json' if resume else 'environment.json'),
                               environment(validate_config({'seed': seed})))
                    write_json(folder / 'config.json', protocol | {'seed': seed, 'condition': condition, 'latent': latent})
                    result = train_updates(model, {k: data[k] for k in ['train', 'validation']}, folder,
                        updates=protocol['updates'], eval_every=protocol['eval_every'], seed=seed,
                        milestones=protocol['milestones'], resume=resume)
                    exported = model.export_classical()
                    x = data['train'].tensors[0][:16]
                    with torch.no_grad():
                        torch.testing.assert_close(model(x), exported(x), atol=1e-10, rtol=1e-9)
                        delta = float((model(x) - exported(x)).abs().max())
                    torch.save(exported.state_dict(), folder / 'classical-export.pt')
                    write_json(folder / 'result.json', dict(status='completed', seed=seed,
                        condition=condition, latent=latent,
                        parameters=sum(p.numel() for p in model.parameters()),
                        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                        export_delta=delta, **result))
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


def _load(folder):
    result = json.loads((folder / 'result.json').read_text())
    if result['status'] != 'completed' or result['updates'] != 12000:
        raise ValueError(f'Incomplete control: {folder}')
    return result


def _load_environment(folder):
    for name in ('environment.json', 'resume-environment.json'):
        path = folder / name
        if path.exists():
            return json.loads(path.read_text())
    return None


def report(root, output):
    root, output = Path(root), Path(output)
    protocol = json.loads((root / 'protocol.json').read_text())
    if protocol['smoke']:
        raise ValueError('Smoke run is not comparable to full controls')
    specs = [('CNN', QT_EXTENDED / 'cnn-n10000-seed{}', 6690)]
    specs += [('QT16', QT_EXTENDED / 'qt16_r3-n10000-seed{}', 457)]
    specs += [('QT106', QT_DEPTH / 'qt106-seed{}', 1627)]
    specs += [('classical_rf208', root / 'classical_rf208-seed{}', 457),
              ('classical_rf1378', root / 'classical_rf1378-seed{}', 1627)]
    records, rows = [], []
    for seed in protocol['seeds']:
        for label, pattern, params in specs:
            folder = Path(str(pattern).format(seed))
            result = _load(folder)
            if result['parameters'] != params:
                raise ValueError(f'Parameter mismatch for {label} seed {seed}')
            records.append({'label': label, 'seed': seed, 'source': str(folder),
                            'result': result,
                            'environment': _load_environment(folder)})
            rows.append(dict(model=label, seed=seed, parameters=params, ratio=params / 6690,
                best_update=result['best_update'],
                train_accuracy=result['best']['train']['accuracy'],
                val_accuracy=result['best']['validation']['accuracy'],
                val_loss=result['best']['validation']['loss'],
                final_val_accuracy=result['final']['validation']['accuracy'],
                train_seconds=result['train_seconds']))
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'runs.csv', index=False)
    write_json(output / 'manifest.json', dict(protocol=protocol, runs=records))
    summary = frame.groupby('model').agg(n=('seed', 'count'), parameters=('parameters', 'first'),
        ratio=('ratio', 'first'), val_mean=('val_accuracy', 'mean'), val_std=('val_accuracy', 'std'),
        val_loss=('val_loss', 'mean'), train_seconds=('train_seconds', 'mean'))
    summary.to_csv(output / 'summary.csv')
    text = '# 匹配參數預算的古典特徵對照\n\n' + summary.to_markdown() + '\n\n' + frame.to_markdown(index=False)
    text += ('\n\n只把量子電路換成固定隨機投射＋softmax；latent＋mapping的'
             '可訓練數與對應QT完全相同（457／1627），mapping、資料、最佳化、'
             '評估皆沿用r3。CNN/QT控制引用既有完成study，古典特徵為新run。'
             '未使用official test；不是論文精確復現，也不能單憑此判定量子優勢或無效。\n')
    (output / 'study.md').write_text(text, encoding='utf-8')
    print(summary.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--report')
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    if args.report:
        report(args.report, args.output)
    else:
        run(args.output, args.smoke)
