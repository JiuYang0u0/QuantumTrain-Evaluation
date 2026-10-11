"""One-shot frozen evaluation on the official MNIST test split.

Training runs never touch this split; this script is the single post-hoc
evaluation for the selected best checkpoints. It also re-derives the
train-carved validation accuracy for reference.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from qt_eval.mnist_data import load_mnist_test, load_mnist_full
from qt_eval.experiment import evaluate
from qt_eval.artifacts import write_json, environment
from paper_full_data import build, CONDITIONS, PARAMETERS, _env


def run(study, output, val_size=6000, batch_size=128):
    root, output = Path(study), Path(output)
    protocol = json.loads((root / 'protocol.json').read_text())
    if protocol.get('smoke'):
        raise ValueError('Smoke studies have no official-test evaluation')
    if protocol.get('official_test_read'):
        raise ValueError('This study already reported official test; refusing to repeat')
    test = load_mnist_test('data/mnist')['test']
    val = load_mnist_full('data/mnist', val_size=protocol['validation_size'])
    if json.loads((root / 'indices.json').read_text()) != val['indices']:
        raise ValueError('Validation indices mismatch with study')
    rows, records = [], []
    for seed in protocol['seeds']:
        for condition in protocol['conditions']:
            folder = root / f'{condition}-seed{seed}'
            result = json.loads((folder / 'result.json').read_text())
            if result['status'] != 'completed':
                raise ValueError(f'Incomplete run: {folder}')
            checkpoint = torch.load(folder / 'best.pt', weights_only=True)
            with torch.random.fork_rng(devices=[]):
                model = build(condition, seed)
                model.load_state_dict(checkpoint['model_state'])
            loaders = {'test': DataLoader(test, batch_size=batch_size),
                       'validation': DataLoader(val['validation'], batch_size=batch_size)}
            scores = {k: evaluate(model, v) for k, v in loaders.items()}
            if condition != 'cnn':
                exported = torch.load(folder / 'classical-export.pt', weights_only=True)
                with torch.no_grad():
                    x, _ = next(iter(loaders['test']))
                    delta = float((model(x) - exported_forward(condition, exported, x)).abs().max())
            else:
                delta = None
            rows.append(dict(condition=condition, seed=seed, parameters=result['parameters'],
                ratio=result['parameters'] / 6690, best_update=checkpoint['update'],
                validation_accuracy=scores['validation']['accuracy'],
                official_test_accuracy=scores['test']['accuracy'],
                official_test_loss=scores['test']['loss'],
                train_seconds=result['train_seconds']))
            records.append({'condition': condition, 'seed': seed, 'source': str(folder),
                            'scores': scores, 'export_delta': delta, 'environment': _env(folder)})
            print(condition, seed, 'official test', round(scores['test']['accuracy'], 4), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'final-test.csv', index=False)
    write_json(output / 'manifest.json', dict(protocol=protocol, official_test_read=True,
        note='single post-hoc evaluation; not used for model selection', runs=records))
    summary = frame.groupby('condition').agg(n=('seed', 'count'), parameters=('parameters', 'first'),
        ratio=('ratio', 'first'), val_mean=('validation_accuracy', 'mean'),
        test_mean=('official_test_accuracy', 'mean'), test_std=('official_test_accuracy', 'std'),
        train_seconds=('train_seconds', 'mean'))
    summary.to_csv(output / 'summary.csv')
    text = '# 官方 MNIST test 一次性凍結評估\n\nStudy: `' + root.name + '`\n\n'
    text += ('此評估在訓練後執行一次，未參與 model selection。官方 test 由不同鏡像提供（gzip 雜湊與經典 MNIST 不同，'
             '但解碼內容已驗證：IDX 標頭、標籤直方圖與像素範圍符合官方 test）。\n\n')
    text += summary.to_markdown() + '\n\n' + frame.to_markdown(index=False)
    text += ('\n\n單 seed 且更新預算非論文 50 epochs（約 23450 updates）時，此結果只是方向確認，不是論文復現。'
             'validation 由官方 train 切出；量子版 CPU 成本遠高於古典基準。\n')
    (output / 'final-test.md').write_text(text, encoding='utf-8')
    print(summary.to_string())
    return output


def exported_forward(condition, state, x):
    from qt_eval.paper_mnist import TargetCNN
    with torch.random.fork_rng(devices=[]):
        model = TargetCNN().double()
    model.load_state_dict(state)
    model.eval()
    return model(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study')
    parser.add_argument('--output', default='reports/final-test')
    args = parser.parse_args()
    run(args.study, args.output)
