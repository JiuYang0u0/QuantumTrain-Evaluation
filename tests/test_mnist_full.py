import json
from pathlib import Path
import torch
from qt_eval.mnist_data import load_mnist_pilot, load_mnist_full

CONTROL = Path('results/paper-data-extended-d206b2e56a/indices-10000.json')


def test_pilot_is_deterministic_and_matches_control_when_available():
    first = load_mnist_pilot('data/mnist', train_size=10000)
    second = load_mnist_pilot('data/mnist', train_size=10000)
    assert first['indices'] == second['indices']
    assert len(first['train']) == 10000
    if CONTROL.exists():
        assert json.loads(CONTROL.read_text(encoding='utf-8')) == first['indices']


def test_pilot_sizes():
    data = load_mnist_pilot('data/mnist')
    assert len(data['train']) == 1000 and len(data['validation']) == 200


def test_full_split_is_disjoint_stratified_and_uses_only_train_file():
    data = load_mnist_full('data/mnist')
    train = set(data['indices']['train'])
    val = set(data['indices']['validation'])
    assert len(train) == 54000 and len(val) == 6000
    assert not train & val
    assert len(train | val) == 60000
    labels = torch.cat([data[key].tensors[1] for key in ('train', 'validation')])
    assert set(labels.tolist()) == set(range(10))
    assert max(train | val) < 60000
