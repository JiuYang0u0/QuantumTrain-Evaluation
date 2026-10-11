import json
from pathlib import Path
import torch
from qt_eval.mnist_data import load_mnist_test, load_mnist_pilot, load_mnist_full


def test_official_test_loader_is_isolated_and_checksummed(tmp_path):
    data = load_mnist_test('data/mnist')
    assert len(data['test']) == 10000
    assert data['indices']['test'] == list(range(10000))
    assert float(data['test'].tensors[0].min()) >= -1.0
    assert float(data['test'].tensors[0].max()) <= 1.0
    assert 'md5' in list(data['hashes'].values())[0]


def test_official_test_indices_do_not_overlap_train_split():
    full = load_mnist_test('data/mnist')
    pilot = load_mnist_pilot('data/mnist')
    assert len(full['test']) == 10000
    # Pilot validation comes from the official training file only.
    assert max(pilot['indices']['validation']) < 60000
    assert list(range(10000)) == full['indices']['test']
