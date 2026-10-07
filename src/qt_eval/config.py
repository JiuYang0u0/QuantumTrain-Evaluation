"""Validated, fully expanded experiment settings."""
import json
import math
from pathlib import Path

DEFAULTS = dict(mode="smoke", model="quantum", q_depth=3, q_delta=0.001,
                diff_method="backprop", shots=None, dtype="float64",
                learning_rate=0.001, batch_size=64, epochs=2, seed=11,
                split_seed=2026, torch_threads=1, mapping_activation='affine',target_activation='relu',weight_transform='none')


def validate_config(config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    c = DEFAULTS | config
    if c['weight_transform'] not in ('none','center_weights'):
        raise ValueError('Unsupported weight_transform')
    if c['target_activation'] not in ('relu','leaky_relu'):
        raise ValueError('target_activation must be relu or leaky_relu')
    if c['mapping_activation'] not in ('affine','tanh'):
        raise ValueError('mapping_activation must be affine or tanh')
    for key in ('epochs', 'batch_size', 'torch_threads', 'q_depth'):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('seed', 'split_seed'):
        if type(c[key]) is not int or not 0 <= c[key] < 2**32:
            raise ValueError(f'{key} must be an unsigned 32-bit integer')
    for key in ('learning_rate', 'q_delta'):
        if not isinstance(c[key], (float, int)) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if c['mode'] not in ('smoke', 'full') or c['model'] not in ('classical', 'quantum'):
        raise ValueError('Unsupported mode or model')
    if c['q_depth'] not in (1, 3, 5):
        raise ValueError('Supported study depths: 1, 3, 5')
    if c['shots'] is not None or c['diff_method'] != 'backprop' or c['dtype'] != 'float64':
        raise ValueError('This protocol requires shots=null, backprop, float64')
    return c


def load_config(path):
    return validate_config(json.loads(Path(path).read_text(encoding='utf-8')))
