import importlib.util
from pathlib import Path
import torch


def test_structure_detects_additive_and_nonadditive_weights():
    path=Path(__file__).resolve().parents[1]/'scripts/leaky_mapping_study.py'
    spec=importlib.util.spec_from_file_location('leaky_study',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    additive=torch.arange(8,dtype=torch.float64)[:,None]+torch.arange(64,dtype=torch.float64)[None,:]
    flat=torch.cat([additive.flatten(),torch.zeros(98,dtype=torch.float64)])
    assert module.weight_structure(flat)['nonadditive_relative'] < 1e-12
    flat[0]+=100
    assert module.weight_structure(flat)['nonadditive_relative'] > .01
    assert module.weight_structure(torch.zeros(610,dtype=torch.float64))['nonadditive_relative']==0
