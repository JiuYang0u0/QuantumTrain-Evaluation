"""Read official training IDX files only; verify checksums before decoding.

The official 10,000-image test set is never downloaded or read here.
"""
import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

FILES={'train-images-idx3-ubyte.gz':'f68b3c2dcbeaaa9fbdd348bbdeb94873',
       'train-labels-idx1-ubyte.gz':'d53e105ee54ea40749a09fcbcd1e9432'}


def _read_train(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    raw={};hashes={}
    for name,expected in FILES.items():
        path=root/name
        if not path.exists():
            request=urllib.request.urlopen('https://ossci-datasets.s3.amazonaws.com/mnist/'+name,timeout=120)
            with request:content=request.read()
            if hashlib.md5(content).hexdigest()!=expected:raise ValueError('MNIST checksum mismatch')
            path.write_bytes(content)
        content=path.read_bytes()
        if hashlib.md5(content).hexdigest()!=expected:raise ValueError('MNIST checksum mismatch: '+name)
        hashes[name]=dict(md5=expected,sha256=hashlib.sha256(content).hexdigest())
        raw[name]=gzip.decompress(content)
    images=raw['train-images-idx3-ubyte.gz'];labels=raw['train-labels-idx1-ubyte.gz']
    if struct.unpack('>IIII',images[:16])!=(2051,60000,28,28) or struct.unpack('>II',labels[:8])!=(2049,60000):
        raise ValueError('Unexpected MNIST IDX dimensions')
    if len(images)!=16+60000*784 or len(labels)!=8+60000:raise ValueError('Truncated IDX')
    targets=np.frombuffer(labels,dtype=np.uint8,offset=8)
    pixels=np.frombuffer(images,dtype=np.uint8,offset=16).reshape(60000,1,28,28)
    return pixels,targets,hashes


def _split_pool(targets):
    return train_test_split(np.arange(60000),test_size=6000,random_state=2026,stratify=targets)


def _dataset(pixels,targets,index):
    return TensorDataset(torch.tensor(pixels[index].copy(),dtype=torch.float64)/127.5-1,
                         torch.tensor(targets[index].copy(),dtype=torch.long))


def load_mnist_pilot(root,train_size=1000,val_size=200):
    pixels,targets,hashes=_read_train(root)
    pool,validation=_split_pool(targets)
    train,_=train_test_split(pool,train_size=train_size,random_state=2026,stratify=targets[pool])
    val,_=train_test_split(validation,train_size=val_size,random_state=2026,stratify=targets[validation])
    result={'indices':{'train':train.tolist(),'validation':val.tolist()},'hashes':hashes}
    for key,index in [('train',train),('validation',val)]:
        result[key]=_dataset(pixels,targets,index)
    return result


def load_mnist_full(root,val_size=6000):
    """Full-training-set coverage: ~54000 train and a train-carved validation.

    The official test set is intentionally not read. Both splits come from the
    official 60000-image training file.
    """
    pixels,targets,hashes=_read_train(root)
    pool,validation=_split_pool(targets)
    if not 0<val_size<=len(validation):raise ValueError('val_size out of range')
    val=validation[:val_size]
    result={'indices':{'train':pool.tolist(),'validation':val.tolist()},'hashes':hashes}
    for key,index in [('train',pool),('validation',val)]:
        result[key]=_dataset(pixels,targets,index)
    return result
