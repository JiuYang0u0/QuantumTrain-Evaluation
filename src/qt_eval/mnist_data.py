"""Read official training IDX files only; verify checksums before decoding."""
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


def load_mnist_pilot(root,train_size=1000,val_size=200):
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
    pool,validation=train_test_split(np.arange(60000),test_size=6000,random_state=2026,stratify=targets)
    train,_=train_test_split(pool,train_size=train_size,random_state=2026,stratify=targets[pool])
    val,_=train_test_split(validation,train_size=val_size,random_state=2026,stratify=targets[validation])
    pixels=np.frombuffer(images,dtype=np.uint8,offset=16).reshape(60000,1,28,28)
    result={'indices':{'train':train.tolist(),'validation':val.tolist()},'hashes':hashes}
    for key,index in [('train',train),('validation',val)]:
        result[key]=TensorDataset(torch.tensor(pixels[index].copy(),dtype=torch.float64)/127.5-1,
                                 torch.tensor(targets[index].copy(),dtype=torch.long))
    return result
