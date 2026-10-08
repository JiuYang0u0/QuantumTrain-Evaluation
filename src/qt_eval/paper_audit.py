"""Arithmetic audit of arXiv:2405.11304v2 Tables 1–3, NOT measured results."""


def dense_parameters(widths):
    return sum(a*b+b for a,b in zip(widths[:-1],widths[1:]))


def compression_rows():
    specifications=[
        ('MNIST',6690,13,[14,4,20,4,1],98.45,
         [16,31,46,61,76,91,106,121],[81.08,84.69,87.78,88.43,90.05,92.41,93.81,93.05]),
        ('FashionMNIST',6690,13,[14,4,20,4,1],88.14,
         [16,31,46,61,76,91,106,121],[65.69,73.48,76.05,74.92,77.31,77.51,80.20,80.74]),
        ('CIFAR10',285226,19,[20,40,200,40,1],62.50,
         [19,95,171,247,323],[28.73,44.76,47.42,56.45,60.69])]
    rows=[]
    for dataset,target,qubits,widths,baseline,depths,accuracies in specifications:
        mapping=dense_parameters(widths)
        for depth,accuracy in zip(depths,accuracies):
            total=mapping+qubits*depth
            rows.append(dict(dataset=dataset,blocks=depth,qubits=qubits,target_parameters=target,
                mapping_parameters=mapping,quantum_parameters=qubits*depth,qt_parameters=total,
                parameter_ratio=total/target,parameter_reduction=1-total/target,
                published_baseline_accuracy=baseline,published_qt_accuracy=accuracy,
                accuracy_gap_pp=baseline-accuracy,source='arXiv:2405.11304v2 Tables 1-3',
                evidence_type='published_not_reproduced'))
    return rows


def public_notebook_count(blocks=16,qubits=13):
    # U3: 3 angles per wire; circular CU3: 3 angles per directed neighboring pair.
    return 6*qubits*blocks+dense_parameters([qubits+1,4,20,4,1])
