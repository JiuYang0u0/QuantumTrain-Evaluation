import pytest


def test_published_parameter_tables_match_architecture_counts():
    from qt_eval.paper_audit import compression_rows
    rows=compression_rows()
    mnist=[r for r in rows if r['dataset']=='MNIST']
    assert mnist[0]['target_parameters']==6690
    assert mnist[0]['mapping_parameters']==249
    assert mnist[0]['qt_parameters']==457
    assert mnist[6]['qt_parameters']==1627
    cifar=[r for r in rows if r['dataset']=='CIFAR10']
    assert cifar[-1]['qt_parameters']==23258
    assert cifar[-1]['parameter_ratio']==pytest.approx(23258/285226)
    assert cifar[-1]['accuracy_gap_pp']==pytest.approx(1.81)


def test_public_notebook_is_not_table_one_qt16():
    from qt_eval.paper_audit import public_notebook_count
    assert public_notebook_count()==1497
    assert public_notebook_count()!=457
