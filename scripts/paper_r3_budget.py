"""Freeze r3 and increase only epochs, with fixed-budget milestone observations."""
import argparse
from paper_r2_learning_pilot import run,report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke',action='store_true');p.add_argument('--report')
    p.add_argument('--output',default='results');p.add_argument('--data',default='data/mnist')
    a=p.parse_args()
    if a.report:report(a.report,a.output)
    else:run(a.output,a.data,a.smoke,extended_r3=True)
