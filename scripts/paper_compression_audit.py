"""Export published claims with explicit non-reproduction labels."""
from pathlib import Path
import pandas as pd
from qt_eval.paper_audit import compression_rows,public_notebook_count

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]/'reports/paper-audit'
    root.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(compression_rows())
    frame.to_csv(root/'published-claims.csv',index=False)
    text='# 論文參數與準確率取捨稽核\n\n以下accuracy均引用arXiv:2405.11304v2 Tables1–3，並非本專案測得。參數量由mapping架構＋qubits×blocks重新計算。\n\n'
    text+=frame[['dataset','blocks','target_parameters','mapping_parameters','quantum_parameters','qt_parameters','parameter_ratio','published_baseline_accuracy','published_qt_accuracy','accuracy_gap_pp']].to_markdown(index=False)
    text+=f'\n\n公開MNIST notebook的U3/CU3 QT-16靜態計數為{public_notebook_count()}，不等於論文RY/CNOT的457。詳細差異與未解設定見../../docs/paper-reproduction-audit.md。\n'
    (root/'audit.md').write_text(text,encoding='utf-8')
    print(frame[['dataset','blocks','qt_parameters','parameter_ratio','accuracy_gap_pp']].to_string(index=False))
