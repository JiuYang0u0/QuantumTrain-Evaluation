# 第八輪：量子特徵敏感度與配對消融

事前固定Tanh、LeakyReLU(.01)、center_weights（不縮放）、depth3、q_delta=.001、Adam lr=.001、20筆train、200epochs、seeds11/22/33。Trainable與Frozen各3runs，初始化與資料順序相同；只改q_params是否requires_grad及納入optimizer。Frozen仍計算初始量子機率，不是純傳統或無量子模型。

先在初始化觀察敏感度，再執行配對訓練；best-validation checkpoint也做相同觀察：
- 固定seed2026生成一個L2正規化方向，theta加epsilon*d，epsilon為.001/.01/.1 radians的總L2擾動。
- 保存probability、weights、logits的L2差值與相對差，以及loss差值。
- 量子機率經處理後的feature直接置零，保留bit編碼及mapping，量測未重訓的干預效果。
- 保存loss對量子參數的梯度norm；Frozen checkpoint的梯度僅為假設解凍時的敏感度，不是訓練時有更新。

主要任務指標：各seed最高/最後train accuracy、best-validation loss與其accuracy。所有training probe固定20筆，未使用test。記錄Frozen每epoch delta=0，量子grad norm=null表示預期不計算，mapping仍須有效更新。

限制：單一方向不代表完整Jacobian或全局敏感度；feature置零不是重新訓練baseline。当前設定仍學不好20筆，消融只能描述此設定，不能證明一般量子方法有效或無效。若觀察到固定與可訓練接近，應再研究特徵幅度、注入方式及純傳統生成器。

命令：`uv run python scripts/quantum_ablation_study.py`，加`--smoke`快速檢查。報告：`uv run python scripts/quantum_ablation_study.py --report results/<study> --output reports/quantum-ablation`。
