# 匹配參數預算的古典特徵對照（r3）

問題：目前 QT16／QT106 的壓縮—精度結果，是量子電路的特性，還是任何同預算的小型生成器都能做到？

## 唯一變因：特徵來源

其餘完全沿用 r3 MNIST paper-guided reconstruction：

| 項目 | QT（對照） | 古典特徵（本輪） |
|---|---|---|
| 特徵來源 | 13 qubits RY/CNOT 電路 → 機率 | 固定隨機投射 + softmax → 機率 |
| 可訓練 latent | theta（208／1378） | z（208／1378） |
| mapping | 14→4→20→4→1，首層 Tanh，249 參數 | 完全相同 |
| 機率後處理 | r3：(0.8·tanh(0.1·4096·p))^0.3，中心化 | 相同 |
| 生成權重 | 6690，全域中心化 | 相同 |

可訓練參數量精確對齊：**208+249=457（對 QT16）**、**1378+249=1627（對 QT106）**。古典投射矩陣為凍結 buffer，不列入可訓練參數。

## 公平性與界線

- latent 初值同為 uniform(−π, π)，投射以固定 seed 2026 產生並以 1/√d 縮放，使 softmax 機率與量子機率量級相近。
- 這是「換掉量子特徵來源」的消融，**不是**完整的古典壓縮方法（如低秩、剪枝）。
- 古典特徵較便宜（每步約少一個量子模擬），故同 updates 不等於同計算成本；會並列報告時間。
- 結果只用於判斷「量子電路在此預算下是否特別」，不能宣稱量子優勢或無效。

## 協定

10000 train／200 validation（與 QT 控制同索引）、12000 updates、batch128 有放回、LR1e-4、eval 每 100、節點 4000/8000/12000、seeds 11/22/33。未使用 official test。每 run 保存完整 latest.pt，支援中斷續訓。

重跑：`uv run python scripts/paper_classical_budget.py`；`--smoke` 為單 seed 4 updates。報告：`uv run python scripts/paper_classical_budget.py --report results/<study> --output reports/paper-classical`。
