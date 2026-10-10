# 完整 MNIST 訓練集覆蓋 pilot

問題：先前都是 10,000 筆訓練資料。論文用完整 60,000 筆。本輪改用完整官方訓練集，確認排名是否改變。

## 資料

- 只讀官方 `train` IDX（60000 張），MD5/SHA256 驗證。
- 分層切分：**54,000 train／6,000 validation**，seed 2026。
- **官方 10,000-image test 完全不讀、不下載**，保留給未來凍結的最終評估。
- validation 由官方 train 切出，不是新資料；此 pilot 仍屬開發性。

## 模型與預算

單 seed（11），與 10,000 筆研究相同更新預算，讓「資料覆蓋」成為主要變因：

| 條件 | 可訓練參數 | 相對 6690 |
|---|---:|---:|
| CNN | 6690 | 100% |
| 低秩 r=3 | 1511 | 22.59% |
| 低秩 r=4 | 1998 | 29.87% |
| QT16 | 457 | 6.83% |
| QT106 | 1627 | 24.32% |

12000 updates、batch128 有放回、LR1e-4、eval 每 500、節點 6000/12000。最終 train 指標用前 2000 筆子集（避免對 54,000 筆做昂貴的全量評估），validation 用全部 6000 筆。

## 解讀界線

- 這是**單 seed**、**12000 updates（非論文約 23450）**、**train-carved validation**、r3 重建設定；不是論文精確復現，也不是官方 test 結果。
- 先跑便宜模型（CNN、低秩），最後才 QT106；即使長跑中斷，古典基準仍可先取得。
- 每 run 具完整 latest.pt，支援 `--resume-root`。
- 計算成本：量子版遠高於低秩與 CNN；量子版是參數效率比較，不是效率優勢。

重跑：`uv run python scripts/paper_full_data.py`；`--smoke` 為單 seed、64 validation。續訓：`--resume-root results/<study>`。報告：`--report results/<study> --output reports/paper-full-data`。
