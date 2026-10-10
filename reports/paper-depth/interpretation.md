# MNIST壓縮深度對照：增加參數明顯換回精度，但成本與變異同步上升

完成 study：paper-depth-3f729b7257，QT106三seed全部完成。CNN/QT16引用既有完成控制paper-data-extended-d206b2e56a。固定10000train/200validation、12000updates（1536000次樣本處理）、batch128有放回、LR1e-4、eval每100。未使用official test，屬r3 paper-guided reconstruction，不是論文精確復現。

## 同一best-validation-loss checkpoint

| 模型 | 可訓練參數 | 相對CNN比例 | 參數減少 | 平均train accuracy | 平均validation accuracy ± 樣本SD | 平均validation loss | 平均train-only秒數 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 6690 | 100% | 0% | 98.66% | 96.33% ± .29pp | .08683 | 323.74 |
| QT106 r3 | 1627 | 24.32% | 75.68% | 90.60% | 90.00% ± 9.18pp | .32945 | 13691.33 |
| QT16 r3 | 457 | 6.83% | 93.17% | 74.95% | 76.00% ± 3.77pp | .78107 | 2505.97 |

參數量增加208→1378（量子）＋249（mapping），使平均validation由76.0%升到90.0%，把對CNN的差距由20.33pp縮到6.33pp。這直接支持「多保留可訓練參數可換回精度」的壓縮—精度取捨。

## 逐seed與固定節點

QT106 best val：seed11 94.0%（update11500）、seed22 79.5%（12000）、seed33 96.5%（11800）。同協定QT16為79.5/72.0/76.5%。QT106三個seed都高於對應QT16，但seed22仍只有79.5%、標準差9.18pp；平均提升伴隨更大seed變異。

固定節點平均validation：

| Updates | CNN | QT16 | QT106 |
|---|---:|---:|---:|
| 4000 | 94.17% | 39.83% | 80.33% |
| 8000 | 95.33% | 68.00% | 86.50% |
| 12000 | 95.67% | 75.83% | 89.17% |

固定節點也呈現同方向（QT106>QT16），故改善不是只靠更多best選模機會。QT106在4000updates已達80.33%，比QT16同時點高40.5pp，學習速度更快。

## 成本

本機train-only：QT106平均約13691秒，QT16約2506秒，CNN約324秒。約5.5×QT16、42×CNN，但同updates非等計算預算。深電路增加參數、梯度與模擬成本；壓縮收益不能只看參數量，需同時報時間與資源。

## 對論文與主張的界線

QT106本次best val平均90.0%接近論文QT106的93.81%，但**不能稱復現**：本輪10000train/200開發validation/12000updates有放回；論文60000train/10000test/50epochs約23450updates，且電路初態、訓練順序、mapping與初始化仍有假設差異。未讀official test。

使用者最初的問題已較清楚：以較少可訓練參數生成大型模型、並用增加參數量換回精度，在此小資料重建下成立；但「差不多準確率＋極低參數」需指明是哪個參數預算，且目前未達CNN的96%或論文的完整資料test結果。

三seed與反覆使用的validation只供開發性描述；不宣稱統計顯著性或量子優勢。所有QT106匯出classical CNN logits delta0。

## 後續建議（未執行）

若要往論文規模，需固定完整60000train與保留official test的協定（或明列另切validation的代價），並繼續揭露來源假設；QT106每run約3.8小時，完整資料約23450updates成本更高，須先估算。另一條是固定預算下加入傳統壓縮基準（如低秩或剪枝），判斷量子生成是否優於非量子生成。不要把QT106改善外推為量子優勢。

完整表格見study.md、runs.csv；原始6組結果在results/paper-depth-3f729b7257及source控制目錄，manifest.json記錄協定、環境與runs。
