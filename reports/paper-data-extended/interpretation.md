# 固定10000筆延長至12000updates：QT明顯改善，壓縮—精度差距仍在

完成study：paper-data-extended-d206b2e56a，6/6。Recovery沿用五組完成紀錄、QT33從相同初始化重跑；原中斷資料保留。來源鏈見manifest.json recovery與原results目錄。不是從舊10200步精確續訓；QT33本次重跑已具備每100updates完整latest.pt保護。

## 同best-validation-loss checkpoint結果

| 模型 | 可訓練參數 | 參數比例 | 平均train accuracy | 平均validation accuracy ± 樣本SD | 平均validation loss |
|---|---:|---:|---:|---:|---:|
| CNN | 6690 | 100% | 98.66% | 96.33% ± .29pp | .08683 |
| QT16 r3 | 457 | 6.831% | 74.95% | 76.00% ± 3.77pp | .78107 |

參數減少93.169%，平均validation準確率差20.33pp。這是可量化壓縮—精度取捨，但仍不支持「接近CNN準確率」。QT各seed best val79.5/72.0/76.5%；best updates12000/12000/11800。CNN best val96.5/96.5/96.0%，best updates10900/10100/9700。

## 固定節點（避免更多選模機會的混淆）

| Updates | CNN平均validation | QT平均validation |
|---|---:|---:|
| 4000 | 94.17% | 39.83% |
| 8000 | 95.33% | 68.00% |
| 12000 | 95.67% | 75.83% |

QT三seed4000→12000分別40→79.5%、18.5→72%、61→76%。所有seed都改善，seed22追上很多；這支持先前低表現的重要因素是更新預算不足/慢學習，但不能據此認定唯一根因。相同10000資料、抽樣流與架構，4000節點重現前輪。同一val只有200筆且反覆使用；不是獨立泛化或統計顯著性證據。

所有QT匯出CNN logits delta0。各run處理1536000次樣本、實際unique10000。更多best-checkpoint選模機會（120對40）仍需揭露；固定節點也有進步，故改善不僅是挑選更多checkpoint。

## 對論文的界線

QT16本次best val76%雖較接近論文QT16報告81.08%，但不能將差5.08pp稱為復現誤差：本輪10000train/200開發validation、有放回抽樣12000updates；論文60000train/10000test、50epochs約23450updates，還有電路/初始化/mapping等來源假設。未讀official test。

論文更接近CNN accuracy的是QT106（1627參數、93.81%），不是457參數QT16。要回答使用者核心問題，下一階段應預先固定較接近論文的資料/步數與多深度壓縮曲線，先估算QT106資源，並繼續明列paper-guided reconstruction而非精確code/table reproduction。取得作者對應設定仍有價值。

## 後續建議（未執行）

停止將低預算差結果視為模型無效。先整理已對齊/未對齊設定，選擇完整資料QT16或固定預算的QT16/QT106深度對照，不一次改資料與深度再歸因。若做深度比較，報告457/1627對6690的參數比例、accuracy差距與實際成本；不把增加參數的改善當免費收益。

完整表格/曲線見study.md，固定節點见milestones.csv；原始結果results/paper-data-extended-d206b2e56a。尚未啟動新研究。
