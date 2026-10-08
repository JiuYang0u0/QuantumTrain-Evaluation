# 固定MNIST r2的50epoch小資料pilot：可更新，但尚未維持準確率

Study `paper-r2-learning-331edce2ee`，6/6完成。1000train/200validation、batch128、50epochs（400updates）、Adam lr1e-4、三seeds。未使用official test。此為明列假設的paper-guided reconstruction r2，不是原論文完整MNIST復現。

| 模型 | 可訓練參數 | 比例 | 平均train accuracy | 平均validation accuracy ± 樣本SD | 平均validation loss |
|---|---:|---:|---:|---:|---:|
| Classical CNN | 6690 | 100% | 85.47% | 85.00% ± 1.50pp | 0.63817 |
| QT16 r2 | 457 | 6.831% | 16.03% | 16.50% ± 5.22pp | 2.27663 |

各seed validation：CNN83.5/85.0/86.5%，QT22.5/13.0/14.0%。所有best-validation-loss checkpoint均為epoch50，best/final相同。此pilot參數減少93.169%成立，但平均validation accuracy差68.50個百分點，**尚不支持維持接近準確率**。

## 梯度與更新

QT三seed的量子與mapping參數在全部50epochs都有非零epoch首尾delta；最後batch量子梯度norm約1.17e-4、2.91e-6、5.48e-5，mapping約1.80、.242、.853。兩組梯度維度/尺度不同，不用norm大小直接比較重要性。

QT train loss從第一epoch的2.436/2.569/2.324降至最後的2.225/2.305/2.260。數值有限且有些進展，但不是可靠分類能力。梯度接通、參數有更新與任務學會是不同證據。

所有QT匯出classical CNN後logits delta0。這支持生成與部署運算一致性，不支持accuracy主張。

## 不可過度解讀

論文QT16的81.08%是60000train/10000test、約23450updates；本輪只有1000train/200validation、400updates，另有初態、電路排序、mapping與機率處理的重建假設。故不能以本pilot否定論文，也不能稱數值重現成功。Best在預算末端，不證明收斂。

本輪平均train-only時間CNN30.88秒、QT279.02秒；各seed差異很大且背景期間有其他工作，不作嚴格速度比較或直接套用先前快pilot的時間估計。

## 下一步建議（未執行）

先固定來源差異，不回到任意activation/LR調參。最值得核對的是公開MNIST程式的概率後處理：它對p施加tanh、0.3次冪及中心化，而目前r2仍採raw p。可將此處理單獨列為r3，先做同初始化的feature/weights/logits與梯度診斷，再設相同1000/200、50epochs的對照。只改概率欄，保留r2其他假設；如果無法證明該處理對應論文，就繼續標示重建變體。

另一條路是固定r2直接擴大資料/更新預算，以分離資料與步數不足；不可同時擴資料又改前處理後把改善歸因於其中一項。完整50epoch60000train仍未啟動。

協定及結果見manifest.json、study.md、runs.csv；原始checkpoint/updates/metrics在results/paper-r2-learning-331edce2ee。核心協定commit7a0ebd8後啟動，環境working_tree如實保存。
