# 第十輪：傳統30參數特徵更容易擬合，尚無量子優勢證據

Study `classical-features-66658d6e1b`，15/15runs完成。固定20筆train、200epochs、Tanh mapping、LeakyReLU、center_weights、Adam lr=.001、seeds11/22/33。所有組共享同seed mapping初始權重；非零feature初始std匹配1000×量子feature。未使用test。

| 特徵來源 | 額外可訓練特徵參數 | 平均最高train | 平均最後train | 最佳checkpoint平均validation accuracy | 平均最佳validation loss |
|---|---|---|---|---|---|
| 量子可訓練 | 30 | 65.00% | 61.67% | 15.51% | 2.20150 |
| 量子固定 | 0 | 53.33% | 50.00% | 16.25% | 2.20330 |
| 零特徵（保留bits） | 0 | 63.33% | 58.33% | 15.88% | 2.20282 |
| 固定隨機特徵 | 0 | 63.33% | 58.33% | 15.51% | 2.20764 |
| 傳統可訓練特徵 | 30 | 90.00% | 90.00% | 17.27% | 2.16407 |

所有組另有16,761個mapping參數。傳統feature是610×30固定隨機基底乘30個係數，不是610個自由embedding；固定gain只在初始化匹配std。基底另占儲存，參數數量匹配不等於計算/記憶體/表示能力/梯度幾何完全匹配。

## 配對證據

傳統可訓練的最後train各為90%、95%、85%；量子可訓練為65%、60%、60%，各seed差25、35、25個百分點。這支持此設定下傳統生成特徵更容易擬合小資料。

Validation accuracy逐seed：傳統15.60/20.89/15.32%，量子15.32/14.21/16.99%，兩好一壞；平均僅高約1.76個百分點。最佳validation loss三seed皆是傳統較低，但仍屬低accuracy、三seed、反覆探索過的validation。不能稱為可靠泛化或正式優勢證明。

最高/最後train與best-validation metrics來自不同epoch；傳統best epochs為14/26/6，不能把最後90%train和best-validation17.27%假装成同一checkpoint的落差。

Quantum frozen/trainable控制結果與第九輪一致。Zero組能達平均58.33%最後train，支持bit編碼＋mapping本身已有擬合能力；不是沒有任何資訊的常數模型。

## 目前結論

量子通路確實可微分，放大量子feature後能影響權重並改變擬合；但當加入參數量相同、初始std匹配的傳統可訓練特徵，沒有看到量子生成器獨有的擬合收益。這個結論只限於当前10-qubit、depth3、小資料架構及訓練預算，不否定一般QuantumTrain或QML。

## 建議下一阶段（未執行）

先停止局部20筆調參，整理十輪研究樹、控制組及失敗案例。將classical trainable與quantum trainable/frozen、直接MLP作為候選，事前固定完整train/validation比較協定與運算預算；在設計凍結前不要再用原test挑設定。原test在第一輪已查看，因此後續驗證需揭露此事，必要時另選未用過的外部資料作最終確認。

作品可定位為「混合量子模型的可重現診斷與公平對照工具」，證據重點是梯度驗證、失活定位、權重生成限制、尺度敏感度與傳統baseline，而不是量子優勢。

## 保存與範圍

原始checkpoint/history位於results/classical-features-66658d6e1b；study.md與runs.csv保存表格/曲線，manifest.json保存協定、環境與每epoch資料。核心/協定於476518c提交後執行，背景期間教材更新，working_tree按實況記錄。
