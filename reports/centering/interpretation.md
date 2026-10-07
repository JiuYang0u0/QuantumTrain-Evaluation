# 第六輪：中心化改善部分seed的訓練擬合，未帶來可靠泛化收益

Study `centering-7f231401b6`，6/6完成。固定Tanh mapping、LeakyReLU(.01)、depth3、LR1e-3、相同20筆train、200epochs、seeds11/22/33；唯一處理是W1/W2各自減平均，bias不變、不除std、不做fan-in scaling。沒有使用test。

## 結果

| 處理 | 各seed最高train accuracy | 各seed最後train accuracy | 平均best-checkpoint validation accuracy | 平均最佳validation loss |
|---|---|---|---|---|
| 原始 | 45%、50%、45% | 40%、45%、15% | 15.60% | 2.19123 |
| W1/W2中心化 | 65%、45%、70% | 60%、30%、65% | 15.88% | 2.20284 |

平均最高train由46.67%到60.00%，平均最後train由33.33%到51.67%。但seed22反而變差，不能說所有初始化都改善。

Validation accuracy平均僅增加約0.28個百分點；最佳validation loss在三個seed都較原始差。以本輪checkpoint選取的目標而言，沒有泛化改善證據。中心化的best epochs為12/16/8，遠早於200；最後較高train accuracy不屬於最佳驗證checkpoint，兩者不可混用。

## 中間量證據

初始化preactivation平均：

| seed | 原始 | 中心化 |
|---|---|---|
| 11 | -1.28224 | 0.04980 |
| 22 | -3.24349 | 0.03222 |
| 33 | 2.72814 | 0.13712 |

中心化消除了大部分初始化共同偏移；但preactivation標準差仍不同，且訓練後均值仍可能漂移（中心化組最後約-1.44/-5.46/-1.58）。這合理：只有weight矩陣全域均值固定為零，bias、各row均值與非均勻輸入仍影響preactivation。不能稱為activation normalization。

## 研究判讀

共同偏移是可操作、部分有效的因素，但不是全部瓶頸。最好的run可達70%最高train，仍低於同預算直接MLP的85–95%（之前的train-only控制）。後者初始化分布不同，只能作基本可擬合性參考。

目前做的是固定20筆的探索性優化診斷，不可推論完整digits的泛化，更不可宣稱量子優勢。三seed沒有足夠依據挑一個設定當「穩定最佳」。

## 下一輪候選（尚未執行）

在固定中心化與其餘設定下，對照是否以每層fan-in目標尺度重新縮放W1/W2，bias處理保持一致。先記錄原始std與目標std，設計近零std的數值處理和梯度測試，避免把中心化、縮放及bias重置一次混在一起。另一條路是研究mapping的索引表示；需分開實驗。

本輪結束後應整理研究樹與負面結果，而非無限制調參。達到可信的小資料baseline後，再進行固定量子參數消融與完整train實驗。

## 追溯

完整表格/圖：study.md；逐epoch、分層統計、初始化對照、配置和環境：manifest.json；本機原始checkpoint：results/centering-7f231401b6。核心與協定於6b34131提交後執行，期間只更新教材文件。控制組最高/最後train重現第四/五輪相同設定結果。
