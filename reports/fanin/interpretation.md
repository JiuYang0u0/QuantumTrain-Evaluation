# 第七輪：持續fan-in縮放降低擬合，驗證結果混合

Study `fanin-104276cae2`，6/6 runs完成。固定Tanh mapping、LeakyReLU(.01)、depth3、LR1e-3、20筆train、200epochs、seeds11/22/33。控制組中心化W1/W2，處理組每次forward額外縮放至1/sqrt(3*fan_in)，bias不變。未使用test。

| 指標 | 僅中心化 | 中心化＋fan-in |
|---|---|---|
| 各seed最高train accuracy | 65%、45%、70% | 30%、30%、25% |
| 平均最高train accuracy | 60.00% | 28.33% |
| 各seed最後train accuracy | 60%、30%、65% | 25%、15%、25% |
| 平均最後train accuracy | 51.67% | 21.67% |
| 各seed最佳validation loss | 2.22214、2.21118、2.17518 | 2.17686、2.25205、2.08740 |
| 平均最佳validation loss | 2.20284 | 2.17210 |
| 平均best-checkpoint validation accuracy | 15.88% | 15.97% |

Validation accuracy取自validation loss最低的checkpoint，不是全程最高accuracy。原始表格、曲線、設定及逐epoch資料見study.md、runs.csv、manifest.json；本機checkpoint保存在results/fanin-104276cae2。

## 結論

持續縮放使三個seed的小資料擬合都變差，未達到本輪「建立可信小資料baseline」的目標。Validation loss兩個seed改善、一個變差；平均loss略好，但accuracy平均僅增加約0.09個百分點，不能稱為穩定分類提升。這可能涉及尺度約束、正則化或優化動態，未單獨證明其原因。

縮放組seed11/33的最佳validation epochs為198/195，接近200上限；因此本輪只支持固定預算比較，不證明各組充分收斂後的優劣。不能將最佳checkpoint的validation結果和最後epoch的train數字假裝成同一模型評估。

這是每次forward的可微分重參數化，並非單純fan-in初始化；它同時約束生成weight尺度及改變梯度幾何。負面結果不能推論所有fan-in初始化都無效。

## 下一步候選（尚未執行）

先保留「中心化、不強制每次縮放」作較能擬合的探索基準。停止繼續疊加超參數，優先量測量子特徵敏感度：在固定模型下檢查改變量子參數是否實質改變生成權重與logits，再以固定/可訓練量子參數的對照定位貢獻。該基準仍未能記住20筆，任何消融都需明確限制在此低表現設定，不用它否定一般量子學習。

另一個獨立問題是mapping索引表示或逐層生成器，應先整理前七輪研究樹再設計新協定；本文件不預設成功方案。

## 追溯

核心與協定commit 0f63870後啟動，背景期間教材更新，environment如實保留working_tree。控制組重現第六輪各seed最高/最後train。完整矩陣經report逐一檢查completed與weight_transform後才匯出；並非中途4/6的暫報。
