# 第八輪：本設定未見量子參數訓練的增量收益

Study `quantum-ablation-11d5b7a1fc`，6/6完成。固定Tanh mapping、LeakyReLU(.01)、center_weights、depth3、LR1e-3、20筆train、200epochs、seeds11/22/33。同seed兩組原始參數相同，唯一訓練差異是量子參數是否更新。Frozen每epoch量子delta皆0，mapping有更新。未使用test。

## 配對結果

| seed | 可訓練：最高/最後train | 固定：最高/最後train | 兩組best-checkpoint validation accuracy |
|---|---|---|---|
| 11 | 65% / 60% | 65% / 65% | 14.48% / 14.48% |
| 22 | 45% / 30% | 45% / 40% | 15.88% / 15.88% |
| 33 | 70% / 65% | 75% / 70% | 17.27% / 17.27% |

平均最後train：可訓練51.67%，固定58.33%。最高train平均60.00%與61.67%。最佳validation epochs兩組皆12/16/8，validation accuracy逐seed完全相同，validation loss僅約1e-5差異。

本設定沒有量子參數訓練帶來收益的證據；固定組最後train較高，但三seed、低表現與有限epoch不足以宣稱固定策略一般較好。可訓練控制組重現第六輪中心化結果。

## 敏感度：連線存在，但輸出作用很小

初始化處理後的量子feature標準差約1.03e-4至1.24e-4；bit特徵為-1/+1，其尺度較大。特徵數值尺度不同是線索，不足以單獨證明因果，mapping可以透過權重重新縮放。

在一個固定單位方向、總L2擾動.1 radians下，初始化量子機率L2差約.00282，生成weights相對變化約.000457至.001073（0.046%至0.107%），logits相對變化約.000186至.001800（0.019%至0.180%）。非零敏感度與梯度支持計算圖確實連接；單方向不代表所有方向或完整Jacobian。

在可訓練組的best-validation checkpoints，將量子feature直接置零而不重訓，生成weights的相對變化約0.0043%至0.0191%，train probe logits相對變化約0.0096%至0.0537%，loss差約-2.56e-5至1.56e-5。這支持在這些早期選到的checkpoint，量子feature對前向輸出的作用很小。

這不是整段訓練的全局結論：敏感度只測初始化及best-validation checkpoint；微小早期差異仍可能在後續優化中放大，末epoch兩組train差異不能忽略。也沒有對feature置零後重新訓練。

## 結論與下一步候選

目前權重生成較像主要依賴bit編碼的傳統mapping，量子feature通路雖可微分，但在所測checkpoint上的影響弱。這比繼續盲目增加電路深度更值得追查。

下一輪可固定當前架構，只比較原始量子feature與固定常數倍率放大（倍率事前固定，不從test選）；同時保留固定/可訓練量子參數配對，避免把特徵尺度改善誤稱量子學習收益。需監測梯度、activation飽和及權重變動，因放大也會改變優化幾何。此建議尚未執行。

若要更廣地評估量子機制，仍需重新訓練的zero/random/classical feature對照。Frozen仍使用量子機率，不能等同無量子模型。本例仍未穩定記住20筆，不推論一般QuantumTrain或QML無效，也沒有量子優勢證據。

## 追溯

完整表格與曲線見study.md，逐epoch、初始化/最佳checkpoint敏感度及環境見manifest.json；原始checkpoint保存於results/quantum-ablation-11d5b7a1fc。核心/協定commit58cd77a後啟動，背景期間教材有更新，working_tree按實況保存。
