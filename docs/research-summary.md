# 十輪探索：从能執行到有可信對照

這份作品的主張是建立可重現的混合量子ML診斷流程，並不主張量子優勢。所有數字的設定與限制應回到各輪報告；小資料結果不與完整資料混算。

| 輪次 | 問題 | 觀察與下一步 |
|---|---|---|
| 1 | 原始適配是否有效 | QT完整資料test約10.2%；Classical約74.26%；後續不再使用test調整 |
| 2 | 非線性mapping能否解決 | Tanh單獨未改善；probe的target ReLU全零 |
| 3 | 負區域梯度是否有用 | LeakyReLU恢復輸入差異，但20筆最高train僅20–25%；直接MLP85–95% |
| 4 | 固定LeakyReLU再改mapping | Tanh最高train平均46.67%，仍有後期退化；W1非加性結構增加 |
| 5 | 降LR是否稳定 | 波動變小但同200epochs擬合未改善；平穩不等於學會 |
| 6 | 權重共同偏移 | 中心化最後train平均51.67%，部分seed退步，validation未可靠改善 |
| 7 | 逐層固定fan-in尺度 | 最後train降至21.67%；不保留此處理作後續基準 |
| 8 | 量子參數是否有貢獻 | Fixed/trainable的best-validation accuracy相同，feature前向敏感度很小 |
| 9 | 量子feature尺度 | 1000×下trainable最後train61.67%對fixed50%；validation未改善 |
| 10 | 傳統feature能否做到 | 同30可訓練參數的傳統feature最後train90%，量子61.67%；尚无量子獨有收益 |

## 已建立的工程證據

固定資料切分、不同seed重跑、有限差分梯度交叉驗證、detach反例、固定參數不更新檢查、逐epoch分布與梯度、checkpoint/版本保存、完整矩陣與不完整結果區分、fresh-kernel教學驗證。

## 累積限制

前十輪是依先前觀察選下一個問題的探索性研究，不能假裝十個獨立確認性實驗。三seed僅描述統計；同seed不代表不同架構同初始化分布。只有生成器mapping可精確配對。20筆最高train只檢查擬合能力；best-validation早期checkpoint與最後train不可混為同一模型。Test在第一輪已查看；validation多次參與架構決策。

## 第二階段的決策

第二階段已完成12runs（full-validation-9c8e7d4048）。開發性validation平均：直接MLP80.22%、傳統生成器58.59%、量子可訓練41.04%、量子固定23.96%；量子學習具有配對增量效用，仍未超越傳統對照。11/12的best epoch在20上限，尚未證明充分收斂。完整解讀見../reports/full-validation/interpretation.md。

停止20筆局部調參，固定四組在完整train/validation上跑20epochs；保留20筆資料研究作診斷證據，不將其排名當完整任務排名。協定見full-validation-protocol.md。此階段也是開發性評估，不是新的盲測或正式量子優勢測試。
