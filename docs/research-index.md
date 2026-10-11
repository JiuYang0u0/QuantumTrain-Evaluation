# 研究索引：不要混用不同任務的結果

## 早期digits診斷
主線與十輪表格：[research-summary](research-summary.md)。這些target610/generator16791配置不是論文壓縮。

報告：
- [最初深度](../reports/interpretation.md)
- [Mapping](../reports/mapping/interpretation.md)
- [Target activation](../reports/target-activation/interpretation.md)
- [Leaky mapping](../reports/leaky-mapping/interpretation.md)
- [Learning rate](../reports/learning-rate/interpretation.md)
- [中心化](../reports/centering/interpretation.md)
- [Fan-in](../reports/fanin/interpretation.md)
- [量子消融](../reports/quantum-ablation/interpretation.md)
- [特徵尺度](../reports/feature-scale/interpretation.md)
- [傳統特徵](../reports/classical-features/interpretation.md)
- [完整digits](../reports/full-validation/interpretation.md)

## 論文主張與MNIST重建
起點：[來源稽核](paper-reproduction-audit.md)／[原文引用表格](../reports/paper-audit/audit.md)。

| 階段 | 固定協定 | 實測解讀 |
|---|---|---|
| r1基礎 | [設定](paper-mnist-protocol.md) | [短測](../reports/paper-mnist-pilot/interpretation.md) |
| r2全域中心化 | [設定](paper-centering-protocol.md) | [短測](../reports/paper-centering/interpretation.md) |
| r2學習預算 | [設定](paper-r2-learning-protocol.md) | [50epochs](../reports/paper-r2-learning/interpretation.md) |
| r3機率處理 | [設定](paper-r3-protocol.md) | [配對](../reports/paper-r3/interpretation.md) |
| r3小資料延長 | [設定](paper-r3-budget-protocol.md) | [500epochs](../reports/paper-r3-budget/interpretation.md) |
| 資料覆蓋 | [設定](paper-data-budget-protocol.md) | [1000/10000](../reports/paper-data-budget/interpretation.md) |
| 固定10000延長 | [設定](paper-data-extended-protocol.md) | [12000updates](../reports/paper-data-extended/interpretation.md) |
| QT106深度 | [設定](paper-depth-protocol.md) | [12000updates](../reports/paper-depth/interpretation.md) |
| 匹配預算古典特徵 | [設定](paper-classical-protocol.md) | [結果](../reports/paper-classical/interpretation.md) |
| 低秩直接訓練基準 | [設定](paper-lowrank-protocol.md) | [結果](../reports/paper-lowrank/interpretation.md) |
| 完整MNIST覆蓋pilot | [設定](paper-full-data-protocol.md) | [結果](../reports/paper-full-data/interpretation.md) |
| 官方test凍結評估 | [設定](paper-final-test-protocol.md) | [結果](../reports/final-test/interpretation.md) |

## 證據層次
原文表格=published；本機小型短測=pilot；反覆validation選設定=exploratory；待凍結方法後的未參與開發評估才能提供更獨立的證據。參數比、生成器與匯出CNN一致性、任務accuracy分別驗證。

完整舊入口文字见[HISTORY](../HISTORY.md)。
