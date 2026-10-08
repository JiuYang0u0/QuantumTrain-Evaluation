# r2固定架構的較長MNIST pilot

目標：區分數值尺度恢復與真正開始學習。保留r2全部假設，只延長從2到50epochs；1000train/200validation索引固定，seeds11/22/33，各direct CNN與QT16，共6runs。每epoch8次更新，50epochs400次；不是論文60000train×50epochs約23450次更新。

QT16：13qubits/RY-CNOT16blocks、mapping14→4→20→4→1，僅首層Tanh；raw p與0/1bits；生成6690weights/bias全域減平均；457獨立可訓練參數。Theta uniform[-pi,pi]，Adam lr1e-4/eps1e-8/weight_decay0、無scheduler，batch128、CPU float64。直接CNN保持相同target架構及其預設初始化。

記錄每epoch train loss/validation、最後batch各群梯度L2 norm、epoch首尾最大參數變化。先驗指標：loss是否下降、是否finite、量子/mapping是否持續更新、同checkpoint的train/validation accuracy及loss、參數比、匯出CNN logits一致性。梯度非零不代表有泛化效益。

選最低validation loss checkpoint，final另存；只用官方train的固定子集，official test不下載不使用。可報在此pilot準確率差距，不與論文81.08%直接比作重現成功/失敗。此輪不改其他超參數，也不替特定seed增加epoch。

完整50epochs/60000train及QT106未啟動。先看此pilot的可學習性與每step成本再決定正式預算。命令：uv run python scripts/paper_r2_learning_pilot.py；加--smoke維持2epochs、單seed，但同1000/200資料。報告用--report results/<study> --output reports/paper-r2-learning。
