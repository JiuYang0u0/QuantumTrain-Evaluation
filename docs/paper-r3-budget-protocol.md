# 固定r3，只延長更新預算

兩組direct CNN/QT16 r3，seeds11/22/33；同1000train/200validation索引。從頭初始化，50→500epochs，batch128（8updates/epoch），共4000updates。LR1e-4、Adam eps1e-8/weight_decay0/無scheduler、CPU float64/threads1、seed、所有r3架構/機率處理/全域中心化均保持不變。

第50/100/200/500epochs各保存當下checkpoint與同checkpoint train/validation指標，另保存全程最低validation loss的best.pt。節點評估以fork_rng隔離，不影響訓練RNG。逐epoch保留最後batch梯度norm與epoch首尾delta。既有r3的50epoch結果是重跑對照，不作新一個獨立seed。

目標：固定資料量下觀察更多更新是否改善學習。預先固定500epochs，不依個別seed的val結果追加預算。每組QT仍457/6690=6.831%；報告accuracy gap和参数比，不只報最高值。

不使用official test，不是論文60000train×50epochs。更多epoch仍只覆蓋1000筆，可能過擬合；best-validation選模與當下節點分開報。仍有重建假設，不能拿小資料accuracy直接否定/證實原表。

重跑：uv run python scripts/paper_r3_budget.py。--smoke是單seed2epochs，milestone為2。報告：--report results/<study> --output reports/paper-r3-budget。六組CPU正式研究可能需數十分鐘至數小時，受本機負載影響，背景執行；不自動擴至完整資料或更深電路。
