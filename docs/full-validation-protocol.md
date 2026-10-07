# 第二階段：完整資料的凍結開發性評估

四組：direct_mlp、classical_trainable、quantum_trainable、quantum_frozen。Seeds11/22/33，共12runs。沿用split_seed2026，完整train約60%、validation約20%；固定20epochs、batch64、Adam lr=.001、CPU float64、threads1。

Target一律64→8→10、LeakyReLU(.01)。直接MLP有610可訓練參數，使用其預設初始化，不套生成權重中心化。生成器固定Tanh mapping、center_weights；量子10qubits/depth3/q_delta=.001/feature_scale1000。傳統feature保持第十輪610×30固定隨機基底與30係數，初始std匹配量子feature；gain保持固定。所有生成器mapping初始權重同seed一致。

這是把已選架構搬到完整train，不因結果對某組加epoch或調LR。各組同epochs但訓練時間/記憶體不同；保存trainable parameter counts及buffer元素數作基本資源說明，不冒稱完整記憶體量測。Frozen量子機率未快取，計時不代表最佳化部署性能。

fit僅接受train/validation；最低validation loss選best.pt。Best checkpoint同時重測train/validation，最後epoch的train/validation另存。主要結果：best-validation loss及accuracy、多seed均值/樣本SD、final指標、訓練時間。

不使用test，也不把多次參與設計與選模的validation當獨立泛化證據。原test第一輪已被查看；若要確認性研究需後續另訂評估設計。本輪完成後先整理結果，而不是自動開始新調參。

`uv run python scripts/full_validation_study.py` 執行正式矩陣，`--smoke`為每split40筆、單seed2epochs。匯出：`--report results/<study> --output reports/full-validation`。
