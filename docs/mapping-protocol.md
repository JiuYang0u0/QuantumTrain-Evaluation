# 第二輪：validation-only 訓練診斷與 Mapping 對照

## 預先設定的比較

- 固定 depth=3、10 qubits、q_delta=.001、CPU float64、Adam lr=.001、batch64。
- affine 與 tanh 保持相同 Linear 權重形狀與初值（同seed）；Tanh只加入前三個Linear之後，不加在輸出層。
- 三 seeds：11/22/33，split seed 2026。
- 情境一：固定訓練集每類最先出現2筆，共20筆，200epochs；檢查能否擬合小資料。
- 情境二：完整train，20epochs；依validation loss選checkpoint。
- 共12 runs；不呼叫test loader，也不報新的test結果。
- Smoke每split40筆、單seed、兩情境各2epochs；只檢查流程。

## 新增觀察

固定training probe（當前訓練集前40筆，tiny為20筆），記錄初始化、每epoch與最佳checkpoint的：

- 610個生成權重的min/max/mean/std。
- logits分布，以及同一類logit在不同圖片之間的標準差。
- 10類prediction counts。
- ReLU零輸出比例、8個hidden units各自的active fraction。
- 每epoch各batch量子／mapping梯度L2 norm平均。
- 每epoch首尾的量子／mapping參數最大絕對差。

snapshot為no_grad觀察，不更新參數、不消耗RNG。每epoch的train accuracy重新評估，與第一輪的訓練中online loss不同。

## 判讀

小資料的最高train accuracy用於檢查擬合能力；不能當泛化結果。best_train/best_validation屬於validation loss選到的checkpoint，final與best observation分開保存。

有梯度但學不會可能涉及表示能力、尺度、初始化或optimizer；零梯度不能直接命名為barren plateau。非線性mapping改善也不能證明量子部分有增量效益。

## 使用

```powershell
uv run python -m qt_eval.training_diagnostics
uv run python -m qt_eval.training_diagnostics --full
uv run python scripts/report_mapping.py results/<實際mapping-study目錄> --output reports/mapping
```

每次執行建立唯一study目錄；中斷時已完成run與history保留，重新執行開新study。來源方法沿用範圍見 provenance.md；Tanh屬本專案延伸，不是上游原設計。
