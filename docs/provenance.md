# 來源與適配紀錄

方法來源：Liu et al., **Quantum-Train: Rethinking Hybrid Quantum-Classical Machine Learning in the Model Compression Perspective**, https://arxiv.org/abs/2405.11304 。

參照 repository：https://github.com/Hon-Hai-Quantum-Computing/QuantumTrain

固定參照 commit：`d5b654116565ae9550e540e021ab0d4160a15c86`。

主要檔案：`appendix_example/CIFAR10/pennylane/tcnnqc_pennylane_mapping_model.ipynb`；補充閱讀 `appendix_example/QTLSTM/QTLSTM_pennylane.ipynb`。

2026-10-07 查閱 GitHub license endpoint 回傳 404；先前檢查的固定版本檔案樹未見 LICENSE。這不是授權使用所有原始碼的證明。因此本專案依公開方法重新撰寫小型模型與評估工具，不直接複製或重新散布上游 notebook，也不替上游指定授權。

## 方法沿用

- H 初始化，交替 RY/RZ 旋轉與相鄰 CZ。
- 量子機率與基底索引組成特徵，經傳統映射網路生成目標網路權重。
- 參照 CIFAR-10 範例的機率平方、tanh、冪次與中心化。
- affine-only mapping，隱藏寬度 40/200/40。

## 本專案適配

- CIFAR-10/CNN 改為 sklearn digits/64→8→10 MLP，共 610 個目標參數。
- 10 qubits、CPU float64、理想機率、Torch backprop；不使用來源 CIFAR-10 示例的 SPSA。
- 新增資料切分管理、梯度診斷、實驗 artifact、完整性檢查、多 seed 報告與繁體中文教材。
- 來源是研究方法，不宣稱本專案重現論文的數值、壓縮比或量子優勢。

此專案是新的作品；不可用其結果推斷 2024 年競賽存檔或遺失版本的訓練正確性。

## 後續階段：MNIST paper-guided reconstruction

- 論文採 CC BY-NC-ND 4.0；本 repo 不重製其圖表或全文，只引用表格數字並重新計算參數比例。
- 比對後發現公開 MNIST notebook 與論文表格不一致（RY/CNOT 對 U3/CU3；mapping activation；評估切分），詳見 [paper-reproduction-audit](paper-reproduction-audit.md)。
- 因此以明列假設重建：r1（RY/CNOT、原始機率）、r2（加入上游全域 weights/bias 中心化）、r3（加入上游機率後處理 tanh/冪次/中心化）。
- 目標模型 64→8→10 CNN 共 6690 參數；QT16=457、QT106=1627。所有結果標示為重建實驗，非精確復現。
- 完整階段清單見 [研究索引](research-index.md)；授權與重用範圍見 [NOTICE](../NOTICE.md)。
