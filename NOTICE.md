# NOTICE 與使用範圍

本 repository 是獨立的教學／研究軟體作品，不是原論文或上游程式碼的官方發行。

## 參考來源

- 論文：Liu, Kuo, Lin, Young, Chang, Hsieh, Goan, *Quantum-Train: Rethinking Hybrid Quantum-Classical Machine Learning in the Model Compression Perspective*, arXiv:2405.11304。本文用於方法描述與表格數字的引用。論文採 CC BY-NC-ND 4.0；本 repository **不重製、不重新散布**該論文或其圖表。
- 上游 repository：[Hon-Hai-Quantum-Computing/QuantumTrain](https://github.com/Hon-Hai-Quantum-Computing/QuantumTrain)，固定參照 commit `d5b654116565ae9550e540e021ab0d4160a15c86`。查核時未見 LICENSE；本 repository **未複製或重新散布**上游程式碼，而是依公開方法重新實作，並記錄實際差異。

## 標示慣例

- 標為 `published` 的準確率與參數為**引用原論文**，不是本機實測。
- 標為 `pilot`、`exploratory`、`paper-guided reconstruction` 的結果為本專案實驗，帶有明列假設，**不代表精確論文重現**。
- 本研究未使用官方 MNIST test 作系列開發評估；驗證集已多次參與開發，非獨立盲測。

## 未隨 repository 發行的內容

- `results/`：本機 checkpoint、逐次訓練紀錄（Git 忽略）。
- `data/`：下載的 MNIST IDX 檔（Git 忽略）。
- `.venv/`：本機虛擬環境（Git 忽略）。

## 無擔保

本軟體與實驗結果僅供研究與教學，不構成量子優勢、部署效能或相容性的保證。詳細來源、重用範圍與方法差異見 [docs/provenance.md](docs/provenance.md) 與 [docs/paper-reproduction-audit.md](docs/paper-reproduction-audit.md)。
