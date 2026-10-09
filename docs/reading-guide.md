# Notebook閱讀地圖

## 路線A：新人入門
1. **01的0–3節**：環境、資料切分、610參數MLP。先能解釋shape與logits。
2. **01的4–7節**：2qubit電路、概率到weights、梯度反例、單步更新。
3. **01的8–10節**：訓練/選模/重載，讀懂artifact。
4. **02**：先定研究問題、預算與矩陣，再產生報告。

前置：Python函式、矩陣、基本PyTorch；量子閘以小例子解釋。此路線使用digits，目的為學習工具，不是論文壓縮。

## 路線B：論文貢獻與MNIST重建
1. **05**：引用表格、M與P_QT計數、公開code不一致的地方。
2. **06的0–3節**：6690/457、CNN形狀、梯度與匯出、真MNIST pilot。
3. **06的4–6節**：全域中心化與機率後處理，對照來源而不是任意调參。
4. **06的7–10節**：epochs/updates/資料覆蓋、統一延長、QT16/QT106壓縮曲線。

05/06不會自動啟動長程研究。06的gradient示例使用合成圖片，不產生MNIST accuracy證據。需實驗時另執行各協定命令。

## 路線C：研究工程診斷
讀完01後，可先選擇閱讀03的對應章節：

| 問題 | 03節次 |
|---|---|
| 不會學、預測同一類 | 0–1、3–6 |
| Mapping與target activation | 2、7–8 |
| 學習率、偏移、權重尺度 | 9–11 |
| 量子分支作用、固定參數、傳統baseline | 12–14 |

閱讀可跳節，**執行不可直接跳過setup與前置變數**：使用Restart & Run All的smoke完成整份，或自行先按序跑所需前置cell。再讀04，理解小資料診斷不能直接替代完整任務。

## 操作與時間預算

- 01–04預設smoke；第一次勿改full。02/03/04會真正執行小型訓練。
- 03章節多，按問題閱讀；不要求新人一次理解全部研究歷史。
- 05僅計數與來源稽核；06包含合成資料QNode/backward，不下載MNIST。
- MNIST pilot首次需下載官方train；長跑需另執行腳本，QT106三seed試測外推約10小時train-only，非工期承諾。
- 圖表要先看資料、seed、checkpoint口徑與坐標；本文`best`預设最低validation loss，不是最高accuracy。

## 教材維護對照

| Notebook | 生成來源 |
|---|---|
| 01、02 | `scripts/build_notebooks.py` |
| 03 | `scripts/build_diagnostics_notebook.py` |
| 04 | `scripts/build_full_validation_notebook.py` |
| 05 | `scripts/build_paper_audit_notebook.py` |
| 06 | `scripts/build_paper_mnist_notebook.py` |

例如：`uv run python scripts/build_paper_mnist_notebook.py`，之後 `uv run python scripts/verify_notebooks.py --notebook notebooks/06_paper_mnist_reconstruction.ipynb --mode smoke`。每節需保留目標、理由、shape、操作、預期觀察、陷阱與練習答案。

相關：[進度與恢復](progress-guide.md)、[研究索引](research-index.md)、[驗證紀錄](verification.md)。
