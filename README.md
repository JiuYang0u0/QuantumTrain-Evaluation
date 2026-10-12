# QuantumTrain Evaluation

**量子模型的教學、可重現實驗與參數壓縮評估工具。** 使用 PennyLane／PyTorch，從梯度診斷走到 MNIST 的 *paper-guided reconstruction*（明列假設重建）。

本作品尚未精確重現原論文。早期 digits 實驗是工程診斷，不是壓縮證據；MNIST 重建才使用較少可訓練參數生成較大 CNN。先讀[來源稽核](docs/paper-reproduction-audit.md)，再解讀準確率。

> **單頁總覽：[reports/OVERVIEW.md](reports/OVERVIEW.md)**——目標、已驗證／未驗證、關鍵數字與下一步。
> 若要移到有 GPU 的桌機重跑，先讀 [reports/gpu-portability.md](reports/gpu-portability.md)。

## 五分鐘開始

在專案根目錄的 PowerShell 執行：

```powershell
uv sync --frozen
uv run jupyter lab
```

選擇本專案 `.venv/Scripts/python.exe` kernel。首次保留 `smoke` 模式，先讀 Markdown 再執行 code，使用 **Restart Kernel and Run All**。Notebook 會印出 Python 路徑供核對。

## 選擇閱讀路線

| 目的 | 閱讀順序 | 完成後能回答 |
|---|---|---|
| 新人入門／使用工具 | **01 → 02** | 權重如何生成？如何執行與保存可重現實驗？ |
| 理解論文壓縮貢獻 | **05 → 06**，不熟梯度先看01 | 參數比例與accuracy gap如何同時驗證？哪些設定是重建假設？ |
| 研究工程／除錯 | **01 → 03 → 04** | 梯度、失活、尺度、消融與傳統baseline如何排查？ |

完整分節、操作成本與前置知識：[**Notebook 閱讀地圖**](docs/reading-guide.md)。

| Notebook | 主題與入口 |
|---|---|
| [01](notebooks/01_quantumtrain_workbook.ipynb) | 從資料、量子機率到權重與梯度；含detach反例 |
| [02](notebooks/02_depth_evaluation.ipynb) | 設計實驗矩陣、估算預算、檢查完整性、產生報告 |
| [03](notebooks/03_training_diagnostics.ipynb) | 15節診斷案例；可依問題分段閱讀，執行仍需由setup順序開始 |
| [04](notebooks/04_full_validation.ipynb) | digits完整train的固定協定比較 |
| [05](notebooks/05_paper_compression_audit.ipynb) | 論文表格與公開程式差異；accuracy均為引用 |
| [06](notebooks/06_paper_mnist_reconstruction.ipynb) | MNIST r1/r2/r3、更新預算、資料量及QT106取捨 |

01–04預設會跑小型smoke；full成本依各自協定不同，不能全部理解為相同epochs。05不訓練。06以合成資料做正確性示例，長程MNIST命令只展示，不會因Run All自動跑QT106。

## 查詢長跑進度

```powershell
uv run python -m qt_eval.progress --list
uv run python -m qt_eval.progress results/<study目錄>
uv run python -m qt_eval.progress results/<study目錄> --json
```

一次查詢會列出保存步數、最後寫入時間、完成數、best/final validation、checkpoint是否存在及程序證據。**`running`檔案不等於程序存活，程序存在也不等於持續前進。** 一般啟動命令未帶study目錄時，只能列為候選程序，不會冒稱確認。

狀態含義、支援範圍與恢復方法：[進度工具說明](docs/progress-guide.md)／[checkpoint續訓](docs/checkpoint-recovery.md)。工具是唯讀快照，不自動重啟或載入 `.pt`。

## 研究結果入口

| 系列 | 能支持什麼 | 入口 |
|---|---|---|
| digits十輪診斷 | 工程正確性、特徵與優化行為；target610小於generator16791，非壓縮驗證 | [研究總結](docs/research-summary.md) |
| digits完整資料 | 當時固定預算下四組生成器／MLP的開發性比較 | [結果](reports/full-validation/interpretation.md) |
| 論文稽核 | 原文參數—準確率取捨及來源差異，不是本機重現 | [表格](reports/paper-audit/audit.md) |
| MNIST r3，10000筆／12000updates | QT16為457/6690=6.83%參數，best-validation平均76.00%；CNN96.33%，尚差20.33pp | [結果](reports/paper-data-extended/interpretation.md) |
| QT106，10000筆／12000updates | 1627/6690=24.32%參數，best-validation平均90.00%；較QT16明顯換回精度（差距20.33pp→6.33pp），但成本約5.5×、seed變異更大 | [結果](reports/paper-depth/interpretation.md) |
| 匹配預算古典特徵 | 固定投射＋softmax取代量子電路，可訓練數與QT16/QT106相同（457／1627）：量子版每個節點與seed都較高（+33.0／+37.5pp）；量子版慢得多 | [結果](reports/paper-classical/interpretation.md) |
| 低秩直接訓練基準 | 更強古典壓縮：rank-r 直接訓練目標CNN權重，r=1/2/3/4/6（537–2972參數）。極低預算量子大幅領先（457預算76.0%對約32%）；約1627時QT106 90.0%對低秩約86.4%，低秩需約1998才達90%。量子優勢隨預算縮小 | [結果](reports/paper-lowrank/interpretation.md) |
| 完整MNIST覆蓋pilot | 54000 train／6000 validation（官方test未讀），相同12000更新、單seed：CNN97.7%、QT106 93.9%（24.32%參數）、低秩r=4 91.1%、低秩r=3 86.1%、QT16 77.9%。量子參數效率領先，中高預算更明顯 | [結果](reports/paper-full-data/interpretation.md) |
| 官方test一次性凍結評估 | 訓練後單次評估：CNN97.95%、QT106 93.62%（24.32%）、低秩r=4 91.34%（29.87%）、r=3 86.30%、QT16 78.77%。與validation相差0.3–0.9pp；QT106以較少參數超過低秩r=4 | [結果](reports/final-test/interpretation.md) |
| **50 epochs 完整資料** | 54,000 train／6,000 validation、23,450 updates。官方test：CNN98.43%、QT106 96.08%（24.32%參數，低2.35pp）、低秩r=4 93.68%（29.87%）、r=3 89.14%、QT16 85.89%（6.83%）、r=2 71.90%、r=1 34.22%。同預算下量子明顯領先低秩 | [評估](reports/final-test-50ep/interpretation.md) · [壓縮曲線](reports/final-test-50ep/compression-curve.md) |

更多：[單頁總覽](reports/OVERVIEW.md)。

更多階段：[研究與協定索引](docs/research-index.md)。歷史完整入口內容移至[HISTORY](HISTORY.md)。狀態以進度工具與原始結果核對，不以README作即時監控。

## 驗證與維護

```powershell
uv run pytest -q
uv run python scripts/verify_notebooks.py --mode smoke
```

驗證器在fresh kernel執行，不覆寫來源notebook的outputs。教材是生成檔：先修改对应 `scripts/build_*notebook*.py`，再重建；對照表見[閱讀地圖](docs/reading-guide.md)。

目錄：`src/qt_eval/`核心、`scripts/`實驗與教材生成、`docs/`協定、`reports/`已匯出證據、`results/`本機checkpoint與紀錄（Git忽略）。完整來源與重用範圍见[provenance](docs/provenance.md)。

## 解讀界線與常見問題

- 參數少不自動等於訓練更快／記憶體更少；生成CNN仍有完整目標weights。
- Validation已多次用於開發；官方MNIST test目前未作本系列開發評估。不要把validation結果稱為論文test重現。
- `best.pt`是選模結果；只有支援的update-budget runner產生`latest.pt`，才包含完整續訓狀態。
- 找不到`qt_eval`：確認已`uv sync --frozen`，kernel指向本專案環境。
- 找不到configs：從專案根目錄或notebooks啟動，別單獨搬走notebook。
- `grad=None`：讀01的detach案例；部分gradient=0不等同計算圖中斷或barren plateau。
- 單seed標準差N/A是正常；smoke結果不能寫研究結論。
