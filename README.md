# QuantumTrain Evaluation — 教學型實驗與評估工具

沿用 **PennyLane QuantumTrain 的方法思想**，以小型 digits 分類任务學習權重生成、量子梯度與可信實驗。提供可逐節閱讀的繁體中文 notebook、測試及自動化評估。這是獨立適配作品，不是原論文數值重現。

## 開始操作（Windows PowerShell）

在此專案根目錄執行，需先有 [uv](https://docs.astral.sh/uv/)：

```powershell
uv sync --frozen
uv run jupyter lab
```

使用此專案 `.venv` 的 Python kernel。若在 VS Code 選 kernel，指定 `.venv/Scripts/python.exe`。Notebook 第一節會印出 `sys.executable` 讓你確認。

### 教材閱讀順序

1. [01 QuantumTrain Workbook](notebooks/01_quantumtrain_workbook.ipynb)：資料圖像 → MLP → 小量子電路 → 機率與權重 → 梯度中斷反例 → 訓練 → 重載模型。
2. [02 Depth Evaluation](notebooks/02_depth_evaluation.ipynb)：定義問題 → 固定矩陣 → pilot 預算 → 批次實驗 → 完整性檢查 → 曲線解讀 → 報告。
3. [03 Training Diagnostics](notebooks/03_training_diagnostics.ipynb)：類別塌縮、權重與activation診斷 → 20筆過擬合 → affine/Tanh mapping對照。僅使用train/validation。

第三份教材文字來源是 `scripts/build_diagnostics_notebook.py`，可用 `uv run python scripts/build_diagnostics_notebook.py` 重建。批次診斷使用 `uv run python -m qt_eval.training_diagnostics`（smoke），加 `--full` 則執行固定depth3、三seeds的20筆200epochs與完整train20epochs對照。此輪不使用test選模或評估；初始Linear與量子參數在同seed下完全對齊，只有mapping隱藏層的Tanh不同。

每一節都有學習目標、設計理由、張量 shape、操作指引、預期觀察、常見錯誤與折疊解答。先閱讀 Markdown，再執行 code。**第一次保留 MODE=smoke，用 Restart Kernel and Run All 從頭執行。**

## Smoke 與 full

| 模式 | 資料 | Seeds / epochs | 用途 |
|---|---|---|---|
| smoke（預設） | 每 split 40 筆 | 11 / 2 | 驗證工程流程，不能寫研究結論 |
| full | 固定完整分層切分 | 11,22,33 / 20 | 4組共12 runs 的適配實驗 |

Notebook 要切 full，可把 setup 的 `MODE = ...` 改成 `MODE = 'full'` 後重啟 kernel。批次操作也可使用：

```powershell
uv run python -m qt_eval.study --config configs/smoke.json
uv run python -m qt_eval.study --config configs/baseline.json --pilot
uv run python -m qt_eval.study --config configs/depth_study.json
```

先執行 validation-only pilot 了解本機耗時，才執行完整矩陣。每次建立新的 `results/study-*/`，其中 `study.json` 列出所屬 runs，`report/depth-study.md` 為摘要。失敗 run 保留 error.txt，批次仍列出不完整組別，命令最後以非零退出碼回報失敗。

`configs/baseline.json` 是短程 full-data pilot 設定，不是正式 20-epoch 報告。程式的實驗矩陣由 `study_configs()` 明確列出，配置檔是每次 run 的共同設定。

## 驗證

```powershell
uv run pytest -q
uv run python scripts/verify_notebooks.py --mode smoke
```

驗證器在 fresh kernels 執行教材，產物使用隔離暫存目錄，不改 notebook 的 outputs/execution counts。pytest 驗證資料互斥、functional/module 等價、有限差分、detach 診斷、checkpoint 選取、失敗保存和完整性統計。

教材原始文字／程式可在 `scripts/build_notebooks.py` 維護，修改後執行 `uv run python scripts/build_notebooks.py` 重新生成兩份教材，再做 smoke 驗證。

## 理解結果

第二輪診斷12/12次完成：Tanh未改善，最後epoch固定train probe的hidden ReLU皆失活。詳見 [mapping報告](reports/mapping/mapping-study.md) 與 [診斷解讀](reports/mapping/interpretation.md)。這是validation-only結果，與第一輪test指標不同。

已完成第一輪12次正式實驗，結果見 [報告](reports/depth-study.md) 與 [解讀](reports/interpretation.md)。這組適配設定中，Classical 平均74.26%，QT約10.2%；量子更新檢查通過並不代表有分類收益。

- 目標 MLP 64→8→10，共610個參數；QT mapping 16,761個、量子10/30/50個，因此本小例子**沒有参数壓縮優勢**。
- 理想模擬、CPU float64、backprop，並未量測 QPU 性能。
- 最佳 checkpoint 由 validation loss 選擇，test 僅最後評估。
- 三 seeds 的平均與樣本標準差是描述統計，不是顯著性或量子優勢證明。
- 深度同時改變參數量；研究範圍與計時定義見 [methodology](docs/methodology.md)。
- 上游引用、授權檢查與實際改動見 [provenance](docs/provenance.md)。

## 常見問題

| 現象 | 檢查方式 |
|---|---|
| 找不到 qt_eval | 從根目錄執行 uv sync；檢查 notebook 印出的 Python 是否位於 .venv |
| Notebook 找不到 configs | 從專案根目錄或 notebooks/ 啟動；不要把單一 notebook 搬到其他目錄 |
| dtype 不一致 | 第一版使用 float64；torch.tensor 的教學例子要明確設定 dtype |
| 量子 grad=None | 查看中途是否用了 detach、numpy 或重新包 tensor；重跑第6節反例與診斷 |
| 某些 gradient=0 | 可能因對角閘或對稱性；看第4節，不直接認定是 barren plateau |
| 訓練很慢 | 確認 MODE、epochs、矩陣數；先 smoke/pilot，固定 threads=1，記錄本機耗時 |
| 只有一筆卻有 N/A | 單 seed 無法估計跨 seed 標準差，這是正確顯示 |

`best.pt` 支援重載評估，不提供精確中斷續訓。`results/` 預設 Git 忽略；發布報告時附 manifest/圖表，保存本機 checkpoint 與原始 metrics 以供查核。
