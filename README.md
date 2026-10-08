# QuantumTrain Evaluation — 教學型實驗與評估工具

固定r3 [500epoch結果](reports/paper-r3-budget/interpretation.md) 已完成：best-checkpoint validation CNN91.00%、QT53.00%，QT457/6690=6.83%參數。第500節點QT平均51.33%（不同於best53%），比第50節點17.17%明顯進步，仍非論文完整MNIST重現。

固定r3的[更新預算研究](docs/paper-r3-budget-protocol.md)：同1000/200資料，CNN/QT16各三seed跑500epochs（4000updates），在50/100/200/500保存節點。命令 `uv run python scripts/paper_r3_budget.py`；不讀official test，並非完整MNIST重現。

MNIST r3 [配對pilot結果](reports/paper-r3/interpretation.md) 已完成：原始/上游機率後處理的平均validation accuracy為16.50%/17.17%。後處理顯著改變量子feature與梯度尺度，但400updates下accuracy改善僅0.67pp，尚未重現論文準確率。

MNIST r3 [機率前處理協定](docs/paper-r3-protocol.md)：只比較raw p與上游tanh/冪次/中心化，不再平方p；其餘r2固定。命令 `uv run python scripts/paper_r3_pilot.py`，加`--smoke`短測。教材06第6節說明振幅與機率的移植差異。

MNIST r2 [50epoch小資料pilot結果](reports/paper-r2-learning/interpretation.md)：CNN validation85.00%，QT16 16.50%，QT仍457/6690=6.83%參數；量子/mapping每epoch持續更新但準確率維持未驗證。資料與來源假設不等同原論文，不能據此否定原表。

固定MNIST r2的[50epoch小資料學習pilot協定](docs/paper-r2-learning-protocol.md)：1000train/200validation、三seeds、CNN/QT16六runs。`uv run python scripts/paper_r2_learning_pilot.py`。雖50epochs，只有400次更新，不等於論文完整資料50epochs；教材06第5節有說明。

MNIST r2 [尺度診斷與配對pilot](reports/paper-centering/interpretation.md) 已完成：上游全域weights/bias中心化使巨量loss回到約2.3–2.6，仍457參數，純CNN匯出一致。這只解決數值尺度，未重現論文accuracy；教材06已新增來源處理比較。

第三階段已採用明列假設的MNIST QT-16重建：[協定](docs/paper-mnist-protocol.md)、[工具書06](notebooks/06_paper_mnist_reconstruction.ipynb)。Target CNN6690，QT457（6.83%）；參數與梯度/匯出已驗證，準確率需另測。真實MNIST短程pilot：`uv run python scripts/paper_mnist_pilot.py`，只下載train，不讀官方test。不是精確論文復現。

> **範圍修正（2026-10-08）：** 前十輪與digits完整資料比較是小型適配/診斷，不是論文壓縮結果重現。新的第三階段見 [論文與程式稽核](docs/paper-reproduction-audit.md)、[參數取捨表](reports/paper-audit/audit.md) 與 [工具書05](notebooks/05_paper_compression_audit.ipynb)。已發現公開MNIST範例的U3/CU3與論文RY/CNOT不一致，尚未啟動精確數值重現。

沿用 **PennyLane QuantumTrain 的方法思想**，以小型 digits 分類任务學習權重生成、量子梯度與可信實驗。提供可逐節閱讀的繁體中文 notebook、測試及自動化評估。這是獨立適配作品，不是原論文數值重現。

## 開始操作（Windows PowerShell）

在此專案根目錄執行，需先有 [uv](https://docs.astral.sh/uv/)：

```powershell
uv sync --frozen
uv run jupyter lab
```

使用此專案 `.venv` 的 Python kernel。若在 VS Code 選 kernel，指定 `.venv/Scripts/python.exe`。Notebook 第一節會印出 `sys.executable` 讓你確認。

### 教材閱讀順序

第二階段入口：[十輪研究總結](docs/research-summary.md)、[完整train協定](docs/full-validation-protocol.md)、[04完整資料工具書](notebooks/04_full_validation.ipynb)。正式矩陣命令：`uv run python scripts/full_validation_study.py`；加`--smoke`快速檢查。第四份教材可由 `scripts/build_full_validation_notebook.py` 重建。

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

第二階段完成：[完整資料結果解讀](reports/full-validation/interpretation.md)。平均validation accuracy：直接MLP80.22%、傳統生成器58.59%、量子可訓練41.04%、量子固定23.96%。量子更新在此設定有增量效用，但仍落後傳統對照；多數best epochs位於20epoch上限，不能推論充分收斂後排名。這是開發性validation，不是test結果。

第十輪完成：[傳統特徵重訓結果](reports/classical-features/interpretation.md)。傳統30參數特徵最後train平均90%，量子30參數61.67%；validation仍低且逐seed不一致，沒有量子優勢證據。建議先整理研究脈絡，再設計完整train協定。

第十輪 [傳統特徵對照協定](docs/classical-feature-protocol.md)：量子固定/可訓練、零特徵、固定隨機特徵與30參數傳統可訓練特徵，共15runs；初始非零feature std與mapping權重對齊。`uv run python scripts/classical_feature_study.py`，可加`--smoke`。第三份教材第14節說明表示與參數預算的比較限制。

第九輪完成：[特徵尺度結果](reports/feature-scale/interpretation.md)。1000×下可訓練量子參數的最後train平均61.67%，固定組50%；但三seed validation accuracy都較固定組低。前向敏感度增加，尚無泛化或量子優勢證據。

第九輪 [特徵尺度×量子可訓練性](docs/feature-scale-protocol.md)：1×/1000×各配對Frozen/Trainable，三seeds共12runs。命令 `uv run python scripts/feature_scale_study.py`，加`--smoke`為四組短測。第三份教材第13節說明尺度、梯度與飽和，避免把倍率收益誤稱量子學習收益。

第八輪完成：[量子敏感度與消融結果](reports/quantum-ablation/interpretation.md)。固定與可訓練量子參數的best-checkpoint validation accuracy逐seed相同；所測checkpoint對量子feature置零的反應很小。這僅描述當前低表現設定，不代表一般QML無效。

第八輪 [量子敏感度與消融協定](docs/quantum-ablation-protocol.md)：先量測機率→權重→logits的擾動影響，再比較固定與可訓練量子參數。`uv run python scripts/quantum_ablation_study.py` 執行六組；加`--smoke`快速檢查。第三份教材第12節區分梯度連接、敏感度與實際貢獻。

第七輪完成：[fan-in結果解讀](reports/fanin/interpretation.md)。持續fan-in縮放使平均最後train accuracy由51.67%降到21.67%；validation loss兩好一壞，平均accuracy僅約15.9%，未建立可靠擬合baseline。

第七輪 [fan-in尺度協定](docs/fanin-protocol.md)：固定中心化，對照每次forward是否將W1/W2標準差調至1/sqrt(3*fan_in)，bias不變。執行 `uv run python scripts/fanin_study.py`（6runs），加`--smoke`快速檢查。第三份教材第11節解釋持續重參數化與單純初始化的差異。

第六輪 [中心化结果](reports/centering/interpretation.md)：平均最後train accuracy由33.33%到51.67%，但seed22退步；三seed最佳validation loss皆變差，沒有可靠泛化收益。

第六輪 [權重中心化協定](docs/centering-protocol.md)：初始化先比較W1/b1/W2/b2與preactivation，再固定Tanh/LeakyReLU、LR1e-3，只對W1/W2各自減平均。執行 `uv run python scripts/centering_study.py`（6runs），可加 `--smoke`。第三份教材第10節說明中心化與尺度正規化的差異。

第五輪結果：[learning-rate解讀](reports/learning-rate/interpretation.md)。降低LR使後段loss波動變小，但固定200epochs下的擬合與validation未改善；平穩不等於已學會或已收斂。

第五輪 [learning-rate協定](docs/learning-rate-protocol.md) 固定Tanh/LeakyReLU，比較1e-3、3e-4、1e-4；使用 `uv run python scripts/learning_rate_study.py` 執行9runs，或加 `--smoke`。第三份教材第9節解釋最好epoch、最後表現與時間波動的差異。

第四輪結果已完成：[固定LeakyReLU的mapping比較](reports/leaky-mapping/interpretation.md)。Tanh提高20筆資料上的最高訓練擬合（平均21.67%→46.67%），但validation收益不穩定，且部分run後期退化；不能只報最好epoch。

第四輪固定LeakyReLU、depth3、20筆train、200epochs，比較affine/Tanh，各3seeds：`uv run python scripts/leaky_mapping_study.py`。可加 `--smoke`；匯出使用 `--report results/<完整study> --output reports/leaky-mapping`。中斷的study不與完整矩陣混用。第三份教材第8節新增權重加性結構與奇異值的引導。

第三輪 [ReLU/LeakyReLU對照](reports/target-activation/interpretation.md)：LeakyReLU恢復輸入差異，但20筆最高訓練accuracy僅20–25%；同樣目標MLP直接訓練可達85–95%。新增的教材第7節說明此現象。重跑：`uv run python scripts/target_activation_study.py`（正式6 runs），或加 `--smoke` 快速檢查；報告命令用 `--report results/<study> --output reports/<新目錄>`。

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
