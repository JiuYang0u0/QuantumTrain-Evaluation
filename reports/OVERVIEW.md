# 專案總覽:從論文到可驗證的壓縮結果

**QuantumTrain Evaluation** — PennyLane／PyTorch 的教學、可重現實驗與參數壓縮評估工具。

本文件是單頁入口:說明全文目標、已驗證與未驗證的事、關鍵數字,以及如何接手。

## 目標

論文 arXiv:2405.11304（Quantum-Train）主張:用量子神經網路QNN＋傳統 mapping 生成一個有 M 個權重的古典神經網路,使**獨立可訓練參數量從 M 降到 O(polylog(M))**,推理時完全不需要量子電腦。

我們要驗證的是三件事:
1. 這個機制能否正確訓練(梯度真的回到量子參數)?
2. 在相同可訓練參數預算下,是否比古典壓縮方法更好?
3. 準確率能維持到什麼程度?

## 現況摘要

| 階段 | 完成 | 代表結果 |
|---|---|---|
| 工具與教材(6 份 notebook) | ✅ | 工程正確性、續訓、官方 test 保護 |
| 十輪小模型診斷 | ✅ | 找出 ReLU 失活、權重偏移、特徵尺度三大問題 |
| 論文／原始碼差異稽核 | ✅ | 公開 MNIST 程式與論文表格不符 |
| paper-guided reconstruction(r1→r2→r3) | ✅ | 逐步補來源處理,損失回到合理範圍 |
| 古典對照(隨機特徵、低秩) | ✅ | 同預算下量子明顯較好 |
| **50 epochs 完整資料＋官方 test** | ✅ | QT-106 24.32% 參數達 96.08% |

## 核心數字(官方 MNIST test,只用一次)

54,000 train／6,000 validation、23,450 更新、seed 11、官方 test 未參與訓練:

| 方法 | 可訓練參數 | 比例 | 官方 test |
|---|---:|---:|---:|
| 直接 CNN | 6690 | 100% | 98.43% |
| **QT-106** | 1627 | 24.32% | **96.08%** |
| 低秩 r=4 | 1998 | 29.87% | 93.68% |
| 低秩 r=3 | 1511 | 22.59% | 89.14% |
| **QT-16** | 457 | 6.83% | **85.89%** |
| 低秩 r=2 | 1024 | 15.31% | 71.90% |
| 低秩 r=1 | 537 | 8.03% | 34.22% |

**可支持的結論**:在此小架構與預算下,論文機制的壓縮效果存在,且相同預算下明顯優於低秩直接訓練。

## 不可支持的(請勿過度解讀)

- **單 seed**:穩定性未驗證(早期 3-seed 顯示 QT 變異可達 9pp)。
- **非論文精確復現**:r3 重建、PennyLane、RY/CNOT、單一深度;論文表格不可直接比較。
- **官方 test 只用一次**:若之後持續用它比較,就不再是凍結評估。
- **論文的 CIFAR-10 結果完全未測**(19/95/171/247/323 blocks、1000 epochs)。
- **無 shots／gate error／真實硬體**;CPU float64 理想模擬。
- **無效率主張**:QT-106 每 run 約 7.3 小時,低秩約 0.15 小時。這是參數效率比較,不是運算速度。

## 如何接手

```powershell
uv sync --frozen
uv run pytest -q                 # 81 項測試
uv run jupyter lab               # 依 01→06 順序閱讀
```

- 教材:[docs/reading-guide.md](../docs/reading-guide.md)
- 研究階段／協定索引:[docs/research-index.md](../docs/research-index.md)
- 進度查詢:`uv run python -m qt_eval.progress --list`
- 續訓說明:[docs/checkpoint-recovery.md](../docs/checkpoint-recovery.md)
- 官方 test 保護:[docs/paper-final-test-protocol.md](../docs/paper-final-test-protocol.md)
- 來源與重用範圍:[NOTICE.md](../NOTICE.md)、[docs/provenance.md](../docs/provenance.md)

## 下一步(優先序)

1. 多 seed 驗證 QT-16／QT-106 穩定性。
2. 論文規模 CIFAR-10 壓縮曲線(需 GPU,見 [GPU 遷移](gpu-portability.md))。
3. shots／parameter-shift 梯度成本評估。
4. 更強的古典壓縮基準(可訓練低秩、剪枝)。

原始 checkpoints 與逐次紀錄在本機 `results/`(Git 忽略),報告與 manifest 在 `reports/`。
