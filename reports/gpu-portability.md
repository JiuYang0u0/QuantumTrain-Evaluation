# GPU 遷移清單

目前所有數值結果都是 **CPU、float64、Torch threads=1** 產生的。若改到有 GPU 的桌機重跑,以下是必須注意與驗證的事項,否則結果不可與既有 CPU 結果混用。

## 1. 這個專案為何慢

| 成本來源 | 說明 |
|---|---|
| PennyLane `default.qubit` | state-vector 模擬,**只用 CPU/NumPy**,不自動用 GPU |
| float64 精度 | 為了有限差分/梯度一致性;GPU 上 float64 通常慢很多 |
| 每 step 重算電路 | 多次前向沒有快取;可審慎評估是否重複計算 |

## 2. 遷移選項(依風險由低到高)

| 選項 | 做法 | 適合 |
|---|---|---|
| A. 維持 CPU 直譯,換較快 CPU | 什麼都不改 | 只求縮短等待 |
| B. `lightning.qubit` 或 `lightning.gpu` | 換 PennyLane 裝置名稱,其他不動 | **最推薦的下一步**;需重新驗證數值一致性 |
| C. `pennylane-lightning[gpu]` + CUDA | 需 CUDA toolkit;float64 支援依卡號而異 | 需要先確認裝置 |
| D. 調整 dtype 為 float32 | 需重跑有限差分與梯度驗證;threshold 需放寬 | 最快但破壞與既有結果可比性 |

## 3. 遷移時必須重跑的驗證

- `uv run pytest tests/test_paper_mnist.py tests/test_paper_depth.py`
  測試內含解析機率、有限差分、匯出一致性;換裝置/換 dtype 後必須再次通過。
- 官方 test **只能再評估一次**:遷移後的結果是新的凍結評估,不可回頭與既有官方 test 比較後選設定。
- CPU 結果與 GPU 結果應視為**不同裝置的兩個 study**,分別記錄 environment.json。

## 4. 既有 artifacts 是否可攜

| 檔案 | CPU ↔ GPU |
|---|---|
| `best.pt`、`latest.pt` | state_dict 只是張量,**可跨裝置載入**(`torch.load(..., weights_only=True)` 後用 `.to(device)`) |
| `protocol.json` 內的 `torch_version`、`threads` | 續訓協定檢查會比對;**換 GPU 後不可直接續跑 CPU study** |
| 抽樣器 RNG | CPU generator 狀態與 CUDA 不同,續訓不 bitwise 等價 |
| `train_seconds` / `train_eval_size` | 計時不可跨裝置比較 |

## 5. 建議的執行順序

1. 在新桌機先 `uv sync --frozen`(或重建環境),確認可 `import pennylane` 並列出可用裝置。
2. 跑 `uv run pytest -q` 確認 81 項測試在新環境通過(尤其是梯度與匯出一致性)。
3. 跑一次最短 smoke:`uv run python scripts/paper_full_data.py --smoke`
4. 再決定是否啟用 lightning 裝置,並以 2-qubit 解析案例比對 CPU/GPU 數值。
5. 長跑建議使用 50-epoch 協定,讓官方 test 評估保持一次性:
   ```powershell
   uv run python scripts/paper_full_data.py --updates 23450 --lowranks 1,2,3,4 --output results\full50-gpu
   uv run python scripts/final_test_eval.py results\<study> --output reports\final-test-50ep-gpu
   ```

## 6. 優先在新環境解的研究問題

1. QT-106 多 seed 穩定性(最便宜、最關鍵)。
2. 論文規模 CIFAR-10 的壓縮曲線(QT16/QT106 已完成的 MNIST 結果是基礎)。
3. 若追求真實硬體,再處理 shots 與 parameter-shift 成本。

**注意**:遷移是為了速度與規模,不是為了讓數字變好看。任何 dtype/裝置變更都必須在報告中揭露,並視為新的實驗。
