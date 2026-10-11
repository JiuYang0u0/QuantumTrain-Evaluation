# 官方 test 一次性凍結評估協定

訓練腳本全程不讀官方 10,000-image test；本工具只在所有 run 完成後，對每個 best-validation-loss checkpoint 執行一次評估，結果寫入 `official_test_*` 欄位，並記錄 `official_test_read=True`。

## 使用

```powershell
uv run python scripts/final_test_eval.py results/<study> --output reports/<dir>
```

- study 必須已完成（所有 `result.json` 為 completed），否則拒絕。
- 重新驗證 study 的 validation 索引，避免錯配。
- 生成器模型同時載入 `classical-export.pt`，核對 test batch 上匯出 CNN 與生成器輸出一致（`export_delta`）。
- 每筆模型只用 test 一次；若同一 study 重複執行，會得到相同數字，但不得據此反覆調整後再宣稱凍結。

## test 資料來源

ossci 鏡像的 `t10k-*.gz` 與經典 MNIST gzip 雜湊不同（容器重新壓縮），但解碼內容經三重驗證：IDX 標頭（2051/2049、10000×28×28）、標籤直方圖等於官方 test 分布、像素範圍 0–255。雜湊以鏡像值記錄於 manifest。

## 解讀

- 官方 test 未參與 model selection；若持續使用它比較新設定，就不再是獨立評估。
- 單 seed 或非論文 50 epochs 的 run 只能用於方向確認，不稱為論文復現。
- 參數比與 accuracy gap 分開報告；量子版 CPU 成本遠高於古典基準，屬參數效率比較。
