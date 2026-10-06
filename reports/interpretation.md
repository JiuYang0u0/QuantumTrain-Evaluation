# 第一輪結果解讀

日期：2026-10-07（本地）；run id 使用 UTC，因此開頭為 20261006。Study：`study-8091e80c8e`。原始資料保留於 `results/study-8091e80c8e/`；可分享的設定、環境、診斷及結果見 [manifest.json](manifest.json)，表格與圖見 [depth-study.md](depth-study.md)。

## 實測觀察

| 模型 | Test accuracy，平均 ± 樣本標準差 | 平均 train-only 秒數 |
|---|---|---|
| Classical MLP | 74.26% ± 8.10 個百分點 | 0.39 |
| QT depth 1 | 10.19% ± 0.16 個百分點 | 7.57 |
| QT depth 3 | 10.19% ± 0.16 個百分點 | 15.17 |
| QT depth 5 | 10.28% ± 接近 0 個百分點 | 19.64 |

所有組別皆完成 seeds 11/22/33，各20 epochs。三個量子深度均通過初始量子梯度與參數更新檢查；這表示所測 batch 的計算圖能回傳及更新，不表示每次 step 都有非零更新，也不保證訓練能解決任務。

本設定下量子版本接近10類分類的機率基準；深度增加未帶來實質精度改善，訓練成本則上升。不可據此宣稱 QuantumTrain 一般無效，也不可宣稱實現量子優勢。

## 限制與下一步假說

- 小型 target 只有610個參數，mapping卻有16,761個登記參數，且全是 affine layers；架構表達限制可能影響結果，但本輪沒有單獨驗證其因果作用。
- 機率平方／縮放、near-zero 初始化、資料與目標模型均影響優化。不能僅凭低 accuracy 歸因於 barren plateau。
- Classical 的三 seed 變異明顯；三次僅供描述。
- 固定 epochs、同 learning rate 不是各模型的最佳超參數，也不等於同計算預算。
- 正式矩陣在測試前凍結；本輪不因 test 結果改設定再冒充同一輪獨立評估。

建議下一輪先做 validation-only 診斷：查看生成權重分布、prediction class histogram、有效梯度與 activation；再選單一變因，例如 nonlinear mapping 或固定量子參數消融。此處只是後續建議，尚未執行，亦未取得改善證據。

## 版本追溯

訓練核心來源 commit 為 `d29736f`。實驗期間另有診斷 JSON 與報告 provenance 的修正，未修改量子電路、mapping、資料或 optimizer；每次環境的 working_tree 狀態如實保存在 manifest。最終報告使用修正版 reporting 重建，沒有重跑訓練。
