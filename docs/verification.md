# 驗證紀錄

## 2026-10-07 第一輪

- 在新建獨立 Git repo 的 `feat/quantumtrain-evaluation` 分支實作。
- Python 3.11.15、PennyLane 0.45.1、Torch 2.14.1+cpu；完整相依套件見 uv.lock。
- pytest 最終回歸：24 passed，涵蓋 data split、functional MLP、有限差分、terminal phase、已知 X 態 bit ordering、detach 反例、診斷隔離、checkpoint、失敗保存、非有限梯度、reporting 完整性及 study pilot。
- Notebook 01：11 個教學 sections，fresh-kernel smoke 通過。
- Notebook 02：9 個教學 sections，fresh-kernel smoke 通過。
- 執行時 Jupyter/ZeroMQ 有 Windows event loop fallback 與本機 TCP transport 提示，未造成 cell 失敗；驗證過程不提供對外 Jupyter 服務。
- 完整資料 depth5、1 epoch validation-only pilot：train-only 1.08498 秒，val loss 2.30420。僅用作預算估計，未評估 test。
- 修正計畫中的 mapping 參數計數：480+8200+8040+41=16,761。
- 額外修正非有限梯度診斷：NaN norm 保存為 JSON null，保持 finite=false，避免錯誤診斷本身因嚴格 JSON 序列化而遺失；回歸測試由失敗到通過。
- 報告 manifest 同時保存 environment 與 diagnostics，讓分享的報告可追溯執行版本與量子更新證據。

## 教材閱讀檢視

已逐節檢查生成來源的20個 section：每節包含目標、理由、shape、程式定位、操作、觀察、判讀、問題及解答。Notebook 01 有可執行的資料圖、電路圖、shape trace、detach 反例與重載驗證；Notebook 02 有明確矩陣、validation-only pilot、缺漏判讀與報告產出。

這是作者／工程檢視，不等於已由真正零背景的新讀者完成 usability 測試。後續可請一位新人記錄卡住的 section，再迭代教材。
