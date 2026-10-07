# 驗證紀錄

## 第四輪：固定LeakyReLU的mapping比較

- 固定depth3、affine/Tanh、相同seeds11/22/33、20筆train、200epochs；未使用test。
- 32項pytest通過，新增加性/非加性/零矩陣結構指標測試。
- 第三份notebook擴為9節，新增權重加性結構與奇異值教學；fresh-kernel smoke通過。
- 首次前景執行因200秒上限中斷，保留results/leaky-mapping-cb2f5b230f，不納入正式比較；完整矩陣改背景執行，結果待完成後匯出。
- 背景study `leaky-mapping-b5be7b4cfc` 6/6完成，報告匯出成功。Tanh最高train平均46.67%，affine21.67%；validation平均約15.60%與15.13%，沒有穩定泛化改善。完整限制及不穩定現象見reports/leaky-mapping/interpretation.md。

## 第三輪：目標activation

- 31項pytest通過；新增負區域梯度、snapshot與真實forward一致、protocol保存測試。
- 三份notebook fresh-kernel smoke通過（11/9/8節）；第3份新增ReLU/LeakyReLU教學。
- 正式6-run target-6d2e608121完成；另做6次train-only Classical tiny控制。結果與限制見reports/target-activation/interpretation.md。
- 保留舊欄位relu_zero_fraction，定義為若使用ReLU時的零比例（preactivation≤0）；新增hidden_zero_fraction表示實際activation的零比例。LeakyReLU下兩者不同。

## 第二輪：訓練診斷與 mapping 延伸

- 新增 affine/Tanh mapping 配置；預設 affine，原實驗配置與模型 state_dict 結構保持相容。
- 新增4項測試：兩组相同初始Linear/量子參數、snapshot不改模型與RNG、diagnostic runner只接train/validation、新配置驗證。
- 全套28項pytest通過。
- 三份notebook fresh-kernel smoke均通過（11、9、7節）。第三份包含預測分布、受控mapping比較、20筆過擬合、整段訓練梯度與activation、結果判讀。
- 正式第二輪協定見 mapping-protocol.md；是否完成與結果以 reports/mapping 的實測報告為準。
- 第二輪 `mapping-710a60bdbc` 已完成12/12 runs；report_mapping.py成功匯出summary、逐epoch manifest與兩張圖，三seed完整。所有run最後epoch固定probe的ReLU零比例1.0、logits跨圖片標準差0.0；Tanh未改善。完整解讀見 reports/mapping/interpretation.md。

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

正式矩陣 `study-8091e80c8e` 已完成12/12 runs，逐一檢查 completed、diagnostics passed 與 best.pt 存在；最終報告已從原始 artifacts 重建至 reports/。具體觀察與限制見 reports/interpretation.md。

已逐節檢查生成來源的20個 section：每節包含目標、理由、shape、程式定位、操作、觀察、判讀、問題及解答。Notebook 01 有可執行的資料圖、電路圖、shape trace、detach 反例與重載驗證；Notebook 02 有明確矩陣、validation-only pilot、缺漏判讀與報告產出。

這是作者／工程檢視，不等於已由真正零背景的新讀者完成 usability 測試。後續可請一位新人記錄卡住的 section，再迭代教材。
