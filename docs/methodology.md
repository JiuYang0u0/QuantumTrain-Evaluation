# 實驗方法與解讀限制

## 問題
在固定資料、目標 MLP、mapping、optimizer 與 epoch 預算下，比較 1/3/5 層量子電路的分類表現與訓練成本。深度同時改變參數量，不能獨立歸因於深度本身。

## 資料與選模
digits 像素除以固定上限 16；分層切分為約 60/20/20，split seed 2026。train 負責更新，validation loss 選最佳 checkpoint，test 僅用於最終評估。不得用測試結果調參。Smoke 每 split 40 筆只檢查工程流程。

## 模型
所有目標網路均為 64→8→10，中間 ReLU，輸出 logits，使用 CrossEntropyLoss。QT 的 1,024 個機率取前 610 個，不重新正規化。基底 bit 順序為 wire 0 在最高位。

處理公式為 f(p)=(0.8*tanh(0.1*512*p²))^0.3，再減平均；p 已是機率，平方是額外特徵轉換，不是從振幅計算機率。預設不加入 epsilon，任何穩定化修改必須另作設定與版本紀錄。

Mapping 11→40→200→40→1 共有 16,761 個參數：480+8,200+8,040+41。未加 activation，函數上可合併成 affine transform；量子特徵本身仍是參數的非線性函數。這個小型示範的生成器大於 610 個目標參數，不能宣稱壓縮。

RY 層從第 1 層開始，RZ 在偶數層，故主實驗用 1/3/5。末端 RZ/CZ 對計算基底機率無影響，某些零梯度是結構特性，不等同梯度中斷。

## 指標
- Loss 按樣本數加權，accuracy 存為 0–1，報告可換成百分比。
- Train seconds 只計訓練 batch forward/backward/update；另記端到端時間，含模型建立、診斷、驗證、checkpoint I/O、測試，截止 result.json 最終寫入之前。
- CPU threads 固定並記錄；同 epochs 不等於同成本。
- 梯度存在、有限且有更新是健康檢查；小電路有限差分則交叉檢查導數。
- 三 seeds 報均值與樣本標準差 ddof=1；單次顯示 N/A。這是描述統計，不是量子優勢或顯著性證明。
- 不合協定的 runs 不合併；缺 seed 與失敗 run 顯示 incomplete，不靜默移除。

## 範圍
default.qubit、shots=None、backprop 是傳統電腦上的理想模擬。沒有測量噪聲與硬體排程，不可推論 QPU 成本。第一版支援最佳模型重載，不提供精確中斷續訓。
