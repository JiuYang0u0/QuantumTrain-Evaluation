# 第二輪結果：Tanh 未改善，診斷定位到隱藏層失活

Study：`mapping-710a60bdbc`；12/12 runs 完成。設定與逐epoch數據見 [manifest.json](manifest.json)，彙整見 [mapping-study.md](mapping-study.md)。本輪只使用 train/validation，沒有新的 test 評估。

## 結果

| 情境 | Mapping | Validation accuracy 平均 ± 樣本SD | 各run最高train accuracy的平均 |
|---|---|---|---|
| 完整train，20epochs | affine | 10.03% ± 0.00百分點 | 10.20% |
| 完整train，20epochs | Tanh | 10.03% ± 0.00百分點 | 10.20% |
| 20筆train，200epochs | affine | 10.40% ± 0.64百分點 | 11.67% |
| 20筆train，200epochs | Tanh | 9.94% ± 0.16百分點 | 10.00% |

Validation 指標取自 validation loss 最低的 checkpoint；最高train accuracy是過擬合能力診斷，不是選模規則。三seeds為11/22/33。

兩組都未能記住20筆資料。單純在mapping加Tanh，在這個凍結設定下沒有改善。

## 比準確率更有資訊的診斷

- 所有12個run的最後epoch，在固定training probe上 `relu_zero_fraction=1.0`，八個hidden units皆無活躍輸出。
- 同一probe上的 `logits_between_images_std=0.0`，輸出不隨圖片改變，每個run均只預測一個類別。
- seeds11與22在初始化probe就全部失活；seed33起初全活躍，最後也全部失活。
- 最後訓練loss接近 ln(10)=2.302585；與無法區分10類的現象一致。
- 量子梯度和參數更新仍可能非零，因此單看更新健康檢查不足以證明模型在學習輸入特徵。

觀察範圍：完整train的probe只取前40筆，不把這個snapshot當成全體資料的逐筆失活證明；tiny情境的probe涵蓋全部20筆。初始化、每epoch及best checkpoint的觀察分開保存在manifest中，上述總結明確指最後epoch。

## 為什麼失活時還能有量子梯度？

目標MLP是 `h=ReLU(xW1+b1)`、`logits=hW2+b2`。當probe的h全為零時，輸出只剩生成的b2，與圖片x無關。QuantumTrain同時生成b2，因此loss仍能經由b2、mapping回到量子參數；但這條路徑不能幫模型區分那些圖片。

這是對目前計算結構和觀測的解釋，不代表已證明失活的最初原因。生成權重的相關性、共同偏移、尺度、初始化與優化均可能參與其中。

## 下一個受控實驗（尚未執行）

優先在20筆train上比較目標MLP的 ReLU 與 LeakyReLU，保持mapping、電路與初始化固定，檢查允許負區域梯度後能否恢复輸入依賴與擬合能力。這是修改目標模型的新協定，所有對照需一致處理，不能混入原深度研究。

若仍不收斂，再單獨研究生成權重的初始化與分層尺度。不要同時換activation、LR與權重正規化。固定量子參數消融等到有基本學習能力後再做。本輪沒有量子優勢、一般QuantumTrain無效或barren plateau的證據。

## 版本說明

本輪在開發期間啟動，因此environment可能記錄上一個commit及非空working_tree；對應診斷程式已提交於 `773cbd2`。原始每run環境如實保留，不能稱為從clean commit啟動的完全封存研究。原始checkpoint/history保存在本機results，匯出的manifest保留逐epoch診斷與協定。
