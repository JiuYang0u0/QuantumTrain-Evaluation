# MNIST QT-16重建pilot：壓縮與匯出成立，準確率尚未重現

Study paper-mnist-pilot-56567c7db0。明列假設見../../docs/paper-mnist-protocol.md，非精確論文重現。官方MNIST train IDX校驗通過；分層取1000train/200validation，seed11、2epochs、batch128、Adam lr1e-4，共16次optimizer更新。不下載、不讀official test。

| 模型 | 獨立可訓練參數 | 參數比例 | train accuracy | validation accuracy | validation loss |
|---|---:|---:|---:|---:|---:|
| CNN | 6690 | 100% | 13.5% | 15.5% | 2.29146 |
| QT16 | 457 | 6.831% | 9.7% | 9.5% | 116795.42 |

兩組均以validation loss最低checkpoint（epoch2）重測train/validation。這是工程pilot，不是論文50epochs/60000train/10000test的結果。不能用此比較判斷論文accuracy是否成立。

## 已驗證
- 目標CNN的6690參數與mapping249＋theta208=457吻合。
- 原始p、0/1bits、13qubits RY/CNOT、首層Tanh的獨立重建可微分；量子梯度norm約29.62，所有模型參數梯度有限。
- 生成weights可匯出獨立CNN；同batch logits最大差0.0，推論不需QNode。
- MNIST資料來源、MD5/SHA256、索引及環境保存於manifest.json；best.pt與classical-export.pt留在results原目錄。

## 先處理的阻礙
QT loss在pilot結束仍約11.7萬，遠高於隨機10類分類的ln10尺度。這是輸出logits/生成權重尺度的警訊，不是只看9.5%就認定模型無效。此r1沒有採用公開變體的生成weights中心化，因論文方法段未充分交代；也未套用舊digits的修正。下一步应先量測初始化/各層activation/生成weights，對照上游前處理差異，定位缺少的假設，而非無聲修改後冒稱忠實重現。

## 預算
QT16兩epoch的train-only約2.35秒，直接CNN約.31秒；以每step外推50epochs×469steps約QT .96小時、CNN .13小時，不含評估/I/O且不保證實際速度。正式長跑暫未啟動，先處理數值尺度與來源假設。

本pilot在開發工作樹執行，環境如實記錄。後續每項重建修改需獨立版本和來源說明，論文引用數字與本機測量分欄呈現。
