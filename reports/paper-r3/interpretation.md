# MNIST r3：機率後處理恢復梯度尺度，準確率改善很小

2026-10-09。Study `paper-r3-ad19c7a0dc`，6/6runs完成。固定r2其他假設、1000train/200validation、50epochs（400updates）、三seeds。r2輸入raw p；r3輸入公開MNIST公式(.8*tanh(.1*4096*p))^.3再減平均，不重複平方p。不讀official test。

| 設定 | 可訓練參數 | 平均train accuracy | 平均validation accuracy ± 樣本SD | 平均validation loss |
|---|---:|---:|---:|---:|
| r2 raw | 457 | 16.03% | 16.50% ± 5.22pp | 2.27663 |
| r3 upstream | 457 | 17.43% | 17.17% ± 5.06pp | 2.26861 |

各seed validation r2→r3：22.5→23.0%、13.0→14.5%、14.0→14.0%。200筆validation每筆為0.5pp，分別只差1、3、0個樣本；三seed共用同一validation切分，不將它們當600個獨立樣本。

所有best-validation-loss checkpoint都為epoch50，尚無充分收斂證據。平均accuracy只增加0.67pp，不能稱可靠改善或論文重現成功。參數比仍457/6690=6.831%，參數量縮減與accuracy維持需分開陳述。

## 初始尺度與更新證據

機率feature標準差從raw約.000167–.000176，變成r3約.14984–.15177。初始量子梯度norm由約2.78e-5–2.19e-4變成.0496–.3498，作用尺度顯著改變。

r3三seed全部50epochs量子參數有非零delta，最後batch量子gradient norm為.556/.0120/.207，mapping為1.995/.258/.975。梯度增加不代表Adam更新同比增加，也不代表學習效率/泛化改善。本輪未發生非有限梯度；没有額外epsilon修正，不能據此保證其他初態或訓練不會遇到零機率奇異性。

所有六個checkpoint匯出獨立CNN的logit delta0。Raw控制重現上一輪r2結果。

## 階段結論

兩項有來源依據的處理已分開核對：全域生成weights中心化去除了巨量前向放大；機率tanh/冪次/中心化改變了量子特徵與梯度尺度。然而小資料400updates下，r3仍未取得可比的分類準確率。

不要再次以這個結果否定原論文60000train/50epochs。仍有初態、CNOT順序、mapping細節、框架與作者初始化等差異。此版本是paper-guided reconstruction，不是精確code/table reproduction。

## 下一步候選（尚未執行）

先固定r3，不再疊加不同來源變體；下一個可辨識問題是訓練步數不足。可先在同1000train/200validation增加固定更新預算，明確以updates報告，並保留classical CNN對照；這會隔離更新次數，但仍不能代替完整60000筆資料。若直接做完整資料，須先固定train/validation/test協定、記錄估計時間，並將資料量和更新次數同時改變的限制寫清楚。

若更高預算仍不學習，應優先尋找作者對應版本/設定，而不是把任意改善逐一冠為原方法。此文件沒有啟動下一輪或官方test評估。

完整表格與曲線見study.md，manifest.json含初始尺度/梯度、每epoch紀錄及環境；原始checkpoint在results/paper-r3-ad19c7a0dc。核心/協定commit b727010後啟動，教材在背景期间更新，環境working_tree如實保存。
