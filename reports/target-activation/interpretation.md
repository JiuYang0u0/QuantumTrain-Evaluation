# 第三輪：恢復負區域梯度，只解決了部分問題

## 協定與結果

Study `target-6d2e608121`；固定depth3、affine mapping、每類2筆共20筆、Adam lr=.001、200epochs，三seeds11/22/33。兩組量子/Linear初始參數依相同seed對齊，唯一變因是目標MLP隱藏activation：ReLU或LeakyReLU(slope=.01)。未使用test。

| Target activation | 每次最高train accuracy | 平均最高train accuracy |
|---|---|---|
| ReLU | 10%、10%、15% | 11.67% |
| LeakyReLU | 20%、25%、20% | 21.67% |

LeakyReLU最後epoch的hidden zero fraction在三seeds皆為0，logits跨圖片標準差皆大於0；ReLU則皆hidden全零、logits標準差0。這是全部20筆train的觀察。

負區域梯度恢復後，模型不再只輸出生成的bias，但仍無法有效擬合20筆。因此ReLU失活是瓶頸之一，不是完整解釋。不能把些微accuracy增加宣稱為量子優勢或穩定泛化改善。

## 補充：直接訓練同一個小型MLP

後續train-only控制實驗，使用同樣20筆、200epochs、Adam lr=.001：

| Target activation | 各seed最後train accuracy |
|---|---|
| ReLU | 90%、85%、90% |
| LeakyReLU | 95%、95%、90% |

重跑命令：`uv run python scripts/tiny_classical_control.py --output results/new-control.json`。與QT並非相同權重初始化分布，這是模型/資料的基本可擬合性控制，不是同初始化的因果比較；結果顯示610參數目標架構本身可以在同epoch數下學到很多，而QT權重生成路徑仍有限制。

## 生成權重的結構線索

Affine-only mapping將每個權重生成為 bit編碼的線性組合＋機率特徵係數＋偏置。第一層512個權重reshape成8×64時，索引高位控制row、低位控制column，因此bit部分呈現row effect＋column effect的加性結構（矩陣rank至多2）；量子機率特徵才提供額外非加性結構。

在LeakyReLU的三個best-validation checkpoints，第一層權重扣除row/column加性近似後的相對Frobenius norm分別約5.53e-7、9.55e-7、4.78e-7；第三奇異值約9.39e-7、3.48e-6、9.29e-7，前兩個奇異值則約0.61至10.96。這支持目前生成的第一層權重幾乎是低秩加性形式。

此觀察是checkpoint分析，不是普遍rank定理：量子機率特徵一般可產生非加性矩陣，其大小與初始狀態、電路及機率縮放都有關係。

## 下一步

先在固定LeakyReLU下再比較affine/Tanh mapping，以避免第二輪的mapping效果被ReLU失活遮住。若Tanh仍不足，再針對輸出權重的共同偏移、各層fan-in尺度做單變因實驗。另可加入固定量子機率對照，但目前不宜用學不會的baseline直接判斷量子學習價值。

上列是下一輪假說，尚未執行。研究屬逐輪exploratory analysis，已有validation參與判斷，不當作全新盲測。

## 保存位置

- `runs.csv`和`target-study.md`保存本輪完整表格。
- `results/target-6d2e608121/`保存protocol、split索引、每run config/environment/history與best.pt，以及classical-tiny-control.json。
- 本輪在開發工作樹啟動，環境紀錄含working_tree差異；不是clean-commit封存重跑。
