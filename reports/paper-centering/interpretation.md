# MNIST r2：上游全域中心化解決前向放大，學習效果仍待長程驗證

初始化診斷：paper-scale-5450d2a62d。配對pilot：paper-centering-e8cef99f87，9/9完成。1000train/200validation、2epochs、每組16次更新、三seeds。未下載/使用official test。

| seed | 直接CNN val loss / accuracy | r1 QT raw val loss / accuracy | r2 QT global-center val loss / accuracy |
|---|---|---|---|
| 11 | 2.29146 / 15.5% | 116795.42 / 9.5% | 2.37174 / 10.5% |
| 22 | 2.29429 / 13.5% | 1444.59 / 10.0% | 2.45946 / 10.5% |
| 33 | 2.28885 / 9.5% | 1070.71 / 13.5% | 2.31373 / 11.5% |

同初始化同probe的前向干預顯示，全域生成偏移是r1巨量loss的直接因素：seed11 fc1 absmax由226694.52降至6.32；三seed初始化loss由127050/2435/1546降至2.42/2.61/2.34。這是前向對照的因果線索，不是完整訓練因果或準確率證據。

r2中心化是上游MNIST code明確存在的處理，但論文方法段未充分說明。本次只加這一項，不同時套用上游U3/CU3、機率縮放或改activation；它是paper-guided r2，不是精確論文數值重現。

QT依舊457個可訓練參數，生成6690個CNN參數，比例6.831%。所有六個QT pilot均匯出獨立CNN，logits delta0。改善的是數值範圍，不是已驗證的壓縮—精度trade-off；目前三組都只16次更新、accuracy低，不能判斷論文效果。

下一步可先固定r2做較長但有限的pilot，觀察train loss是否下降、量子與mapping參數是否持續更新，並檢查是否仍有來源設定缺口。若啟動完整50epochs，須明确使用60000train且留原test到凍結評估，或宣告54000/6000開發切分與原表不同；不能混淆。尚未開始新長跑。

詳細參數假設見../../docs/paper-mnist-protocol.md、../../docs/paper-centering-protocol.md。原始checkpoint/metrics位於results/paper-centering-e8cef99f87，manifest.json保存協定、逐run結果、環境與初始化診斷。
