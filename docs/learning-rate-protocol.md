# 第五輪：降低 learning rate 是否減輕後期退化？

正式執行前固定：Tanh mapping、LeakyReLU(.01)、depth3、q_delta=.001、CPU float64、Adam；同一固定train split每類前2筆，共20筆，200 epochs、batch64（每epoch一個更新）。三seeds11/22/33與三個learning rates .001/.0003/.0001，共9runs。

只改learning rate；同seed初始化與資料順序相同。每個rate使用不同結果子目錄，checkpoint不混用。完整紀錄僅train/validation，不讀test。這是已觀察過validation之後的探索性延伸，不能當獨立盲測。

事前定義觀察：最高與最後train accuracy、peak-to-final drop、最後20epochs train loss與accuracy的時間標準差、最後loss，以及validation loss最低checkpoint的accuracy。時間標準差用ddof=0，它不是跨seed不確定度；跨seed摘要使用ddof=1。

小learning rate可能只是收斂更慢；不能把低波動直接解讀為成功。沒有到100%也不是方法一般無效的證明。第一輪結果不覆寫，不事後增加特定seed的epochs。

執行：`uv run python scripts/learning_rate_study.py`；smoke加 `--smoke`（單seed、2epochs）。匯出：`uv run python scripts/learning_rate_study.py --report results/<實際study> --output reports/learning-rate`。中斷run保留，report拒絕缺漏矩陣。
