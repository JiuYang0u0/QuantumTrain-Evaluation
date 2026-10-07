# 第六輪：生成權重共同偏移

先前train-only初始化診斷：seeds11/22/33，QT W1均值約-.067/-.165/.130，preactivation平均約-1.28/-3.24/2.73；Classical相同架構初始化preactivation接近零（並非同初始化分布的因果比較）。W2尺度也有差異，但本輪不調整尺度。

選定單一處理：生成W1與W2各自減去整個矩陣的平均；b1/b2保持原值。此操作保留各weight block標準差，並透過torch張量運算保留梯度。不做逐neuron中心化、不除以std、不加fan-in scaling。

凍結：Tanh mapping、LeakyReLU(.01)、depth3、q_delta=.001、Adam lr=.001、20筆固定train、200epochs、三seeds11/22/33；none/center_weights共6runs。不使用test，同seed保持相同原始模型參數。

觀察：最高/最後train accuracy、train loss、best-validation-loss checkpoint的accuracy、每層weight/bias統計、preactivation與hidden活躍狀態。中心化可能移除有用的自由度，成功與否依數據判斷，不預設改善。

運行：`uv run python scripts/centering_study.py`；加`--smoke`為單seed2epochs。匯出：`uv run python scripts/centering_study.py --report results/<study> --output reports/centering`。
