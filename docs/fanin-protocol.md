# 第七輪：固定中心化後只改權重尺度

兩組：center_weights / center_fanin。固定Tanh mapping、LeakyReLU(.01)、depth3、q_delta=.001、Adam lr=.001、20筆train、200epochs、seeds11/22/33，共6runs。只接train/validation，不使用test。

處理公式：對W1/W2各自減平均後，除以 sqrt(max(mean(centered²),1e-12))，乘以 1/sqrt(3*fan_in)。W1 fan_in=64，目標std≈.07217；W2 fan_in=8，目標std≈.20412。此尺度來自PyTorch Linear預設均勻初始化的理論標準差，但不等於獨立均勻初始化，也不是He scaling。

每次forward都處理，梯度穿過中心化與分母；它改變可行參數化與優化幾何，不只是初始大小。Bias不中心化、不缩放。variance在sqrt前floor，避免常數block的sqrt零點造成非有限導數；低於floor的block不保證達目標std，常數block仍為零。

主要觀察：同200epochs的最高/最後train accuracy、最後train loss、最佳validation loss與其accuracy、初始化與每epoch的block/preactivation分布。不能只以最高train挑成功seed。不增減特定seed的epochs。

重跑：`uv run python scripts/fanin_study.py`，可加`--smoke`。報告：`uv run python scripts/fanin_study.py --report results/<study> --output reports/fanin`。
