# 固定更新/樣本處理預算的資料覆蓋比較

保持r3與CNN初值、Adam lr1e-4、batch128；1000/10000 train × CNN/QT16 × 三seed，共12runs。兩個train集合由同train pool分層抽樣，保存實際交集，不假設必然nested；validation固定同200筆且與兩者互斥。

每组4000updates、每次128個樣本，共512000次樣本處理。使用固定seed獨立generator均勻有放回抽樣，避免1000筆每epoch尾批只有104、10000筆尾批只有16造成曝光數不同。記錄實際unique examples seen。相同seed對齊初始化，但不同集合的batch內容不同。

此抽樣協定不同於上一輪epoch shuffle，故**必須重跑1000筆控制，不直接混用前輪500epochs數字**。固定每100updates驗證，共40個選模機會，兩資料規模一致。Best依validation loss，best train/validation同checkpoint重測；final另存。

不增加epoch、不改深度/mapping/尺度、也不使用official test。資料量及集合組成會同時變，三seeds只改模型初始化/抽樣、不估計資料切分不確定度；仍屬探索性開發評估，不稱論文完整重現或資料量因果定理。

命令：uv run python scripts/paper_data_budget.py。--smoke單seed4updates，但仍真實1000/10000資料及同validation。4000updates仍小於論文完整MNIST約23450updates，且最大僅10000train。完成後再決定完整資料預算。
