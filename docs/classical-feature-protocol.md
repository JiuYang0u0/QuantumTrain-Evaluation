# 第十輪：重訓特徵對照

五組：1000× quantum trainable、1000× quantum frozen、zero、random fixed、classical trainable。三seeds11/22/33，每組相同20筆train、200epochs、Adam lr=.001、Tanh mapping、LeakyReLU(.01)、center_weights。只train/validation，不讀test。

相同seed複製完全相同的mapping初始權重。傳統feature使用固定seed2026的610×30高斯隨機基底，各欄先中心化並設std=1；30維係數初值複製同seed量子theta。固定gain將初始feature std對齊1000×量子feature，訓練中不再自動調整。Random fixed與classical trainable有相同初始feature，僅係數是否更新不同。Zero保留bit編碼、量子feature欄為0。

30可訓練傳統係數對照30量子參數，mapping皆16,761個參數。可訓練參數數量及初始std匹配不代表表達能力、梯度尺度、記憶體或計算預算完全相同；固定隨機基底占額外儲存。傳統組的gain參考量子初始化std，是尺度控制，不是任務標籤學習；不把它當完全無量子準備成本的部署比較。

記錄最高/最後train、最佳validation loss與其accuracy、feature grad/std/delta、參數delta與Tanh飽和。所有固定组参数delta须0；可訓練feature须有有限梯度。配對差異依三seed逐一報告，沒有量子優勢預設。

`uv run python scripts/classical_feature_study.py` 啟動15runs；加`--smoke`為5組2epochs。`--report results/<study> --output reports/classical-features`匯出。當前仍是低樣本探索，不是獨立盲測。下一步應整理研究脈絡，而不是根據test追逐更好數字。
