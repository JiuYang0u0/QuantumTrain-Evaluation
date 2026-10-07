# 第九輪：特徵尺度與量子學習的2×2對照

倍率1×/1000×，各自配對量子參數trainable/frozen，seeds11/22/33，共12runs。倍率1000是根據第八輪初始化feature std約1e-4事前指定，非從test或本輪結果挑選。不是每次除std的自動標準化。

固定Tanh mapping、LeakyReLU(.01)、center_weights、depth3、q_delta=.001、Adam lr=.001、固定20筆train、200epochs。倍率放在probability_features後、與bit編碼串接前，只有一個量子特徵欄放大。兩組同seed原始參數一致；放大必然改變forward及梯度尺度，不能宣稱有效learning rate保持相同。

主觀察：最高/最後train accuracy、最低validation loss checkpoint的accuracy/loss、初始化及best checkpoint敏感度；監測scaled feature std、量子/mapping梯度與Tanh飽和（每層abs(output)>.99比例）。Frozen量子delta必須恆0，null grad是預期。

只用train/validation，不讀test；這是低表現設定上的探索，不能把倍率改善或Frozen仍有效直接當量子優勢證據。12runs全列出，不挑最好seed；與第八輪1×控制比較重跑一致性。

命令：`uv run python scripts/feature_scale_study.py`；加`--smoke`為單seed、2epochs、四組。匯出：`uv run python scripts/feature_scale_study.py --report results/<study> --output reports/feature-scale`。
