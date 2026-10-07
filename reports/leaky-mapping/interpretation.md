# 第四輪：非線性增加擬合能力，但沒有穩定的驗證收益

完整 study：`leaky-mapping-b5be7b4cfc`，6/6 runs完成。固定LeakyReLU(.01)、depth3、相同20筆train、200epochs、Adam lr=.001、三seeds11/22/33；唯一比較變因為mapping隱藏層是否加入Tanh。未使用test。

## 觀察

| Mapping | 各seed最高train accuracy | 最後epoch train accuracy | best-validation checkpoint的validation accuracy |
|---|---|---|---|
| Affine | 20%、25%、20% | 20%、15%、20% | 15.32%、18.94%、11.14% |
| Tanh | 45%、50%、45% | 40%、45%、15% | 13.93%、16.71%、16.16% |

- 最高train accuracy的平均從21.67%提高到46.67%，增加25個百分點，屬小資料上的探索性擬合改善。
- Validation accuracy的平均僅由約15.13%到15.60%，不同seed方向不一致（11/22下降，33上升）。不支持穩定泛化改善。
- Tanh seed33從最高45%下降到最後15%，顯示訓練不穩定。不得只報最好epoch而省略最後狀態。
- 最佳checkpoint依validation loss選取，不是最高validation accuracy；其train accuracy與最高train accuracy也不同。

## 權重結構

在best-validation checkpoints，W1(8×64)扣除row/column加性近似後的相對Frobenius norm：

- Affine：5.53e-7、9.55e-7、4.78e-7。
- Tanh：0.0668、0.0730、0.0514。

非線性mapping確實產生更多非加性權重結構，伴隨較高的訓練擬合上限。不過結構指標量測的是best checkpoint，而最高train accuracy可能在其他epoch；不能由兩者並列就聲稱已證明完整因果機制。

## 結論與下一輪

ReLU失活與affine mapping的限制都值得處理；單獨改mapping曾被失活遮住，固定LeakyReLU後可看到非線性帶來的擬合差異。但Tanh仍未記住20筆，也有明顯不穩定，距離可信的任務baseline仍有差距。

下一個優先假說是生成權重的尺度與共同偏移：在固定Tanh/LeakyReLU下，只改生成权重的逐層中心化與fan-in尺度，先用train/validation檢查是否穩定收斂。亦可先做較低learning rate的單變因對照；兩項應分開執行，不能同時修改後歸因。這些下一步尚未執行。

不宣稱量子優勢。量子參數的增量貢獻仍需固定量子參數與純傳統生成器對照。

## 追溯

[study.md](study.md) 保存表格與曲線，[manifest.json](manifest.json) 保存完整設定、環境、診斷與結構數值。本機 `results/leaky-mapping-b5be7b4cfc/` 保存checkpoint/history。中斷的 `leaky-mapping-cb2f5b230f` 不納入比較。本轮在開發工作樹啟動，環境如實保存，並非clean-commit封存重跑。
