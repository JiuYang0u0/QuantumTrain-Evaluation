# 固定r3的更新預算研究：開始有效學習，仍有38pp準確率差距

Study paper-r3-budget-e20332a432，6/6完成。固定1000train/200validation、三seeds、batch128、500epochs=4000updates；r3與CNN其餘設定不變。未使用official test，非論文完整MNIST重現。

## 同一最低validation loss checkpoint的比較

| 模型 | 參數 | 平均train accuracy | 平均validation accuracy ± 樣本SD | 平均validation loss |
|---|---:|---:|---:|---:|
| CNN | 6690 | 97.63% | 91.00% ± 1.32pp | .31404 |
| QT16 r3 | 457 | 55.00% | 53.00% ± 4.00pp | 1.42832 |

QT僅用6.831%獨立可訓練參數（減少93.169%），但目前平均validation差38pp，沒有接近CNN準確率。這是在已用於開發/選模的200筆validation上的描述，不是獨立泛化估計。

## 預定epoch節點（當下模型，非best checkpoint）

| Epoch / updates | CNN平均validation accuracy | QT平均validation accuracy |
|---|---:|---:|
| 50 / 400 | 85.00% | 17.17% |
| 100 / 800 | 88.83% | 17.33% |
| 200 / 1600 | 90.67% | 17.50% |
| 500 / 4000 | 91.83% | 51.33% |

第50節點重現前輪r3/CNN，支持配對重跑一致性。QT明顯進展主要在200到500epochs之間；更多更新是影響學習的重要因素，不能根據前輪400updates的低accuracy直接否定方法。

## Best與final不同

- CNN best epochs229/214/219，best val91.5/92.0/89.5%；最後train全部100%，final val91.5/92.5/91.5%。Final validation loss均比best高，accuracy卻不一定下降：交叉熵也反映信心分配，兩指標不可混用。
- QT best epochs500/500/495，best val49/53/57%；final val49/53/52%。Seed33在最後5epochs accuracy下降5pp；最佳點靠近預算末端，仍不能認定已充分收斂。
- QT全部匯出獨立CNN，logit delta0；壓縮表示和classical inference驗證成立，準確率維持尚未成立。

## 下一個實驗決策（尚未執行）

現在已有學習進展，應避免繼續任意改架構。下一步有兩個不同問題：
1. 在相同更新預算下增加訓練樣本覆蓋，檢查資料量作用；需按updates而非epochs對齊，另訂訓練/驗證切分與checkpoint觀察。
2. 固定資料/預算、比較論文QT16與更深的QT設定，產生真正的參數—accuracy trade-off；更深電路也增加運算成本，須先估算。

若目標仍是論文重建，優先往完整資料的固定協定推進並持續揭露來源假設，而不是把此小樣本53%對原論文81.08%當直接重現誤差。原表60000train×50epochs約23450updates，這輪只重複1000筆4000updates。

原始metrics、updates、50/100/200/500節點checkpoint、best及classical-export在results/paper-r3-budget-e20332a432；manifest、milestones.csv、runs.csv與study.md保存對應結果。Train-only本機時間平均CNN104.41秒、QT802.18秒，未作受控多次performance benchmark，不推論QPU速度。核心協定commit4c996c6後啟動。
