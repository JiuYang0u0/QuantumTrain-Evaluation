# 第五輪：降低學習率減少後段loss波動，但未改善固定預算下的擬合

Study：`learning-rate-f37260840c`，正式9/9runs完成。Tanh mapping、LeakyReLU(.01)、depth3、同20筆train、Adam、200epochs、seeds11/22/33；只改learning rate。沒有使用test。

## 結果

| LR | 平均最高train accuracy | 平均最後train accuracy | 平均peak-to-final退化 | 最後20epochs loss時間SD的平均 | 平均best-checkpoint validation accuracy |
|---|---|---|---|---|---|
| 1e-3 | 46.67% | 33.33% | 13.33百分點 | 0.04018 | 15.60% |
| 3e-4 | 25.00% | 21.67% | 3.33百分點 | 0.00701 | 15.32% |
| 1e-4 | 31.67% | 20.00% | 11.67百分點 | 0.00284 | 14.48% |

時間SD是同一run後20epochs的波動（ddof=0），不是獨立抽樣的統計不確定度；各seed原始結果見runs.csv。Validation accuracy對應最低validation loss checkpoint，不是最高accuracy epoch。

## 判讀

1. 降低LR時後段loss的時間波動變小，但這不證明收斂：平滑趨勢也會影響時間SD。更不能把波動較低視為已解決學習問題。
2. 3e-4的平均退化幅度較小；1e-4仍曾達較高accuracy後回落。穩定性不是所有指標都隨LR單調改善。
3. 同200epochs下，1e-3的平均最高與最後train accuracy仍較高。小LR可能學得更慢，不能推論其充分收斂後一定較差。
4. 三組validation平均約14.48–15.60%，沒有一致改善。所有組都未能有效記住20筆，因此「只降低LR」尚不足以建立可靠baseline。
5. 1e-3的各seed最高/最後train accuracy與第四輪吻合（45/40、50/45、45/15），支持此控制組的重跑一致性。

## 下一個受控問題（未執行）

優先研究生成權重的初始化／尺度，固定Tanh、LeakyReLU與一個事前選定LR，不再繼續盲目掃描學習率。先量測每層生成weight/bias的均值、std、fan-in尺度與preactivation分布，確認共同偏移和尺度是否形成瓶頸，再選單一處理方式對照。

若採逐層中心化或fan-in scaling，兩者應分開測試；避免一次改兩項後無法歸因。仍先用train/validation小資料測試。固定量子參數消融應在有基本擬合能力後進行，不能用低表現直接否定量子參數貢獻。

## 證據與版本

完整結果、曲線見study.md；manifest.json保存每run配置、環境與逐epoch診斷。本機results/learning-rate-f37260840c保存checkpoint與原始檔。訓練協定及核心先提交678e462再啟動；背景執行期間教學文件有更新，環境working_tree按實況記錄。本輪為探索性研究，不是全新盲測或量子優勢證據。
