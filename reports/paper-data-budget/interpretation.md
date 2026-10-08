# 固定更新預算的資料覆蓋研究：CNN受益，QT結果高度依賴seed

Study paper-data-budget-9fc8b65ab4，12/12runs完成。各組4000updates、batch128有放回抽樣、512000次樣本處理，共用200validation，不讀official test。1000/10000集合交集947筆，並非巢狀子集；資料量與集合組成同時不同。

| 模型 | Train size | 參數 | Best train平均 | Best validation平均 ± 樣本SD | Validation loss平均 |
|---|---:|---:|---:|---:|---:|
| CNN | 1000 | 6690 | 97.80% | 91.00% ± 1.32pp | .30925 |
| CNN | 10000 | 6690 | 96.33% | 94.33% ± .76pp | .14878 |
| QT16 r3 | 1000 | 457 | 58.70% | 56.33% ± 6.01pp | 1.38581 |
| QT16 r3 | 10000 | 457 | 39.47% | 39.83% ± 21.25pp | 1.76872 |

Best指每100updates驗證、最低validation loss的checkpoint，train與validation在同checkpoint重測。三seeds只估計初始化與抽樣變動，不估計資料切分不確定度。

## 配對細節

CNN validation隨資料量1000→10000為91.5→94.5%、92.0→93.5%、89.5→95.0%，三seed都改善。

QT則為50.5→40.0%、62.5→18.5%、56.0→61.0%，兩次下降、一次上升。較大資料組seed22停留低表現，導致平均和標準差變差；不能把平均下降推論為「更多資料必然有害」。六個QT的最低validation loss都在第4000更新，尚未證明充分收斂。

實際unique examples seen皆等於各組集合大小1000或10000。雖更新與曝光總數相同，每個樣本平均曝光約512次對51.2次；這是固定預算下增加多樣性的取捨，不是同每樣本訓練量的比較。

## 參數—準確率口徑

QT仍457/6690=6.831%可訓練參數。此協定下1000組CNN/QT平均accuracy差34.67pp，10000組差54.50pp，尚不支持接近CNN準確率。這不是論文60000train/23450updates/官方test的精確重現，也有r3來源假設。

所有QT匯出classical CNN logits delta0。1000控制組因改用有放回抽樣，不應與前輪epoch-shuffle數字當成相同協定；本輪已重跑控制。Validation只有200筆且已多次參與開發，不能把其均值當獨立泛化估計。

## 下一步建議（尚未執行）

先查10000組seed22是否只有慢學習或存在表示/優化瓶頸，再決定固定更新預算延長或論文深度對照。不要僅因平均較差就回退資料，也不要同時改深度、mapping、初始化。若繼續往論文規模，需要固定完整train/validation/test協定並明列來源未決假設；取得作者對應版本仍是最直接縮小重現差距的方式。

結果與曲線見study.md；manifest.json含配置、環境、索引及驗證歷史，原始best.pt/history在results/paper-data-budget-9fc8b65ab4。核心/協定commitb9ad207後啟動，期間報告/教材更新，working_tree如實記錄。
