# MNIST QT-16：paper-guided reconstruction r1

狀態：依使用者同意，以公開假設重建，**不是精確論文數值重現**。來源arXiv:2405.11304v2 Tables1–3與差異表paper-reproduction-audit.md。

## 已核對與固定假設

| 項目 | 本次選擇 | 證據/界線 |
|---|---|---|
| Target CNN | Conv1 1→8 k5、pool2、Conv2 8→12 k5、pool2、fc192→20→10；無額外ReLU | 上游CNN實作，6690與Table1計數吻合；pool仍非線性 |
| QNN | 13qubits、16blocks，每block13 RY＋12 CNOT依0→1至11→12依序執行 | 論文RY/CNOT與208角度吻合；具體gate順序是公開假設 |
| 初態 | 全零，不加H | 重建假設，不宣稱已恢復作者設定 |
| Mapping | 14→4→20→4→1，只第一層Linear後Tanh，最終linear | 依§II.3的initial layer Tanh描述；其他層無activation是字面重建假設 |
| Bit編碼 | 0/1，wire0最高位 | 依論文Eq2例子，不使用舊實驗±1 |
| 機率欄 | 原始p取前6690，不再正規化、不平方、不乘1000 | 依方法段直接probability描述；上游縮放公式不無聲混入 |
| 生成weights | 不中心化、不fan-in scaling | 方法段未充分指示這些處理，保留不處理作r1假設 |
| 初始化 | theta uniform[-pi,pi]，mapping PyTorch預設；seed11 | 只知道作者random，分布未對齊；本次先固定可追溯設定 |
| 模擬 | PennyLane default.qubit、Torch backprop、CPU float64 | 論文用TorchQuantum；移植差異明列 |
| 優化 | Adam lr1e-4、batch128；eps1e-8、weight_decay0，無scheduler | 主文設定＋明列預設，不混入公開變體scheduler |
| MNIST | 原始28×28、像素/255後(x-.5)/.5 | 對齊上游Normalize；不用digits，不resize |

## 參數與成果目標

Target6690；mapping249；QNN208；QT457，占6.831%（減少93.169%）。這是**架構計數**，不表示準確率維持。原表QT16是81.08%對98.45%，不是QT106的93.81%。需要分別呈現參數壓縮及accuracy gap。

## 本階段pilot（非表格結果）

只下載官方MNIST train兩個IDX檔，驗證已知MD5。固定split seed2026：先從60000筆分層留6000 validation，再从train pool取1000 train、validation pool取200 validation（分層）。不下載、不讀官方test。

比較direct CNN與QT16，各seed11、2epochs、batch128、Adam lr1e-4。目的是驗證完整資料路徑、估算50epochs成本及梯度，不以小型pilot判斷論文效果。保存索引、配置、檔案雜湊、train/validation曲線、生成器checkpoint及純CNN匯出；匯出前後同batch logits必須一致。

單batch步驟時間外推50epochs×ceil(60000/128)，只作CPU估計，不含評估/I/O，且實際成本會改變。完整50epoch及QT106預算待pilot結果評估後決定，不自動宣稱已復現。
