# 第三階段：回到論文的參數—準確率主張

查核日期2026-10-08。狀態：論文/程式差異已識別，**精確數值重現尚未開始**。

## 第一手來源

- 論文最新arXiv版本v2（2024-06-10）：https://arxiv.org/html/2405.11304v2 ，§II.2、§II.3、§III與Tables1–3。
- 舊v1：https://arxiv.org/html/2405.11304v1 ，同項表格本次核對數字相同。
- Repository chenyu_dev固定commit：d5b654116565ae9550e540e021ab0d4160a15c86。
- MNIST notebook：https://github.com/Hon-Hai-Quantum-Computing/QuantumTrain/blob/d5b654116565ae9550e540e021ab0d4160a15c86/appendix_example/MNIST/torchquantum/torchquantum_FF_mapping_model_L16.ipynb
- main固定commit b21db14500512b9f99fb1e7e7ca81af0de8489e8 的相同MNIST檔案blob也為7a344e0049d05acd5440a882145e175716197cc3，因此只換main不解決差異。

## 真正應驗證的主張

QT用量子參數加mapping參數生成M個目標模型權重。比較P_QT/M與同任務準確率差距，不是只比量子與傳統模型誰accuracy較高。參數減少不自動等於總記憶體或wall time下降。訓練後可匯出完整classical weights做純傳統推論，是另一項可驗證貢獻。

| 論文報告（非本機重現） | 直接模型 | QT參數 | 比例 | 直接accuracy | QT accuracy | 差距 |
|---|---:|---:|---:|---:|---:|---:|
| MNIST QT-16 | 6690 | 457 | 6.83% | 98.45% | 81.08% | 17.37pp |
| MNIST QT-106 | 6690 | 1627 | 24.32% | 98.45% | 93.81% | 4.64pp |
| FashionMNIST QT-121 | 6690 | 1822 | 27.23% | 88.14% | 80.74% | 7.40pp |
| CIFAR10 QT-323 | 285226 | 23258 | 8.15% | 62.50% | 60.69% | 1.81pp |

因此「差不多準確率＋很低參數量」在論文中是依資料集及深度而異的trade-off；不能把最小參數QT-16與最高accuracy當同一設定。

## 實際程式與論文的差異

| 項目 | 論文v2 | 公開MNIST notebook | 影響 |
|---|---|---|---|
| 目標CNN | 6690參數，fc192→20→10 | Conv1 1→8 k5、Conv2 8→12 k5，無ReLU，pool兩次；fc192→20→10 | 208+2412+3860+210=6690，計數吻合；詳細activation以code為證據 |
| QNN | 每block RY＋線性CNOT，13角度 | 每block 13個U3＋circular CU3，78角度 | 16blocks從208变1248，與表格不符 |
| Mapping | 14→4→20→4→1，文字說初始層用Tanh | 相同249參數，forward無任何activation | 計數相同不代表方法相同 |
| 基底bit | 文中例子0/1 | -1/+1 | 需明列選擇 |
| 機率處理 | 方法段說以probability作mapping輸入，未完整指定code縮放 | 從state amplitude取abs平方一次，再tanh/冪次/中心化 | 與之前PennyLane例子的p再平方不一致 |
| 輸出處理 | 方法段未明確指定 | 生成weights全域減平均 | 不能無聲加入舊專案的逐層center |
| 訓練 | 50epochs、batch128、Adam lr1e-4 | QT同上，另weight_decay1e-5、eps1e-6、ReduceLROnPlateau | 未揭露細節需追溯 |
| Classical baseline | 表格98.45%，完整train60000/test10000 | cell0僅1epoch、batch200、lr.001 | 該cell不充分重現表格baseline |
| 選模／評估 | 表格test accuracy | 依單batch train accuracy存模型；另將原test分1000/9000後評估1000 | 評估協定不一致 |

公開notebook的QT-16結構計數為1248+249=1497（22.38%），不是Table1的457。尚未安裝TorchQuantum執行該檔，這是明確gate定義與標準U3/CU3三角度語意的靜態計數。

## 我們之前作品的界線修正

原digits目標610參數、generator16791，約27.53倍，根本不是壓縮配置。其十輪診斷有效描述本機適配，但不能拿來支持或反駁論文Table1–3。後續Tanh/LeakyReLU/center/1000×是另外的實驗改動；保留為教材與歷史，不作論文重現證據。

另：論文文字將tanh稱為unbounded，數學上不正確；tanh本身值域(-1,1)，後續linear輸出才可不受此界限。重現教材應清楚說明，不直接沿用該敘述。

## 下一個可執行里程碑

首選MNIST 6690參數CNN、13qubits與mapping249，先做QT-16的壓縮/梯度/權重匯出驗證，再視預算跑QT-106。QT-16本身不預期93.81%，對應原表是81.08%。CIFAR10近準確率設定需19qubits/323blocks/1000epochs，不能把短程CPU run冒稱完成。

在啟動長程訓練之前，需解決一個真實阻礙：取得與Table1對應的RY/CNOT程式或固定設定（電路初態、CNOT排序、mapping activation、機率前處理、初始化、baseline預算、評估checkpoint）。可向作者詢問；或由使用者決定採「依論文重建但部分細節有假設」路徑，將所有假設事前列出，結果不稱精確重現。

目前已有可重跑的paper_audit.py參數稽核與第五份教學入口。**尚未訓練MNIST，不輸出新的準確率。**
