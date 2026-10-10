# 驗證紀錄

## 低秩直接訓練基準（完成）
- 新增低秩模組：目標CNN權重以rank-r因子直接訓練，無生成器與量子電路；參數487·r+50（r=1→537、2→1024、3→1511、4→1998、6→2972）。
- 4項測試通過（參數量、全秩精確重建、梯度、匯出一致、rank邊界）；成套pytest通過。
- 正式15/15完成（paper-lowrank-42eadd6099，含中斷後續訓）。極低預算量子大幅領先（457：QT16 76.0%對低秩約32%）；約1627：QT106 90.0%對低秩約86.4%，低秩需約1998才達90%；r=6（2972）95.33%。量子優勢隨預算縮小但未消失。報告reports/paper-lowrank/interpretation.md。

## 匹配預算古典特徵對照（完成）
- 新增古典隨機特徵生成器：固定投射＋softmax取代量子電路，mapping與r3後處理不變；可訓練數精確對齊457/1627。
- 4項針對性測試通過（mapping與量子相同且單一Tanh、參數量、梯度、softmax總和、匯出一致、非量子僅z可訓練）。
- 真MNIST 2組smoke完成；正式6runs完成（paper-classical-ea7265d9ee）。每個固定節點與seed，量子版都較高：457預算76.0%對43.0%（+33.0pp）、1627預算90.0%對52.5%（+37.5pp）；量子版CPU慢得多。限制見reports/paper-classical/interpretation.md。

## README導航與唯讀進度工具
- README改為快速入口，舊內容保留HISTORY.md；新增reading-guide/research-index/progress-guide。四份入口的相對連結自動檢查無缺漏，Notebook本體未改動。
- 新增qt_eval.progress，psutil列為直接依賴；分離宣告狀態、保存進度與程序證據。首次命令未含目標study時只標candidate；--resume-root精確比對；权限不足為unknown，不以running檔案推論存活。
- 全套68項pytest通過：含六項progress測試，覆蓋僵置running、精確resume歸属、候選程序、權限不足、損壞JSON、完成與checkpoint不反序列化。
- 真實完成study d206b2e56a查詢顯示6/6 completed_reported、無對應程序；best/final指標分列。工具不輪詢、不修改run、不載入pt。

## QT106深度對照（完成）
- 新增QT106參數1627/量子1378、有限非零gradient、純CNN匯出一致性測試通過。
- 真MNIST4updates smoke完成，milestone/latest/export保存成功。
- CPU warmup後3step median：QT16 .166秒、QT106 1.001秒，非完整工期；原數據results/qt106-step-benchmark.json。
- 教材06十一節fresh-kernel smoke通過。正式三seed於核心協定commitc00140d後啟動，每100steps完整checkpoint；CNN/QT16引用已完成控制。
- study paper-depth-3f729b7257三seed完成（多次中斷後由latest.pt續訓）；status completed，控制身份/預算核對後報告匯出。平均best val CNN96.33%、QT106 90.00%、QT16 76.00%；QT106 ratio24.32%、export delta0。限制見reports/paper-depth/interpretation.md。

## 固定10000筆12000updates完成
- 最終recovery study paper-data-extended-d206b2e56a完成6/6，報告核對完成身份、預算、索引互斥及milestones後成功匯出。
- 沿用五組、QT33從頭重跑；QT33之4000/8000節點重現中斷前結果，12000完成，三QT export delta0。
- Best val平均CNN96.33%/QT76%，固定12000節點CNN95.67%/QT75.83%。參數比6.831%，精度維持仍未成立；paper比較限制見reports/paper-data-extended/interpretation.md。

## 完整續訓保護
- 全套61項pytest通過；含Dropout的中斷/不中斷對照：模型、Adam state、抽樣器、history及best/final指標一致；資料/預算不匹配拒絕且不覆寫。
- latest.pt於每個驗證節點原子replace；包含best副本，恢復修正history/best的一致性。保存範圍與限制見checkpoint-recovery.md。
- 原recovery aa21c465c5停在QT33的10200updates，無訓練程序。五個完成組保留；舊QT33無完整checkpoint必須從頭重跑。
- 核心commitda9ef40後重新背景啟動，最後一組新訓練具完整保存保護。尚未取得本次完整结果。

## 12000updates矩陣中斷恢復
- 原paper-data-extended-c3e310decb有CNN11/QT11/CNN22三組完成，QT22保存至3800updates後程序消失；無error.txt，原因未確定，不再以running檔案狀態認定程序存活。
- Recovery新增目錄，核對協定、資料索引、completed身份/預算與best.pt可讀後沿用；QT22/CNN33/QT33由相同seed初始化重跑，不覆寫原資料。
- 兩組完整extended smoke的recovery驗證通過，原目錄保留，recovery.json列來源與重跑清單。
- 核心commit3cf87af後重新啟動背景工作；結果待完成。恢復是run層級，不是optimizer/RNG續訓。

## 固定10000筆延長updates（進行中）
- 先讀前輪歷史：seed22末段loss下降、量子梯度增大；尚不能確認慢啟動根因。
- CNN/QT全部三seeds從頭12000updates，核心協定commite48ce35後背景啟動；保留4000/8000/12000。
- update-budget針對性測試、兩組真MNIST extended smoke、報告節點匯出、教材06十節fresh-kernel通過。未重跑全套測試，未讀test。

## MNIST資料覆蓋（完成）
- 新update-budget測試通過：精確updates、完整batch曝光數、固定eval schedule及checkpoint。
- 真實1000/10000資料四組4updates smoke完成，報告驗證train/validation互斥與validation相同、曝光預算及run身份後匯出。
- 教材06九節fresh-kernel通過。本輪未重跑全部測試，上一輪全套58通過，新增針對性測試1通過。
- 正式12runs在核心/協定commitb9ad207後啟動，背景執行中，未讀official test。
- Study paper-data-budget-9fc8b65ab4完成12/12，報告核對updates/exposure/identity、共用validation及索引互斥後匯出。Train集合交集947非nested；unique examples均覆蓋全組。CNN1000/10000 val91%/94.33%，QT56.33%/39.83%，所有QT export delta0；限制見reports/paper-data-budget/interpretation.md。

## 固定r3更新預算（完成）
- 同1000train/200validation，CNN/QT16三seeds，500epochs4000updates；核心/協定commit4c996c6後背景啟動。
- 單seed2epoch smoke配對訓練、節點checkpoint/metrics、報告milestones.csv完整性檢查通過，重現既有r3短測數值。
- 教材06八節fresh-kernel smoke通過。本輪改動為研究runner與文件；上一輪全套58項pytest通過，本輪沒有宣稱重新跑全套。
- Study paper-r3-budget-e20332a432完成6/6，報告核對完成狀態、seed/condition、500epoch歷史及四個milestones後匯出。Best val CNN91%、QT53%；固定500節點QT51.33%。第50節點重現前輪，QT export delta0。完整限制見reports/paper-r3-budget/interpretation.md。

## MNIST r3機率後處理（完成，2026-10-09）
- 全套58項pytest通過，新增來源公式不重複平方p、同初始化、梯度/匯出一致性與無效設定測試。
- 單seed2epoch raw/upstream配對smoke與report通過；第六份7節教材fresh-kernel通過。
- 正式三seed×兩組50epoch pilot於核心/協定commit b727010後背景啟動；未使用official test。詳細假設見paper-r3-protocol.md。
- Study paper-r3-ad19c7a0dc完成6/6，報告核對狀態/condition/seed/50epoch紀錄後匯出；raw重現前輪r2，三個r3全部50epochs量子delta非零，所有匯出delta0。Validation平均16.50%→17.17%，僅小改善，原論文accuracy仍未重現。詳見reports/paper-r3/interpretation.md。

## 固定MNIST r2較長pilot（完成）
- 全套56項pytest通過，新增epoch observer不改訓練與checkpoint的回歸驗證。
- 第六份6節教材fresh-kernel smoke通過；1000/200、2epoch、單seed CNN/QT的pilot/report整合smoke通過。
- 50epoch六runs於核心/協定commit7a0ebd8後背景啟動；每run400updates，不是論文完整MNIST50epochs。記錄最後batch梯度與epoch首尾delta，未讀test。
- Study paper-r2-learning-331edce2ee完成6/6，報告核對狀態、condition/seed與50epoch歷史後匯出。QT三seed量子/mapping在全部50epochs有非零delta，export delta0；CNN/QT validation平均85%/16.5%。參數比成立，accuracy維持尚未成立，解讀見reports/paper-r2-learning/interpretation.md。

## MNIST r2全域中心化
- 三seed同初始化/同probe前向對照確認共同偏移造成巨量CNN輸出；詳見paper-scale-5450d2a62d。
- 直接CNN/r1/r2三組三seed的2epoch MNIST pilot完成9/9；r2 val loss回到2.31–2.46，accuracy尚低，不判定論文效果。
- 5項paper模型/中心化針對性測試通過；第六份5節教材fresh-kernel smoke通過。新增全域變換、有限梯度及匯出語意測試。
- 報告與9run manifest位於reports/paper-centering；下一個長程pilot尚未啟動。

## MNIST QT-16 paper-guided reconstruction r1
- 使用者同意明列假設重建；協定見paper-mnist-protocol.md。
- 全套54項pytest通過；新增6690/457計數、functional CNN等價、量子梯度、2qubit解析機率/有限差分、原始p與0/1bit接口、獨立CNN匯出驗證。
- 第六份教材4節fresh-kernel smoke通過。
- 真實MNIST pilot兩組完成（1000train/200validation、2epochs）；官方train檔checksum通過。QT梯度有限、匯出logit delta0，但loss約11.7萬，需先診斷尺度，未啟動50epoch長跑。
- 原始目錄results/paper-mnist-pilot-56567c7db0；報告reports/paper-mnist-pilot/interpretation.md；沒有新的論文準確率重現主張。

## 第三階段：論文壓縮來源稽核（2026-10-08）
- 讀取arXiv v1/v2與固定上游MNIST notebook，核對main/chenyu_dev相同blob。
- 參數表稽核2項測試通過；published-claims.csv/audit.md已產生，accuracy標為published_not_reproduced。
- 第五份教材4節fresh-kernel smoke通過。
- 明確阻礙：論文RY/CNOT QT16=457；公開MNIST U3/CU3結構1497，mapping activation及評估也不一致。未啟動MNIST訓練、沒有新的準確率。
- 詳細來源與待確認設定見paper-reproduction-audit.md；需要對應程式或使用者選定公開假設重建路徑。

## 第二階段：完整train/validation（完成）
- 十輪研究樹與限制整理於research-summary.md；凍結協定見full-validation-protocol.md。
- 48項pytest通過，新測試驗證runner只接train/validation、Frozen不變與一致target activation。
- 第四份教學notebook五節fresh-kernel smoke通過，含完整四組短訓練與報告匯出。
- 正式12runs於核心/協定commit1dd4cfa之後背景啟動，未使用test；結果待完成。
- Study full-validation-9c8e7d4048完成12/12。報告核對condition/seed、完成狀態、固定參數delta與20epoch歷史後成功匯出；結果及收斂限制見reports/full-validation/interpretation.md。

## 第十輪：傳統特徵重訓對照（完成）
- 46項pytest通過，含共享mapping初值、非零feature尺度對齊、固定旗標與傳統30參數梯度檢查。
- 五組smoke訓練與報告匯出成功；第三份15節教材fresh-kernel smoke通過。
- 正式15runs於核心/協定commit476518c後啟動，背景執行中；未使用test。設定見classical-feature-protocol.md。
- Study classical-features-66658d6e1b完成15/15，報告核對完成狀態、condition/seed及固定參數delta後匯出。傳統可訓練最後train90%，量子61.67%；validation平均17.27%對15.51%，逐seed兩好一壞。限制見reports/classical-features/interpretation.md。

## 第九輪：量子feature尺度（完成）
- 44項pytest通過，含同初始化倍率隔離、有限量子梯度、scaled敏感度、Tanh飽和、無效倍率及Frozen delta檢查。
- 第三份14節教材fresh-kernel smoke通過；2×2四組smoke與報告匯出通過。
- 正式12runs於核心/協定commit206f1aa後背景啟動，未使用test；結果待完成。
- Study feature-scale-e11a85e95b完成12/12，報告核對completed、倍率、train_quantum與Frozen delta後匯出。1000×可訓練最後train平均61.67%對固定50.00%，但validation accuracy逐seed較低；完整限制見reports/feature-scale/interpretation.md。首次報告命令逾時，重試只重建報告，未重跑訓練。

## 第八輪：量子敏感度與消融（完成）
- 41項pytest通過；固定量子參數每epoch delta=0、mapping更新，以及敏感度不改原模型/RNG均驗證。
- 第三份教材13節fresh-kernel smoke通過；配對smoke與報告匯出通過。
- 正式6runs於核心/協定commit58cd77a後背景啟動，未使用test；結果待完成。
- Study quantum-ablation-11d5b7a1fc完成6/6，報告逐run核對completed、train_quantum旗標與Frozen delta=0後成功匯出。兩組best-checkpoint validation accuracy逐seed相同，未見量子訓練增量收益；量子feature置零在所測checkpoint的前向作用很小。詳見reports/quantum-ablation/interpretation.md。

## 第七輪：fan-in尺度（完成）

- 39項pytest通過；新增目標std、bias保留、一般block梯度及常數block有限backward測試。
- 第三份教材擴為12節，fresh-kernel smoke通過；fan-in兩組實驗與報告整合smoke通過。
- 正式6runs於核心及協定commit 0f63870之後啟動，背景執行中，未讀test。設定詳見fanin-protocol.md。
- 正式study fanin-104276cae2完成6/6，逐run完成狀態/transform核對及報告匯出成功；平均最後train51.67%→21.67%，validation結果混合，完整限制見reports/fanin/interpretation.md。

## 第六輪：生成權重中心化（完成）

- 36項pytest通過；新增中心化保持std/bias、梯度連接及diagnostic配置保存測試。
- 三份notebooks fresh-kernel smoke通過（11/9/11節）；第三份新增每層weight/bias及preactivation統計。
- Centering兩組smoke及報告匯出整合檢查通過。
- 正式6runs在核心及協定commit 6b34131之後背景啟動，結果待完成；設定見centering-protocol.md。
- Study centering-7f231401b6完成6/6，報告匯出成功。中心化平均最高/最後train為60.00%/51.67%，原始46.67%/33.33%；但seed22退步，三seed最佳validation loss皆較差。見reports/centering/interpretation.md。

## 第五輪：Learning rate

- 實驗與協定在commit 678e462提交後啟動，正式9runs背景執行中；不使用test。
- 34項pytest通過，包含驗證設定實際傳入Adam、寫入artifact，以及無效LR不建立run。
- LR三組smoke及報告匯出整合檢查通過。
- 第三份教材擴為10節，新增學習率、時間波動與未收斂的判讀；fresh-kernel smoke通過。
- 正式study learning-rate-f37260840c完成9/9runs，報告成功匯出。較低LR減少後20epochs loss時間波動，但未改善同200epochs預算的平均擬合或validation表現；具體數字與限制見reports/learning-rate/interpretation.md。

## 第四輪：固定LeakyReLU的mapping比較

- 固定depth3、affine/Tanh、相同seeds11/22/33、20筆train、200epochs；未使用test。
- 32項pytest通過，新增加性/非加性/零矩陣結構指標測試。
- 第三份notebook擴為9節，新增權重加性結構與奇異值教學；fresh-kernel smoke通過。
- 首次前景執行因200秒上限中斷，保留results/leaky-mapping-cb2f5b230f，不納入正式比較；完整矩陣改背景執行，結果待完成後匯出。
- 背景study `leaky-mapping-b5be7b4cfc` 6/6完成，報告匯出成功。Tanh最高train平均46.67%，affine21.67%；validation平均約15.60%與15.13%，沒有穩定泛化改善。完整限制及不穩定現象見reports/leaky-mapping/interpretation.md。

## 第三輪：目標activation

- 31項pytest通過；新增負區域梯度、snapshot與真實forward一致、protocol保存測試。
- 三份notebook fresh-kernel smoke通過（11/9/8節）；第3份新增ReLU/LeakyReLU教學。
- 正式6-run target-6d2e608121完成；另做6次train-only Classical tiny控制。結果與限制見reports/target-activation/interpretation.md。
- 保留舊欄位relu_zero_fraction，定義為若使用ReLU時的零比例（preactivation≤0）；新增hidden_zero_fraction表示實際activation的零比例。LeakyReLU下兩者不同。

## 第二輪：訓練診斷與 mapping 延伸

- 新增 affine/Tanh mapping 配置；預設 affine，原實驗配置與模型 state_dict 結構保持相容。
- 新增4項測試：兩组相同初始Linear/量子參數、snapshot不改模型與RNG、diagnostic runner只接train/validation、新配置驗證。
- 全套28項pytest通過。
- 三份notebook fresh-kernel smoke均通過（11、9、7節）。第三份包含預測分布、受控mapping比較、20筆過擬合、整段訓練梯度與activation、結果判讀。
- 正式第二輪協定見 mapping-protocol.md；是否完成與結果以 reports/mapping 的實測報告為準。
- 第二輪 `mapping-710a60bdbc` 已完成12/12 runs；report_mapping.py成功匯出summary、逐epoch manifest與兩張圖，三seed完整。所有run最後epoch固定probe的ReLU零比例1.0、logits跨圖片標準差0.0；Tanh未改善。完整解讀見 reports/mapping/interpretation.md。

## 2026-10-07 第一輪

- 在新建獨立 Git repo 的 `feat/quantumtrain-evaluation` 分支實作。
- Python 3.11.15、PennyLane 0.45.1、Torch 2.14.1+cpu；完整相依套件見 uv.lock。
- pytest 最終回歸：24 passed，涵蓋 data split、functional MLP、有限差分、terminal phase、已知 X 態 bit ordering、detach 反例、診斷隔離、checkpoint、失敗保存、非有限梯度、reporting 完整性及 study pilot。
- Notebook 01：11 個教學 sections，fresh-kernel smoke 通過。
- Notebook 02：9 個教學 sections，fresh-kernel smoke 通過。
- 執行時 Jupyter/ZeroMQ 有 Windows event loop fallback 與本機 TCP transport 提示，未造成 cell 失敗；驗證過程不提供對外 Jupyter 服務。
- 完整資料 depth5、1 epoch validation-only pilot：train-only 1.08498 秒，val loss 2.30420。僅用作預算估計，未評估 test。
- 修正計畫中的 mapping 參數計數：480+8200+8040+41=16,761。
- 額外修正非有限梯度診斷：NaN norm 保存為 JSON null，保持 finite=false，避免錯誤診斷本身因嚴格 JSON 序列化而遺失；回歸測試由失敗到通過。
- 報告 manifest 同時保存 environment 與 diagnostics，讓分享的報告可追溯執行版本與量子更新證據。

## 教材閱讀檢視

正式矩陣 `study-8091e80c8e` 已完成12/12 runs，逐一檢查 completed、diagnostics passed 與 best.pt 存在；最終報告已從原始 artifacts 重建至 reports/。具體觀察與限制見 reports/interpretation.md。

已逐節檢查生成來源的20個 section：每節包含目標、理由、shape、程式定位、操作、觀察、判讀、問題及解答。Notebook 01 有可執行的資料圖、電路圖、shape trace、detach 反例與重載驗證；Notebook 02 有明確矩陣、validation-only pilot、缺漏判讀與報告產出。

這是作者／工程檢視，不等於已由真正零背景的新讀者完成 usability 測試。後續可請一位新人記錄卡住的 section，再迭代教材。
