"""Teach paper-guided reconstruction independently from digits adaptations."""
from build_notebooks import section,write,SETUP
cells=section('0. 重建而非精確重現','先理解哪些設定由論文確認、哪些是公開假設。','上游MNIST notebook與表格gate及activation不符，需要明確版本。',
 '6690 target weights ← 208量子＋249mapping，共457個可訓練參數。','獨立paper_mnist.py不沿用舊digits的1000×、中心化或LeakyReLU。',
 '先執行setup並讀協定；本教材預設不下載資料、不啟動長程訓練。','能把參數計數與accuracy維持分開。','paper QT16是81.08%，不是QT106的93.81%；尚無本機相應accuracy。',
 '457參數是否已證明論文成功？','只確認架構壓縮；仍需可信accuracy和相同任務/預算。',SETUP)
cells+=section('1. 參數與CNN形狀','逐層核對6690與457。','Target與生成版要共用運算才能比較。',
 '(B,1,28,28)→(B,8,12,12)→(B,12,4,4)→(B,192)→20→10。','兩conv k5與pool2；沒有額外ReLU，pool本身非線性。mapping首層Tanh。',
 '列出target參數形狀與QT獨立變數數量。','CNN6690、mapping249、theta208。','展開後CNN仍有6690weights，壓縮口徑是獨立可訓練變數。',
 '13qubits為什麼足夠？','2^12<6690≤2^13，取8192機率的前6690作features。',
 '''from qt_eval.paper_mnist import TargetCNN,PaperQuantumTrain
torch.manual_seed(11)
target=TargetCNN().double()
qt=PaperQuantumTrain()
print([(n,tuple(p.shape),p.numel()) for n,p in target.named_parameters()])
print('Target:',sum(p.numel() for p in target.parameters()))
print('Mapping:',sum(p.numel() for p in qt.mapping.parameters()),'QNN:',qt.theta.numel())
print('Parameter ratio:',sum(p.numel() for p in qt.parameters())/6690)
display(Markdown((ROOT/'docs/paper-mnist-protocol.md').read_text(encoding='utf-8')))''')
cells+=section('2. 梯度與純傳統推論','驗證輸出CNN不需QPU或QNode。','訓練參數壓縮及classical deployment是兩項不同貢獻。',
 '合成(B=2,1,28,28)→logits(2,10)，生成weights可載入獨立CNN。','QNode Torch backprop→mapping→functional conv；export_classical產生脫離量子通路的state_dict。',
 '只用合成資料檢查backward與export logits相同；不解讀其loss為MNIST成果。','theta.grad非零有限，匯出前後最大差接近0。','匯出完整CNN消除了推論量子依賴，但不減少其6690權重儲存。',
 'checkpoint只存457變數與匯出6690weights有何不同？','前者是生成表示，需要重建；後者可直接傳統推論。',
 '''torch.set_num_threads(1)
x=torch.randn(2,1,28,28,dtype=torch.float64)
y=torch.tensor([0,1])
loss=torch.nn.functional.cross_entropy(qt(x),y)
loss.backward()
print('Synthetic gradient norm:',float(qt.theta.grad.norm()))
exported=qt.export_classical()
with torch.no_grad():
    torch.testing.assert_close(qt(x),exported(x),atol=1e-10,rtol=1e-9)
    print('Export delta:',float((qt(x)-exported(x)).abs().max()))''')
cells+=section('3. MNIST pilot與預算','以真實資料驗證資料路徑及成本。','13qubits/16blocks比舊小電路昂貴；先pilot再決定長跑。',
 '官方train60000→固定分層1000 train/200 validation；只2epochs。','mnist_data.py只下載train IDX、檢查MD5與SHA256紀錄，不下載official test。',
 '需要跑pilot時從根目錄執行下列命令；閱讀產物與估計時間。此cell只顯示命令。','結果必須標記pilot；不能與論文完整50epoch直接比較。','外推時間不含validation/I/O，非保證值；不因pilot準確率差立即改多項方法。',
 '為何不直接跑CIFAR QT323？','19qubits/323blocks/1000epochs成本差異巨大，先完成較小且有壓縮比的MNIST設定驗證。',
 '''print('uv run python scripts/paper_mnist_pilot.py')
print('Protocol: docs/paper-mnist-protocol.md')
print('Artifacts: results/paper-mnist-pilot-*/{classical_cnn,qt16}/')
print('This workbook used synthetic data for correctness, not paper accuracy reproduction.')''')
cells+=section('4. r2全域中心化：追溯缺少的來源處理','理解loss爆大的前向原因。','r1原始生成weights的共同偏移經CNN累積放大；上游MNIST code有全域中心化。',
 '6690個生成weights/bias一起減平均 → 同架構CNN logits。','不同於舊digits的逐層weight中心化；r2只加此一步，457參數不變。',
 '比較相同seed、合成probe的r1/r2權重均值及logit尺度，再讀真MNIST診斷協定。','r2的全域均值近零，但不保證各層均值都零，也不保證準確率高。','小loss並不代表學到；這是去除數值放大的一步，非論文復現成功。',
 '為何要保留r1而不是直接覆寫？','因為重建假設變了；保留對照才能追溯哪個來源處理影響數值。',
 '''torch.manual_seed(11)
r1=PaperQuantumTrain()
torch.manual_seed(11)
r2=PaperQuantumTrain(global_center=True)
for name,m in [('r1',r1),('r2',r2)]:
    with torch.no_grad():
        w=m.generated_weights()
        print(name,'weight mean',float(w.mean()),'logits absmax',float(m(x).abs().max()))
display(Markdown((ROOT/'docs/paper-centering-protocol.md').read_text(encoding='utf-8')))
report=ROOT/'reports/paper-centering/interpretation.md'
if report.exists():display(Markdown(report.read_text(encoding='utf-8')))''')
cells+=section('5. 長一點的pilot：相同epochs不是相同資料預算','知道50epochs小子集仍不是論文50epochs。','數值穩定後要觀察loss與更新的持續性，不能只以短測低accuracy定論。',
 '1000train / batch128 → 每epoch8steps，50epochs400steps。','論文60000train每epoch469steps，50epochs23450steps；還有不同樣本覆蓋，不能只對齊steps代替資料。',
 '閱讀凍結協定；正式命令只顯示不在本教材自動啟動。結果可用--report從既有study產生。','更新紀錄分量子/mapping，norm是最後batch，不是整epoch平均；best/final分開。','參數比6.83%與accuracy gap分開報；validation不是官方test。',
 '如果只有mapping更新、theta完全不動，該做什麼？','先看theta梯度是否None、有限或接近零，以及步驟delta；不要只根據loss下降宣稱量子有學習。',
 '''display(Markdown((ROOT/'docs/paper-r2-learning-protocol.md').read_text(encoding='utf-8')))
print('uv run python scripts/paper_r2_learning_pilot.py')
print('Pilot updates:',50*8,'paper-size updates:',50*469)
saved=ROOT/'reports/paper-r2-learning/interpretation.md'
if saved.exists():display(Markdown(saved.read_text(encoding='utf-8')))''')
cells+=section('6. r3機率前處理：p已經是機率','分辨振幅平方與機率再次平方。','上游TorchQuantum先取振幅abs平方；PennyLane qml.probs已完成這步，移植時不能再平方。',
 '8192個p → 前6690個 → f=(.8*tanh(.1*4096*p))^.3 → f-mean(f)。','r3只替換mapping最後一個feature欄；0/1bits、mapping與全域生成中心化不變。',
 '印出raw/upstream的feature尺度，確認相同參數量，讀凍結配對協定。','後處理改變feature與梯度尺度；不能預設accuracy提升。','p=0的分數冪導數可能奇異；本輪未加epsilon，若non-finite就停止，不無聲改原式。',
 '為什麼不是沿用之前digits的p²？','那是另一份範例的路徑；本輪對齊MNIST來源，必須追溯資料語意而非複製相似公式。',
 '''for processing in ('raw','upstream'):
    torch.manual_seed(11)
    m=PaperQuantumTrain(global_center=True,probability_processing=processing)
    with torch.no_grad():
        p=m.qnode(m.theta);f=m.mapping_inputs(p)[:,-1]
        print(processing,'p min',float(p.min()),'feature mean/std',float(f.mean()),float(f.std(unbiased=False)))
display(Markdown((ROOT/'docs/paper-r3-protocol.md').read_text(encoding='utf-8')))
saved=ROOT/'reports/paper-r3/interpretation.md'
if saved.exists():display(Markdown(saved.read_text(encoding='utf-8')))''')
cells+=section('7. 固定資料增加更新：不要混淆樣本與步數','用預先指定節點觀察學習進度。','r3的best在50epoch上限，下一個單變因是訓練時間而非架構。',
 '1000筆/batch128=8steps，50/100/200/500epochs對應400/800/1600/4000updates。','從頭重跑相同seed、r3設定；節點保存當下模型，best.pt另存validation loss最低者。',
 '讀協定與命令，正式背景長跑不在此cell啟動；結果完成後讀milestones.csv。','第50節點可核對前輪r3，後續節點看loss與accuracy是否仍改善。','4000updates仍只重複1000筆，不能當作看到更多獨立資料；best與最後指標須分開。',
 '節點更高train卻更低validation怎麼解讀？','可能過擬合或波動，需看同checkpoint的train/validation、loss及三seed，不能只挑高值。',
 '''print([(epoch,epoch*8) for epoch in (50,100,200,500)])
display(Markdown((ROOT/'docs/paper-r3-budget-protocol.md').read_text(encoding='utf-8')))
print('uv run python scripts/paper_r3_budget.py')
saved=ROOT/'reports/paper-r3-budget/interpretation.md'
if saved.exists():display(Markdown(saved.read_text(encoding='utf-8')))''')
cells+=section('8. 資料覆蓋：對齊updates與樣本處理次數','分開資料量與訓練預算。','固定epochs時更多資料也帶來更多updates，無法直接辨認資料覆蓋的作用。',
 '1000/10000 train，各4000updates×128=512000次樣本處理，同200validation。','有放回均勻抽樣確保每batch128，記錄unique samples seen；每100updates評估，兩組40次選模機會。',
 '讀協定；完整12runs需背景執行，此cell僅展示預算計數。','1000與10000集合共享validation但不假設nested；實際交集保存在protocol。','新抽樣與舊epoch shuffle不同，所以必須重跑1000控制；相同曝光次數不代表相同獨立資料量。',
 '為什麼不能拿上一輪1000結果直接對10000？','除了資料量，batch尾批與抽樣協定也改變；新一輪重新對齊控制才可解釋。',
 '''print('Examples processed per condition:',4000*128)
display(Markdown((ROOT/'docs/paper-data-budget-protocol.md').read_text(encoding='utf-8')))
print('uv run python scripts/paper_data_budget.py')
saved=ROOT/'reports/paper-data-budget/interpretation.md'
if saved.exists():display(Markdown(saved.read_text(encoding='utf-8')))''')
cells+=section('9. 慢啟動：延長所有seed而非只救最差一次','區分低表現、梯度斷線與慢學習。','10000筆seed22後段loss才開始下降，需更多固定預算驗證，不能只換seed。',
 '10000train固定，4000→12000updates，每次128；節點4000/8000/12000。','CNN/QT各三seed全部從頭跑，保持抽樣流；舊best.pt沒optimizer狀態，不作精確resume。',
 '讀協定，完成後看固定節點與best/final；本cell不啟動長跑。','前4000可核對舊結果，後續觀察是否追上或仍停滯。','120次驗證比40次更容易選到好checkpoint，不能只報best提升；validation不是盲測。',
 '為何不只延長seed22？','選擇性增加最差或最好seed的預算會使組間比較不一致。',
 '''display(Markdown((ROOT/'docs/paper-data-extended-protocol.md').read_text(encoding='utf-8')))
print('uv run python scripts/paper_data_budget.py --extended')
print('Examples processed:',12000*128)
saved=ROOT/'reports/paper-data-extended/interpretation.md'
if saved.exists():display(Markdown(saved.read_text(encoding='utf-8')))''')
cells+=section('10. QT106：多保留參數能否換回精度？','把參數量與accuracy同時放入比較。','QT16的457參數不該配上論文QT106的93.81%；需要各自訓練並報ratio。',
 '16/106blocks → 208/1378theta，加249mapping=457/1627；目標仍6690。','固定10000train/12000updates，引用已完成CNN/QT16對照，新跑QT106三seed。',
 '先手算比例，再讀協定；深電路正式執行用腳本，本cell不跑106block訓練。','參數比例6.83%→24.32%，與計算成本同時上升。','同seed不代表mapping初值一樣，因theta大小耗用亂數不同；同updates不等於同時間。',
 '更多參數得到更高accuracy是否等於量子優勢？','不是；這是壓縮—精度曲線，还需傳統壓縮對照及獨立評估。',
 '''print({'QT16':457/6690,'QT106':1627/6690})
display(Markdown((ROOT/'docs/paper-depth-protocol.md').read_text(encoding='utf-8')))
print('uv run python scripts/paper_depth_study.py')
print('Resume: uv run python scripts/paper_depth_study.py --resume-root results/<study>')
saved=ROOT/'reports/paper-depth/interpretation.md'
if saved.exists():display(Markdown(saved.read_text(encoding='utf-8')))''')
write('06_paper_mnist_reconstruction.ipynb','# 工具書06：MNIST QT-16有假設重建\n\n建立正確壓縮配置，再驗證可微分訓練與純CNN匯出。',cells)
