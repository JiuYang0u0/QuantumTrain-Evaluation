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
write('06_paper_mnist_reconstruction.ipynb','# 工具書06：MNIST QT-16有假設重建\n\n建立正確壓縮配置，再驗證可微分訓練與純CNN匯出。',cells)
