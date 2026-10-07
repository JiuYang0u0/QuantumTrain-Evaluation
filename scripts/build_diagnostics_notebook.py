"""Generate the third teaching workbook."""
from build_notebooks import section,write,SETUP

cells=[]
cells+=section('0. 模型有更新，為什麼沒學會？','區分更新正確與任務有效。','第一輪 QT 約10.2%，現在診斷而非再次搜尋 test 最佳設定。',
 '(B,64) → 610 weights → hidden (B,8) → logits (B,10)。','本輪固定 depth3，研究 affine-only mapping 與 Tanh mapping。所有數據只來自 train/validation。',
 '先執行 setup，保持 smoke；正式模式使用 full。','顯示 CPU、float64 與模式。','這是第二轮探索；原測試集已被看過，不能當全新的盲測。',
 'loss.backward 成功是否等於學會？','不等於。還需要看訓練資料是否可擬合、logits 是否區分輸入及驗證表現。',SETUP)
cells+=section('1. 先看輸出與隱藏層','檢查是否塌縮成同一類。','平均 loss 看不到預測集中在哪裡。',
 '固定40筆 train probe → 10類計數、8個hidden活躍比例與權重/logits統計。','snapshot 不 backward、不消耗RNG，資料固定才能前後比較；ReLU負輸入變成0。',
 '看初始化預測圖與統計，再把圖和後續訓練結果比較。','可能出現初始化偏向單一類；這本身不是bug證明。','全部或大部分 ReLU=0 可能妨礙學習，但要配合梯度看，不能單憑一張圖定因。',
 'logits在不同圖片之間幾乎不變意味什麼？','目前模型沒有很好區分輸入；檢查權重與activation，不能只調電路深度。',
 '''from qt_eval.training_diagnostics import snapshot,run_mapping_study
data=load_digits_splits(smoke=MODE=='smoke')
probe=data['train'].tensors[0][:40]
torch.manual_seed(11)
model=QuantumTrain(3)
initial=snapshot(model,probe)
display(initial)
plt.bar(range(10),initial['prediction_counts']); plt.xlabel('Predicted digit'); plt.ylabel('Count on fixed train probe'); plt.show()
print('圖說：固定訓練probe的預測類別數量，不是測試準確率。')''')
cells+=section('2. 只改 Mapping 的非線性','建立受控的兩組模型。','多個Linear相乘仍是affine，生成610個權重的表達能力可能受限。',
 '兩組皆 (610,11) → (610,1)，相同16,761個mapping參數。','Tanh加在前三個Linear之後，最後層不加。重設相同seed，使量子與各Linear初始權重逐一相同。',
 '列印兩組架構並逐一assert初始參數一致。','參數相同但生成權重可以不同，因forward函數不同。','初始化尺度、LR與電路保持固定；若一起改，不能解釋是哪項改動的效果。',
 '為何不在這輪同時改電路？','先隔離mapping非線性這一個變因。',
 '''torch.manual_seed(11)
affine=QuantumTrain(3,mapping_activation='affine')
torch.manual_seed(11)
tanh=QuantumTrain(3,mapping_activation='tanh')
for a,b in zip(affine.parameters(),tanh.parameters()):
    torch.testing.assert_close(a,b)
print(affine.mapping); print(tanh.mapping)
print('Same initial parameters; generated weights identical?',torch.allclose(affine.generated_weights(),tanh.generated_weights()))''')
cells+=section('3. 小資料過擬合與完整訓練','先測能否記住每類2筆，共20筆。','連小資料都學不起來時，先檢查優化與表示能力，不急著擴大資料。',
 'tiny固定20筆、full固定train split；只提供validation，不傳test。','run_mapping_study：smoke兩種資料規模各2epochs、單seed；full為tiny200epochs＋完整train20epochs，各3seeds。',
 '執行凍結矩陣；正式可能需數分鐘。既有第一輪結果不改寫。','保存12個full runs或4個smoke runs、每epoch觀察與參數delta。','tiny不能保證100%訓練accuracy；max_train_accuracy只供記憶能力診斷，正式checkpoint仍依validation loss。',
 '把tiny結果的高訓練accuracy當泛化能力可以嗎？','不行；它只檢查能否擬合固定小資料。',
 '''study=run_mapping_study(OUTPUT/'mapping-studies',smoke=MODE=='smoke')
summary=pd.read_csv(study/'summary.csv')
display(summary)
print('Artifacts:',study)''')
cells+=section('4. 看梯度與activation隨時間變化','分辨初始化檢查與整段訓練。','有些模型開始能更新，後來卻停滯。',
 'history每epoch一列，gradient norm為batch平均，delta為epoch首尾差。','梯度、參數delta、ReLU零比例與logits跨圖片標準差搭配解讀。',
 '畫完整train情境seed11的兩組曲線；可改成tiny觀察。','兩條loss和gradient曲線来自本輪run，不混入舊結果。','epoch首尾delta可能因更新抵銷而小；不能直接等同每step沒有更新。',
 '量子梯度很小是否一定是barren plateau？','不是；先排除電路對称、資料、尺度及mapping。',
 '''fig,axes=plt.subplots(1,3,figsize=(14,4))
for activation in ('affine','tanh'):
    record=json.loads((study/f'{activation}-seed11-full'/'diagnostic.json').read_text())
    h=pd.DataFrame(record['history'])
    axes[0].plot(h.epoch,h.train_loss,label=activation)
    axes[1].plot(h.epoch,h.quantum_grad_norm,label=activation)
    axes[2].plot(h.epoch,[r['relu_zero_fraction'] for r in h.observation],label=activation)
for ax,label in zip(axes,['Train cross entropy','Quantum gradient norm (batch mean)','ReLU zero fraction on train probe']):
    ax.set(xlabel='Epoch',ylabel=label); ax.legend()
plt.tight_layout(); plt.show()
print('圖說：固定seed11，不是跨seed平均；train_loss在epoch結束後重新評估。')''')
cells+=section('5. 預測、權重與logits的變化','檢查模型是否學到輸入差異。','單一accuracy可能掩蓋class collapse或不活躍hidden units。',
 'initial/best/final observation → 權重統計、logits分布、prediction histogram。','best由validation loss決定；final是最後epoch，兩者不能混用。',
 '比較best checkpoint的預測和初始預測，查看每個hidden的活躍率。','可能觀察到非線性改善，也可能仍學不動；依實際輸出說明。','probe僅40筆train，分布不能直接代表完整validation或test。',
 '全部預測同一類就能證明程式有bug嗎？','不能，可能是表示或優化問題；搭配小資料實驗與梯度檢查定位。',
 '''for activation in ('affine','tanh'):
    record=json.loads((study/f'{activation}-seed11-full'/'diagnostic.json').read_text())
    print(activation,'best epoch:',record['best_epoch'])
    display(pd.DataFrame({'initial':record['initial']['weights'],'best':record['best_observation']['weights']}))
    display(record['best_observation'])''')
cells+=section('6. 統計與結論的界線','從驗證結果提出下一個可檢驗假說。','一次改動有收益，不代表已證明量子部分有幫助。',
 '各情境×activation×seed → 均值與樣本SD；smoke單seed無SD。','下一步才是固定量子參數消融。本轮沒有使用test，不產生新的test主張。',
 '彙整表格，寫出觀察、證據、限制各一句。','單seed std為NaN，表示N/A；full三seed只作描述統計。','不能用本輪validation結果反覆調整後稱validation為獨立最終評估。',
 'Tanh較好是否代表量子電路提供優勢？','不能；需固定量子參數與純傳統生成器等消融比較。',
 '''display(summary.groupby(['tiny','activation'])[['train_accuracy','val_accuracy','max_train_accuracy','train_seconds']].agg(['mean','std']))
print('Scope:', 'smoke pipeline only' if MODE=='smoke' else 'validation-only exploratory mapping comparison')
print('No test split was passed to diagnostic training.')''')
cells+=section('7. 負區域梯度：ReLU與LeakyReLU','解釋失活與仍有量子梯度如何同時存在。','當hidden全零，logits只剩生成b2；量子梯度仍可通過b2，但圖片差異消失。',
 '(20,64) → preactivation (20,8) → hidden → logits。','LeakyReLU在負區域保留.01斜率；snapshot的hidden_zero_fraction是實際輸出零比例，relu_zero_fraction則是preactivation≤0的比例，兩者不能混用。',
 '用相同seed比較初始化觀察，再閱讀第三輪報告；此cell不啟動正式研究。','LeakyReLU可能恢復輸入依賴，但不保證20筆能擬合。','這是改target模型的新協定，不能合併到原深度報告；真正受控比較應保持mapping及其他設定一致。',
 '為什麼第二輪Tanh沒用，不代表非線性mapping永遠沒用？','如果target hidden已經全零，mapping改善可能被遮蔽；需在可傳遞負區域梯度的target下再比較。',
 '''for target in ('relu','leaky_relu'):
    torch.manual_seed(11)
    observed=QuantumTrain(3,target_activation=target)
    s=snapshot(observed,probe)
    print(target,{k:s[k] for k in ['hidden_zero_fraction','relu_zero_fraction','logits_between_images_std']})
interpretation=ROOT/'reports/target-activation/interpretation.md'
if interpretation.exists():
    display(Markdown(interpretation.read_text(encoding='utf-8')))''')
cells+=section('8. 生成權重是否只是列與欄的偏移？','用矩陣結構解釋表達限制。','Affine mapping作用在索引bits上，可能生成高度相關的權重；這個限制不能單看參數總數。',
 '前512個weights → W1(8,64) → row/column加性近似與8個奇異值。','A=row_mean+column_mean-global_mean；relative=norm(W1-A)/norm(W1)。純加性矩陣rank至多2，但量子機率特徵可能提供額外結構。',
 '先計算初始化權重的加性殘差；完整研究固定LeakyReLU後對照affine/Tanh，執行命令見README。','殘差小表示近似加性；殘差大不保證分類好。','這裡是初始化教學值，不能當最佳checkpoint的研究結果；rank或殘差不是量子優勢指標。',
 '為什麼在LeakyReLU下重新比較mapping？','第二輪可能被target ReLU失活遮蔽；固定可傳遞負區域梯度的target，可更清楚測量mapping變因。',
 '''for mapping in ('affine','tanh'):
    torch.manual_seed(11)
    m=QuantumTrain(3,mapping_activation=mapping,target_activation='leaky_relu')
    with torch.no_grad():
        w=m.generated_weights()[:512].reshape(8,64)
        additive=w.mean(0,keepdim=True)+w.mean(1,keepdim=True)-w.mean()
        print(mapping,'initial nonadditive relative:',float((w-additive).norm()/w.norm().clamp_min(1e-15)))
        print('singular values:',torch.linalg.svdvals(w).tolist())
saved=ROOT/'reports/leaky-mapping/interpretation.md'
if saved.exists():
    display(Markdown(saved.read_text(encoding='utf-8')))''')
cells+=section('9. 小學習率比較穩，還是只是學得慢？','區分訓練波動、退化與尚未收斂。','第四輪部分seed最高準確率和最後準確率差很多，因此單看最好epoch會誤導。',
 'epoch history → peak-to-final drop、最後20epochs loss的時間標準差。','第五輪只改Adam learning rate .001/.0003/.0001，固定Tanh/LeakyReLU、20筆、200epochs、三seeds。跨seed SD與同一run的時間SD不是一回事。',
 '先用下面的當次smoke資料學會計算指標；正式研究命令見learning-rate-protocol.md，不在本cell自動啟動。','印出drop與時間波動。若研究報告已存在，再讀其結果。','smoke只有2epochs，不能判斷後期退化；小LR波動小也可能完全沒學會，須同時看accuracy及loss。',
 '如果小LR最後20epochs仍緩慢下降，可以宣稱已收斂嗎？','不能；本輪僅比较同200epochs預算，不代表不同LR都達收斂。',
 '''import numpy as np
d=json.loads((study/'tanh-seed11-full'/'diagnostic.json').read_text())
acc=[r['train_accuracy'] for r in d['history']]
loss=[r['train_loss'] for r in d['history']]
print({'peak_to_final_drop':max(acc)-acc[-1], 'tail_loss_std':float(np.std(loss[-20:]))})
display(Markdown((ROOT/'docs/learning-rate-protocol.md').read_text(encoding='utf-8')))
lr_report=ROOT/'reports/learning-rate/interpretation.md'
if lr_report.exists():
    display(Markdown(lr_report.read_text(encoding='utf-8')))''')
cells+=section('10. 每層權重偏移與中心化','區分平均偏移、尺度與bias。','digits輸入為非負，W1共同偏移經64維加總可能使hidden整體偏正或偏負。',
 '生成610參數分成W1(8,64)、b1(8)、W2(10,8)、b2(10)。','中心化只把兩個weight矩陣各自減平均，bias不變、std不變。這與每個neuron减平均或fan-in scaling不同。',
 '在相同seed初始化的模型上查看四個block及preactivation；比較none/center_weights。','W1/W2均值接近零，std維持不變；preactivation仍受bias和輸入結構影響，不一定為零。','初始化變得均衡不代表訓練一定改善；正式結果仍需完整6runs與validation。',
 '為什麼不把所有610個參數一起減平均？','不同層及bias角色不同；全域中心化不能隔離每層權重共同偏移。',
 '''for transform in ('none','center_weights'):
    torch.manual_seed(11)
    m=QuantumTrain(3,mapping_activation='tanh',target_activation='leaky_relu',weight_transform=transform)
    observed=snapshot(m,probe)
    print(transform)
    display(pd.DataFrame(observed['blocks']).T)
    print('Preactivation:',observed['preactivation'])
display(Markdown((ROOT/'docs/centering-protocol.md').read_text(encoding='utf-8')))
saved=ROOT/'reports/centering/interpretation.md'
if saved.exists():
    display(Markdown(saved.read_text(encoding='utf-8')))''')
cells+=section('11. Fan-in尺度：初始化與重參數化不同','理解尺度目標與穩定分母。','中心化只移除偏移，不保證不同層的weight尺度適合其輸入維度。',
 'W1 fan_in64、W2 fan_in8 → 目標std 1/sqrt(192)、1/sqrt(24)。','center_fanin每次forward都中心化並縮放；分母sqrt(max(mean(centered²),1e-12))。bias不變，梯度穿過分母。',
 '以同seed印出center_weights與center_fanin的分層統計；看協定的數值邊界。','一般非退化block會達到指定std；常數block仍是零，不能假裝有變異。','這不是只改初始化，而是持續限制尺度的可微分重參數化；不能直接把結果歸因於更好的初始值。',
 '為什麼先floor variance再sqrt？','避免零點sqrt導數不穩定；clamp也會改變小variance區域的梯度，必須揭露其門檻。',
 '''for transform in ('center_weights','center_fanin'):
    torch.manual_seed(11)
    m=QuantumTrain(3,mapping_activation='tanh',target_activation='leaky_relu',weight_transform=transform)
    print(transform)
    display(pd.DataFrame(snapshot(m,probe)['blocks']).T)
display(Markdown((ROOT/'docs/fanin-protocol.md').read_text(encoding='utf-8')))
saved=ROOT/'reports/fanin/interpretation.md'
if saved.exists():
    display(Markdown(saved.read_text(encoding='utf-8')))''')
cells+=section('12. 有梯度，量子特徵卻可能影響很小','分辨連接、敏感度與訓練貢獻。','既然一般調參仍難擬合，先問量子分支是否真正影響weights/logits。',
 'theta(3,10)擾動 → 1024個機率 → 610個權重 → (20,10)logits與loss。','measure_sensitivity在副本上測一個固定單位方向及feature置零；Frozen訓練是固定初始機率、只更新mapping。',
 '先看初始化的擾動結果；正式六組訓練用協定命令執行，此cell只做觀察。','差值可以非零卻非常小；不能只以有grad判定有任務收益。','單方向不是全局Jacobian；置零是未重訓干預。Frozen的null grad是預期，不可當detach bug；量子delta須恆為0。',
 'Frozen仍能分類，是否表示完全不需要量子？','它仍使用初始量子機率；需另有zero/random/classical特徵與重訓baseline才可判斷更廣的作用。',
 '''from qt_eval.sensitivity import measure_sensitivity
torch.manual_seed(11)
m=QuantumTrain(3,mapping_activation='tanh',target_activation='leaky_relu',weight_transform='center_weights')
sx,sy=data['train'].tensors
ids=torch.cat([torch.where(sy==i)[0][:2] for i in range(10)])
evidence=measure_sensitivity(m,sx[ids],sy[ids])
display(pd.DataFrame(evidence['perturbations']))
print({k:v for k,v in evidence.items() if k!='perturbations'})
display(Markdown((ROOT/'docs/quantum-ablation-protocol.md').read_text(encoding='utf-8')))
saved=ROOT/'reports/quantum-ablation/interpretation.md'
if saved.exists():
    display(Markdown(saved.read_text(encoding='utf-8')))''')
cells+=section('13. 放大量子特徵：尺度改善不等於量子學習收益','用2×2設計分離尺度與參數更新。','量子feature std約1e-4，bit編碼為±1；固定倍率可能改變mapping的依賴方式。',
 '610個後處理量子特徵乘1或1000 → (610,11)mapping輸入；其他欄不變。','固定倍率在中心化機率特徵後施加，同時放大量子分支梯度；不等於變動std標準化。',
 '觀察兩倍率的初始化feature std、梯度與Tanh飽和；完整12runs命令見協定。','同seed原始参数相同，但輸出與梯度可不同；放大太多可能飽和。','若Frozen放大也改善，不可將改善歸因於量子參數學習；需比較同倍率的Frozen/Trainable。',
 '量子feature變大後梯度變大，是學得更好的證明嗎？','不是；需看完整訓練、validation與飽和。固定optimizer LR不代表量子通路的有效更新尺度相同。',
 '''for scale in (1.,1000.):
    torch.manual_seed(11)
    m=QuantumTrain(3,mapping_activation='tanh',target_activation='leaky_relu',weight_transform='center_weights',feature_scale=scale)
    obs=snapshot(m,sx[ids])
    sense=measure_sensitivity(m,sx[ids],sy[ids])
    print(scale, {'feature_std':obs['scaled_feature_std'],'tanh_saturation':obs['mapping_tanh_saturation'],
                  'quantum_grad_norm':sense['quantum_loss_grad_norm']})
display(Markdown((ROOT/'docs/feature-scale-protocol.md').read_text(encoding='utf-8')))
saved=ROOT/'reports/feature-scale/interpretation.md'
if saved.exists():
    display(Markdown(saved.read_text(encoding='utf-8')))''')
cells+=section('14. 對照傳統特徵：參數量匹配仍不等於完全公平','分辨特徵存在、可訓練性與量子機制。','1000×量子參數更新改善train，還需要其他可訓練特徵來判斷是否為量子特有。',
 '610×30固定隨機基底 × 30個係數 → 610個feature；mapping輸入仍(610,11)。','五組共享mapping初值；非零feature初始std一致。傳統gain僅初始化校準一次，random fixed與classical trainable起始feature逐元素一致。',
 '列印各組feature std與可訓練參數量；完整15runs用協定命令執行，不在此cell自動跑。','zero的feature std為0，其他組近似相同；傳統與量子trainable均多30個參數。','匹配參數量不等於表達能力或運算成本匹配；隨機基底另占儲存。驗證差異不能直接宣稱量子優勢。',
 'random fixed與quantum frozen不同時，能歸因於量子學習嗎？','兩者都沒有訓練feature參數，差異來自表示及優化互動；需要同表示固定/可訓練的配對。',
 '''from qt_eval.feature_controls import make_controls
controls=make_controls(11)
for name,m in controls.items():
    with torch.no_grad():
        print(name, 'feature std:',float(m.feature_values().std(unbiased=False)),
              'trainable parameters:',sum(p.numel() for p in m.parameters() if p.requires_grad))
torch.testing.assert_close(controls['random_fixed'].feature_values(),controls['classical_trainable'].feature_values())
display(Markdown((ROOT/'docs/classical-feature-protocol.md').read_text(encoding='utf-8')))
saved=ROOT/'reports/classical-features/interpretation.md'
if saved.exists():
    display(Markdown(saved.read_text(encoding='utf-8')))''')
write('03_training_diagnostics.ipynb','# QuantumTrain 工具書03：有梯度卻學不會\n\n透過20筆過擬合測試與affine/Tanh對照，學習定位訓練問題。',cells)
