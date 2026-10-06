"""Build source workbooks from reviewable prose/code. No experiments run here."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]


def section(title,goal,why,shapes,principle,operation,observation,pitfall,question,answer,code):
    text=f'''## {title}

**本節目標**：{goal}

**為什麼需要這一步**：{why}

**輸入／輸出與 shape**：{shapes}

**關鍵原理與程式定位**：{principle}

**操作步驟**：{operation}

**預期觀察**：{observation}

**結果判讀與常見錯誤**：{pitfall}

**理解問題**：{question}

<details><summary>提示與解答</summary>

{answer}

</details>
'''
    return [nbf.v4.new_markdown_cell(text),nbf.v4.new_code_cell(code.strip())]


SETUP='''
import os, sys, json, time
from pathlib import Path
import torch, pennylane as qml
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from torch.utils.data import DataLoader
from qt_eval.config import load_config, validate_config
from qt_eval.data import load_digits_splits
from qt_eval.models import make_model, ClassicalMLP, functional_mlp
from qt_eval.quantum import QuantumTrain, make_qnode, basis_codes, probability_features
from qt_eval.diagnostics import check_training_step
from qt_eval.experiment import run_experiment, evaluate
from qt_eval.reporting import summarize_runs, write_report
from qt_eval.artifacts import SOURCE_COMMIT

# 教學預設 smoke；只有準備好正式預算後才手動改成 full。
MODE = os.environ.get('QT_EVAL_MODE', 'smoke')
ROOT = Path.cwd()
if not (ROOT / 'pyproject.toml').exists():
    ROOT = ROOT.parent
assert (ROOT / 'pyproject.toml').exists(), '請由專案根目錄或 notebooks 目錄啟動'
OUTPUT = Path(os.environ.get('QT_EVAL_OUTPUT', str(ROOT/'results'/'notebooks')))
CONFIG = load_config(ROOT/'configs'/('smoke.json' if MODE=='smoke' else 'depth_study.json'))
assert CONFIG['mode'] == MODE
torch.set_num_threads(CONFIG['torch_threads'])
torch.manual_seed(CONFIG['seed'])
print('MODE:', MODE, '| device: cpu | dtype: float64')
if MODE == 'smoke':
    print('工程檢查：小資料、短訓練，不可用來判斷模型研究表現。')
'''


def write(name,title,cells):
    notebook=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(title)]+cells)
    notebook.metadata.kernelspec={'display_name':'Python 3 (qt-eval)','language':'python','name':'python3'}
    notebook.metadata.language_info={'name':'python','version':'3.11'}
    (ROOT/'notebooks').mkdir(exist_ok=True)
    nbf.write(notebook,ROOT/'notebooks'/name)


def workbook():
    cells=[]
    cells+=section('0. 閱讀地圖與操作模式','知道閱讀顺序與執行預算。',
        '先固定環境與模式，才能分辨程式檢查和科學實驗。',
        '設定字典 → MODE 與結果目錄；此節尚不訓練模型。',
        '流程：資料 → 量子機率＋bit 編碼 → Mapping → MLP 權重 → logits → loss。前置知識：Python 函式、張量、基本反向傳播；量子基礎會在第 4 節示範。',
        '從專案執行 uv run jupyter lab，選目前 .venv 的 Python kernel；依序執行。第一次保持 smoke。重跑前用 Restart Kernel and Run All。',
        '印出 smoke、cpu、float64。完整第一輪時間將由第 7 節的小型實測估計。',
        '只重新執行中段可能沿用舊模型或舊設定；已保存輸出不代表本次執行。',
        '為什麼預設不啟動 full？','全新讀者應先確認環境與梯度，再投入正式運算；smoke 結果不能作為結論。',SETUP)
    cells+=section('1. 環境、來源與改動','辨認方法來源和這個作品自己的工作。',
        '框架版本與改動會影響再現性；來源標記不能只寫「用了量子」。',
        '套件與文件 → 版本表及原始碼參照。',
        'docs/provenance.md 記錄論文、固定 commit 與獨立適配；目前是 digits/MLP，不是原論文數值重現。',
        '檢查版本及 Python 路徑，打開來源說明。',
        'Python 位於本專案 .venv，PennyLane 與 Torch 可匯入。',
        '安裝在一個環境卻用另一個 kernel，會出現找不到 qt_eval；先核對 sys.executable。',
        '改用 digits 是否仍能稱為 CIFAR-10 結果重現？','不能；保留架構思想但改變任務，應稱適配實驗。',
        '''print(sys.executable)
print({'torch':torch.__version__, 'pennylane':qml.__version__, 'source_commit':SOURCE_COMMIT})
display(Markdown((ROOT/'docs/provenance.md').read_text(encoding='utf-8')))''')
    cells+=section('2. 資料、圖像與切分','看懂原始圖像如何變成模型輸入。',
        '資料洩漏會使比較失去可信度，必須先於訓練檢查。',
        '8×8 圖 → (N,64) float64；標籤 (N,) long。',
        'data.py 固定除以 16，分層 60/20/20；split seed 固定 2026。',
        '印出各 split 數量，確認交集為空，再看第一張圖。',
        'smoke 每 split 40 筆，圖像像素在 [0,1]。',
        '不能先看 test 再選模型。實驗 seed 改初始化，split seed 改資料分割，兩者作用不同。',
        '為什麼除以 16 不需要 fit？','16 是資料已知的像素上限，不是從完整資料估計出的統計量。',
        '''splits=load_digits_splits(CONFIG['split_seed'],smoke=MODE=='smoke')
print({k:len(splits[k]) for k in ('train','validation','test')})
index_sets=[set(splits['indices'][k]) for k in ('train','validation','test')]
assert all(not index_sets[i]&index_sets[j] for i in range(3) for j in range(i))
x,y=splits['train'].tensors
plt.imshow(x[0].reshape(8,8),cmap='gray',vmin=0,vmax=1)
plt.title(f'Digits train example; label={int(y[0])}')
plt.axis('off'); plt.show()
print(x.shape,y.shape,x.dtype,y.dtype)''')
    cells+=section('3. 傳統 MLP 與 610 個參數','手算參數量並確認兩種 forward 等價。',
        'QT 只改權重來源；下游 MLP 的結構必須相同，才能有意義地比較。',
        '(B,64) → (B,8) → (B,10)；B 是 batch size。',
        'models.py：64×8+8+8×10+10=610；CrossEntropyLoss 接 logits，不要先加 softmax。',
        '建立 baseline，flatten 權重，再對照 functional_mlp 的 logits。',
        '參數數量 610，兩個輸出的最大差為零或數值誤差內。',
        '重新建立 nn.Parameter 或 .data.copy_ 容易切開生成器的計算圖，切片與 reshape 才保留連結。',
        'ReLU 應放在哪裡？','第一個線性層後；最後層維持 logits。兩個比較組要一致。',
        '''baseline=ClassicalMLP().double()
flat=torch.cat([p.flatten() for p in baseline.parameters()])
print('Target parameters:',flat.numel())
print('Maximum difference:',float((baseline(x[:4])-functional_mlp(x[:4],flat)).abs().max().detach()))
print('Logits shape:',baseline(x[:4]).shape)''')
    cells+=section('4. 小電路與量測機率','理解 H、RY/RZ、CZ 與量測的關係。',
        '10 qubits 太大，不適合第一次看電路；先用 2 qubits 觀察 4 個機率。',
        'theta (L,2) → probabilities (4,)；L 是旋轉/CZ 重複層數。',
        'quantum.py 的 make_qnode：H 產生疊加，RY 改變振幅，RZ/CZ 改變相位；末端純相位操作不改計算基底機率。',
        '畫一層電路，比較追加 RZ/CZ 層前後機率。',
        '機率和約 1，追加末端相位層前後一致；正式比較用 1/3/5，末層為 RY。',
        '相位可能影響之後的旋轉；不能推論 RZ 在所有位置都沒用。這裡的層數不是硬體編譯深度。',
        '若末端 RZ 的梯度是零，能直接說程式壞了嗎？','不能。此量測對該參數不敏感；這與 grad=None 的中斷不同。',
        '''small=make_qnode(2)
theta=torch.tensor([[0.3,-0.7]],dtype=torch.float64,requires_grad=True)
print(qml.draw(small)(theta))
fig,ax=qml.draw_mpl(small)(theta); plt.show()
with_phase=torch.tensor([[0.3,-0.7],[0.9,0.4]],dtype=torch.float64)
print('Before:',small(theta).detach().numpy())
print('After: ',small(with_phase).detach().numpy())
torch.testing.assert_close(small(theta),small(with_phase))''')
    cells+=section('5. 從機率到 MLP 權重','沿著 shape 追蹤生成器。',
        '量子位元不是一個對一個儲存 CNN/MLP 權重；這是用機率特徵生成權重的架構。',
        '(L,10) → (1024,) → (610,1)；bit 編碼 (610,10)；合併 (610,11) → (610,1) → (610,)。',
        '2^9<610≤2^10，因此需 10 qubits。f(p)=(0.8 tanh(0.1×512×p²))^0.3 再中心化。p 已是機率；平方是來源的额外轉換。',
        '逐步印出 shape 與第 0 個權重。mapping 全為 Linear，16,761 個參數。',
        '生成的第 0 個數值就是 fc1.weight[0,0]；其值保有 grad_fn。',
        '前 610 個機率未重新正規化，只作特徵。多層 affine 可合併為 affine；生成器比 610 大，本例沒有壓縮優勢。',
        '可訓練參數與生成的權重，為什麼數量不同？','前者是 optimizer 更新的來源；後者由來源計算得到，是 target forward 使用的張量。',
        '''qt=QuantumTrain(CONFIG['q_depth'],CONFIG['q_delta'])
p=qt.qnode(qt.q_params)
f=probability_features(p[:610],10)
combined=torch.cat([qt.codes,f[:,None]],dim=1)
generated=qt.mapping(combined).flatten()
print({'probabilities':tuple(p.shape),'features':tuple(f.shape),'combined':tuple(combined.shape),'weights':tuple(generated.shape)})
print('First bit code / raw p / feature:',qt.codes[0],p[0].detach(),f[0].detach())
print('fc1.weight[0,0]:',generated[0].detach(),'grad_fn:',generated.grad_fn)
print('Mapping parameters:',sum(p.numel() for p in qt.mapping.parameters()))''')
    cells+=section('6. 梯度連接與 detach 反例','能分辨量子參數、梯度與生成權重。',
        'loss 下降可能只因 mapping 在更新，必須觀察量子分支。',
        'loss scalar → logits → generated weights → mapping → probabilities → q_params (L,10)。',
        'diagnostics.py 在副本跑單步，保留正式模型與 RNG；detach 反例故意切斷 qnode 輸出。',
        '比較正常和 detached 副本的診斷；不要對正式 qt 設 detach_quantum=True。',
        '正常 quantum/mapping 均有梯度與 delta；反例 quantum.has_grad=False，mapping 仍有梯度。',
        'requires_grad=True 不是計算圖連接證明；有非零梯度也不是量子優勢。有限差分測試見 tests/test_core.py。',
        '把 detached tensor 的 requires_grad 設為 True 能接回去嗎？','不能；它只是新的 leaf，並沒有到原量子參數的鏈式法則。',
        '''import copy
healthy=check_training_step(qt,x[:20],y[:20])
broken=copy.deepcopy(qt)
broken.detach_quantum=True
disconnected=check_training_step(broken,x[:20],y[:20])
print('Healthy:',healthy)
print('Detached:',disconnected)
assert healthy['status']=='passed' and disconnected['status']=='failed' ''')
    cells+=section('7. 最小訓練與成本預估','親自看到參數更新與梯度大小。',
        '在完整訓練前以一小步驗證回饋，也取得大致計時。',
        '(最多32,64) batch → scalar loss；step 前後 theta 相同 shape。',
        'Adam 使用 loss.backward() 的梯度更新；此小例子是正式模型的副本。',
        '印出 loss、quantum grad norm、最大參數改變與時間。',
        '有效參數會改變，估算只含 train step，不含 validation 與 I/O。',
        '第一次執行有 warm-up；單 batch 估計非硬體 benchmark，更非準確總時間。',
        '怎麼估算 smoke 的 train 時間？','步數×每步耗時只是粗估；還需模型初始化、驗證、測試與寫檔時間。',
        '''demo=copy.deepcopy(qt)
optimizer=torch.optim.Adam(demo.parameters(),lr=CONFIG['learning_rate'])
before=demo.q_params.detach().clone()
start=time.perf_counter()
loss=torch.nn.functional.cross_entropy(demo(x[:32]),y[:32])
loss.backward()
gradient_norm=float(demo.q_params.grad.norm())
optimizer.step()
step_seconds=time.perf_counter()-start
print({'loss':float(loss.detach()),'quantum_grad_norm':gradient_norm,'max_delta':float((demo.q_params.detach()-before).abs().max()),'step_seconds':step_seconds})
steps=((len(x)+CONFIG['batch_size']-1)//CONFIG['batch_size'])*CONFIG['epochs']
print('Approximate train-only seconds:',steps*step_seconds,'(batch size differs; rough estimate)')''')
    cells+=section('8. 訓練、驗證與測試','執行完整且無 test 選模的流程。',
        '单步成功不能代替完整訓練；驗證曲線能發現泛化與最佳 epoch 差異。',
        'CONFIG → unique run 目錄；metrics 每 epoch 一列。',
        'experiment.fit 只接 train/validation，最低 validation loss 保存 best.pt；run_experiment 載入後才評估 test。',
        '執行當前設定，查看原始紀錄並畫 loss。',
        'run status completed，兩條曲線有 epoch 軸；smoke 不要求 loss 單調下降或高 accuracy。',
        '訓練 loss 下滑且 validation 上升可能過擬合；小 smoke 資料不能判定方法有效。',
        '為什麼不用 test accuracy 挑 best epoch？','挑選過程會利用 test 資訊，最後 accuracy 不再是獨立評估。',
        '''run=run_experiment(CONFIG,OUTPUT/'workbook')
history=pd.read_csv(run/'metrics.csv')
display(history)
ax=history.plot(x='epoch',y=['train_loss','val_loss'],marker='o')
ax.set_ylabel('Mean cross entropy'); plt.show()
print('Caption: each point is a sample-weighted epoch loss; validation does not update parameters.')
result=json.loads((run/'result.json').read_text())
print(run,result)''')
    cells+=section('9. 保存與重載','找到 artifact 並重建相同預測。',
        '報告數字需要能回到具體模型和設定。',
        'config.json＋best.pt → 同一組 test logits 與 metrics。',
        'checkpoint 保存 state_dict；權重不是整個 pickled model，讀取自有產物使用 weights_only=True。',
        '重新建立模型、載入 best.pt，重新計算固定 test 結果核對。此處是再現確認，不能據此改參數。',
        '重載後 test loss/accuracy 與當次 result 一致。',
        'best.pt 不含 optimizer/RNG 狀態，不能稱為精確中斷續訓。',
        '只保存一個 accuracy 數字足夠嗎？','不足；還需要 split、設定、來源版本、checkpoint 與紀錄才能追溯。',
        '''print([p.name for p in run.iterdir()])
loaded=make_model(CONFIG)
checkpoint=torch.load(run/'best.pt',weights_only=True)
loaded.load_state_dict(checkpoint['model_state'])
repeated=evaluate(loaded,DataLoader(splits['test'],batch_size=CONFIG['batch_size']))
assert repeated==result['test']
print('Reload verified:',repeated,'best epoch:',checkpoint['epoch'])''')
    cells+=section('10. 小結與下一個實驗','把執行成功轉為可驗證的敘述。',
        '要能說明證據的範圍，而不是把量子名詞當成結論。',
        '診斷＋曲線＋版本 → 一份可追溯的學習摘要。',
        '研究延伸在 notebook 02；學習變因固定後才比較深度。',
        '閱讀方法限制，試著用自己的話描述三種參數，再進入第二份教材。',
        '能指出量子梯度健康檢查、目標參數 610、mapping 16,761，以及 smoke 的限制。',
        '本輪不是壓縮或 QPU 加速實驗。不要因成功跑通就宣稱量子優勢。',
        '最直接的下一個消融是什麼？','固定量子參數、只訓練 mapping；比較時維持相同初始化、資料與訓練預算。',
        '''display(Markdown((ROOT/'docs/methodology.md').read_text(encoding='utf-8')))
print('Next: notebooks/02_depth_evaluation.ipynb')''')
    write('01_quantumtrain_workbook.ipynb','# QuantumTrain 工具書 01：從量子機率到可訓練權重\n\n閱讀與動手並重；預設 smoke。每節提供目標、理由、shape、操作、判讀與練習。',cells)


def study():
    cells=[]
    cells+=section('0. 研究問題與閱讀地圖','明確定義要回答的問題。','在跑數據前先決定變因，避免事後挑故事。',
        '設定 → 電路深度比較；target 固定 64→8→10。','同 epoch 預算下，1/3/5 層的準確率與成本如何變化？延伸自 workbook 01，使用相同模組。',
        '先完成第一份教材，再執行 setup；預設 smoke。','印出模式；full 才會跑 4 組×3 seeds×20 epochs。',
        '這是小任務適配，深度與參數量同時變動，不能宣稱單獨的深度因果效應。',
        '研究的成功是否要求更深一定更好？','不需要；穩定地觀察沒有改善或成本增加也有價值。',SETUP)
    cells+=section('1. 凍結比較矩陣','理解固定項與變動項。','如果資料或 optimizer 一起變，難以解釋差異。',
        '4 組 × seeds → 4（smoke）或12（full）份配置。','固定 split_seed、LR、batch、epochs、mapping；seed 只控制初始化與 shuffle。',
        '列印全部矩陣；正式執行前檢查設定。','classical、quantum-depth1/3/5，full seeds=11/22/33。',
        '相同 seed 不代表不同深度模型具有完全相同初始化；這是重跑識別，不是等價模型配對。',
        '為什麼保持 split_seed 不變？','避免把不同測試資料難度混入初始化變異。',
        '''from qt_eval.study import study_configs, GROUPS
matrix=study_configs(CONFIG)
display(pd.DataFrame(matrix)[['mode','model','q_depth','epochs','seed','split_seed']])''')
    cells+=section('2. 用 pilot 估計成本','在完整矩陣前知道預算。','深電路的每步成本可能更高；先測再決定正式運算。',
        '最深設定的短 pilot → 每 epoch 時間與矩陣粗估。','pilot 用 full 資料但只1 epoch（smoke 則小資料）；固定 seed，結果不併入正式報告。',
        '執行1 epoch pilot；full 保持 validation-only，不讀 test；短測後不調參。','顯示 train-only 粗估，還有 validation、I/O 等額外成本。',
        '最深模型的估算不是精確總耗時；pilot 的 test 不用於設定決策。',
        '能用 pilot test 選深度嗎？','不能；本 pilot 只執行 fit(train,validation)，根本不接 test。',
        '''from qt_eval.study import pilot_training
pilot=pilot_training(CONFIG,OUTPUT/'pilot')
print(pilot)
print('Rough upper-style train-only estimate:',pilot['train_seconds']*CONFIG['epochs']*len(matrix),'seconds')''')
    cells+=section('3. 批次執行與 artifact','把矩陣變成可追蹤 runs。','逐 run 保存，才能保留失敗原因而不只剩最好的數字。',
        '配置列表 → manifest 與每次 run 目錄。','study.run_study 順序執行；報告只接此矩陣的 run 清單，不掃入舊 runs。',
        '執行當前模式；需真正完整實驗時在 setup 明確設 full 再 Restart & Run All。','每 run 顯示完成或失敗，保存 study.json 與獨立結果。',
        '重新執行會建立新 study，不覆蓋歷史。不可替換表現差的 seed。',
        '若有一個 run 失敗，該怎麼報告？','保留 failed 與原因，組別標 incomplete；先診斷而非只刪除該筆。',
        '''from qt_eval.study import run_study
study_dir=run_study(CONFIG,OUTPUT/'studies')
manifest=json.loads((study_dir/'study.json').read_text())
runs=[study_dir/p for p in manifest['runs']]
print('Study:',study_dir,'| attempted:',len(runs))''')
    cells+=section('4. 完整性與缺漏','辨認缺 seed、整組缺失與重複執行。','只平均成功的幾筆容易形成選擇偏差。',
        'run records → expected/completed/failed/missing_seeds。','reporting.summarize_runs 拒絕重複 group/seed 和不合協定；空組也顯示。',
        '查看摘要是否每組完整；smoke 必須顯式 opt-in。','smoke 每組 expected=1；full expected=3。','completed 不代表模型效果好，只代表流程與保存成功。',
        '一個 seed 的標準差能填0嗎？','不能；沒有跨執行變動的估計，應顯示 N/A。',
        '''summary=summarize_runs(runs,allow_smoke=MODE=='smoke',expected_groups=GROUPS)
display(summary)
print('All planned runs complete:',bool((summary.status=='complete').all()))''')
    cells+=section('5. 精度、曲線與時間','一起看任務結果與代價。','參數量少不代表訓練快，loss 好也不代表泛化好。',
        '每 epoch 紀錄與 final test → 曲線、均值、樣本標準差。','reporting.py 的圖表保留每個 seed，accuracy 的誤差棒是 ddof=1 標準差。',
        '先讀每組各次結果，再看自動報告；切勿用此 test 結果調整同一輪模型。','loss 軸是平均 cross entropy；時間單位為秒；單 seed 不畫誤差棒。',
        '若差距小於 seed 波動，結論應保守；三次不構成強統計證據。',
        '為什麼要留下單 seed 曲線？','平均會掩蓋不穩定或失敗的訓練過程。',
        '''from IPython.display import Image
report=write_report(runs,study_dir/'report',allow_smoke=MODE=='smoke',expected_groups=GROUPS)
for image in ('train_loss.png','val_loss.png','val_accuracy.png','accuracy_mean.png','train_seconds_mean.png'):
    display(Image(filename=str(report.parent/image)))
print('Caption: epoch curves are individual runs; bar charts aggregate completed runs. Accuracy 0–1; time seconds; error bars sample SD.')''')
    cells+=section('6. 梯度與成本的界線','避免把健康檢查誤認為研究突破。','有梯度只是更新所需條件，不保證量子分支有幫助。',
        'diagnostics＋parameter counts → 每組的更新證據。','量子、mapping 登記參數量與 target weights 分開；讀取每次 run 的診斷。',
        '比對 quantum norm/delta 與訓練時間。','量子組有有效更新；classical 沒有量子欄位。','backprop 理想模擬不等於 QPU；此 mapping 大於 target，不可稱壓縮。',
        '如果量子梯度很小，能直接叫 barren plateau 嗎？','不能。要先檢查初始化、對稱性、尺度、有效參數、連線，再設計專門的縮放實驗。',
        '''for path in runs:
    diagnostic=path/'diagnostics.json'
    if diagnostic.exists():
        print(path.name,json.loads(diagnostic.read_text())['groups'])
display(summary[['group','parameters','train_seconds_mean']])''')
    cells+=section('7. 匯出與再現','由既有結果重新產生相同數字。','報告不是手打表格；讀者需要能追到 runs 與配置。',
        '既有 artifacts → depth-study.md、summary.csv、manifest.json 與PNG。','write_report 不訓練，只重建報告；README 說明路徑。',
        '開啟 Markdown 報告，核對 manifest 中的來源 run ids。','數據與第4節一致，圖表來自第5節。','複製報告時一併保存 manifest 與圖表；本機完整 checkpoint 留在 results。',
        '為什麼不掃 results 下的所有目錄？','可能混入 pilot、不同設定或重複 seed；應用明確的矩陣 manifest。',
        '''display(Markdown(report.read_text(encoding='utf-8')))
print('Report:',report)
print('Configuration manifest:',report.parent/'manifest.json')''')
    cells+=section('8. 寫出有證據的結論','把結果轉成可核對的技術敘述。','作品價值在正確解釋，不在強行找到優勢。',
        '數據＋限制 → 觀察、證據、下一個實驗。','先描述哪組的精度和時間，再交代 seeds、模擬器與目標；不要推論未做的比較。',
        '閱讀限制；若是 full，依自己的實測寫三句觀察與一項下一步。','smoke 只得出流程檢查結論；full 才討論本設定的表現。','不能用「量子比較好」取代指定指標與條件；若接著改 mapping，需另起研究協定。',
        '最直接測量量子學習貢獻的下一步？','相同條件下固定量子參數，只訓練 mapping，做消融對照。',
        '''display(Markdown((ROOT/'docs/methodology.md').read_text(encoding='utf-8')))
print('Conclusion scope:', 'pipeline checks only' if MODE=='smoke' else 'this fixed digits protocol only')''')
    write('02_depth_evaluation.ipynb','# QuantumTrain 工具書 02：設計可信的電路深度實驗\n\n先固定問題與預算，再執行、檢查完整性、閱讀曲線並匯出。',cells)


if __name__=='__main__':
    workbook()
    study()
