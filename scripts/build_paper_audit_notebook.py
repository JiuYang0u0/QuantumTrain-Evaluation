"""Teach source alignment before claiming paper reproduction."""
from build_notebooks import section,write,SETUP

cells=section('0. 我們真正要重現什麼？','將貢獻拆成參數比例和準確率差距。','之前610參數小模型的生成器更大，沒有測到壓縮主張。',
 '論文Tables1–3 → M、P_mapping、P_quantum、accuracy。','ratio=(mapping+quantum)/M；生成M個weights不表示M個獨立可訓練參數。',
 '執行環境cell，再閱讀固定v2來源。此notebook不訓練、不下載MNIST。','明確區分published與本機observed。','把最小參數數量與最大accuracy配在一起會虛構trade-off。',
 '前面小模型失敗能否否定論文壓縮？','不能；既沒有壓縮比，也沒有對齊論文資料/電路/模型。',SETUP)
cells+=section('1. 計數與取捨表','用程式核對paper宣稱。','參數口徑要包括mapping，不能只計量子角度。',
 'MNIST mapping14→4→20→4→1=249；13×L量子參數。','QT16=249+208=457，QT106=249+1378=1627；準確率引用而非本機實測。',
 '列印參數比例與accuracy gap，找出最小參數和較高準確率各是哪一列。','QT16的MNIST accuracy81.08%，不是QT106的93.81%。','少參數不自动等於少總記憶體、快訓練；推論若展開完整weights仍有M個數值。',
 'CIFAR10較接近baseline的是哪個設定？','QT323：23258/285226≈8.15%，60.69%對62.50%，差1.81pp；需1000epochs，非短測結果。',
 '''from qt_eval.paper_audit import compression_rows,public_notebook_count
claims=pd.DataFrame(compression_rows())
display(claims[['dataset','blocks','target_parameters','qt_parameters','parameter_ratio','published_qt_accuracy','accuracy_gap_pp']])
assert set(claims.evidence_type)=={'published_not_reproduced'}''')
cells+=section('2. 官方範例也需要版本對齊','辨認論文與公開notebook差異。','同一repository可包含後續變體，不能只憑檔名L16認定對應表格。',
 'RY/CNOT16blocks→208角度；U3/CU3版本→1248角度。','U3與CU3各3角度，每block13+13個gate：6×13×16；加mapping249為1497。',
 '讀差異表，列出仍需作者或固定程式確認的設定。','能指出gate、mapping activation、機率處理與test評估不一致。','目前只做靜態稽核，沒有執行TorchQuantum原notebook；數值結果仍未重現。',
 '換成main分支就能修正嗎？','此次查核main和chenyu_dev該MNIST檔案blob相同，不能解決差異。',
 '''print('Public notebook static parameter count:',public_notebook_count())
display(Markdown((ROOT/'docs/paper-reproduction-audit.md').read_text(encoding='utf-8')))''')
cells+=section('3. 開始訓練前的證據門檻','避免再以不同方法回答原論文問題。','精確設定有缺口，應先標示，而不是用常見寫法偷偷補齊。',
 '來源矩陣 → 選定重現或有假設的重建協定。','先MNIST6690/13qubits，驗證CNN等價、mapping249、壓縮比例、梯度及匯出純CNN一致性，再測試執行成本。',
 '核對未決電路初态/排序、mapping nonlinear位置、機率處理、初始化及評估；取得對應程式或明確採用重建假設後才長跑。','此階段得到可信audit，而不是新的accuracy。','50epochs全MNIST與19qubits深CIFAR是不同預算；smoke只能證明工程流程。',
 '重建時使用了合理假設，可稱精確重現嗎？','不可以；應列出假設，結果標成paper-guided reconstruction。',
 '''print('Status: source audit complete; exact numerical reproduction unresolved.')
print('MNIST target:',6690,'QT16:',457,'ratio:',457/6690)
print('Next: resolve paper/code mismatches before full training.')''')
write('05_paper_compression_audit.ipynb','# 工具書05：論文壓縮主張與來源稽核\n\n此頁準確率均為論文引用，不是本專案重現。',cells)
