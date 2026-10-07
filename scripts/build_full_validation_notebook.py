"""Stage-two guided workbook; reuse core training and reporter."""
from build_notebooks import section,write,SETUP

cells=[]
cells+=section('0. 從小資料診斷到完整資料','區分能記住20筆與解決完整任務。','前十輪找到多項瓶頸，下一步應固定架構而不是無限制調參。',
 '固定train/validation切分 → 四組×三seeds；smoke為四組×單seed。','read docs/research-summary.md追蹤每個決策；validation是開發資料，不是未看過的盲測。',
 '先保持smoke，順序執行；full會跑12runs。','印出模式與來源環境。','原test第一輪已看過；本輪不讀test，也不聲稱獨立泛化證據。',
 '為何不直接用20筆結果排名？','不同資料量與更新步數會改變優化，局部擬合能力不等於完整任務表現。',SETUP)
cells+=section('1. 研究樹與凍結協定','理解四組如何被選出。','基準必須同時包含直接模型及傳統生成器，避免只有量子內部比較。',
 '生成器610個weights → 相同64→8→10 LeakyReLU目標MLP。','Direct 610參數；生成器mapping16,761加0或30feature參數。相同參數量不等於相同計算成本。',
 '閱讀總結及協定，核對20epochs、batch64、LR.001與三seeds。','生成器共享mapping初值，direct有不同參數化與初始化。','量子feature較有效不代表比傳統可訓練feature更有效。',
 'Frozen量子時間是否代表最優部署成本？','不是；目前沒有快取固定機率，本研究記錄的是此程式的實際時間。',
 '''display(Markdown((ROOT/'docs/research-summary.md').read_text(encoding='utf-8')))
display(Markdown((ROOT/'docs/full-validation-protocol.md').read_text(encoding='utf-8')))
from qt_eval.full_validation import models_for_seed
for name,m in models_for_seed(11).items():
    print(name, 'trainable:',sum(p.numel() for p in m.parameters() if p.requires_grad),
          'buffer elements:',sum(b.numel() for b in m.buffers()))''')
cells+=section('2. 執行矩陣與版本保存','完成新的獨立run目錄。','明確矩陣避免把不同協定的舊結果混在一起。',
 'protocol＋splits → 每run config/environment/metrics/best.pt/result。','run_condition只接受train和validation，不接受test；fit由validation loss選模型。',
 '先跑四組smoke確認功能；full模式才使用完整資料20epochs。','建立unique full-validation目錄，列出每run進度。','重跑會建立新study；不要用新的較好seed取代原結果。',
 '為什麼保存final和best兩種指標？','避免拿最後train與早期best validation當同一模型比較。',
 '''import sys
sys.path.insert(0,str(ROOT/'scripts'))
import full_validation_study as study_tools
full_study=study_tools.run(OUTPUT/'full-validation',smoke=MODE=='smoke')
print(full_study)''')
cells+=section('3. 報告與同checkpoint比較','正確讀取validation、train與時間。','checkpoint選取會影響報告，不能只取每種指標各自最高值。',
 '12runs（smoke4） → 均值/樣本SD與曲線。','best_train和best_validation在相同best.pt重新評估；final另列。Train loss曲線是訓練中平均，非整個epoch結束後的train loss。',
 '匯出並閱讀表格與曲線，檢查n與epochs。','smoke n=1時SD顯示N/A；full n=3。','同epochs不是同時間；CPU模擬不是QPU成本。Validation已參與架構選擇，不能稱test accuracy。',
 '表格best_train比final_train低是否一定有bug？','不一定；validation選的最佳checkpoint可能更早，final仍可能繼續擬合train。',
 '''study_tools.report(full_study,full_study/'report')
display(pd.read_csv(full_study/'report/summary.csv'))
from IPython.display import Image
display(Image(filename=str(full_study/'report/curves.png')))
print('Caption: epoch vs train/validation cross entropy; one line per condition/seed.')
display(Markdown((full_study/'report/study.md').read_text(encoding='utf-8')))''')
cells+=section('4. 完成階段總結','把結果放回研究問題。','下一步應由證據决定，不自動追逐更好數字。',
 '指標＋限制 → 可支持的敘述與未驗證的假說。','完整資料可能改變排名；我們只報凍結預算內的開發性結果。',
 '寫出觀察、限制、下一個確認實驗各一句；完整報告若已保存可在下方閱讀。','能區分工程成功、擬合成功、泛化證據與量子優勢。','即使Quantum勝出一次，也不足以排除傳統優化與不同預算的影響。',
 '作品集最可信的賣點是什麼？','可重現工具、梯度驗證、因果線索的受控對照及誠實報告，而不是預設量子優勢。',
 '''saved=ROOT/'reports/full-validation/interpretation.md'
if saved.exists():
    display(Markdown(saved.read_text(encoding='utf-8')))
else:
    print('Formal result interpretation not yet available; this notebook run remains local evidence.')''')
write('04_full_validation.ipynb','# QuantumTrain 工具書04：完整資料的開發性評估\n\n先整理十輪研究，再比較四種固定模型。',cells)
