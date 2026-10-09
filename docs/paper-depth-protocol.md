# r3 QT16/QT106參數—精度對照

固定既有10000train/200validation與索引，12000updates、batch128有放回、LR1e-4、eval每100、節點4000/8000/12000。QT106新跑seeds11/22/33，CNN/QT16引用paper-data-extended-d206b2e56a已完成結果，不當新獨立重複。

唯一架構變因為RY/CNOT blocks16→106，theta208→1378，mapping249固定；總457→1627，對6690目標模型比例6.831%→24.320%。r3 raw概率後處理、0/1bits、首層Tanh、全域weights/bias中心化固定。

相同seed但theta數量不同會消耗不同初始化亂數，**mapping初值不逐元素配對**；這是固定初始化政策下的架構比較，不是單独depth因果效應。同updates也不是同計算成本。CPU單步benchmark先暖機1步再量3步，估計僅train-only，不含validation/checkpoint，不是精確工期。

先驗指標：best-validation loss checkpoint的train/validation、final、固定節點、參數比例與實測時間；CNN基準用于accuracy gap。未使用official test，不直接對比論文test93.81%稱成功。

每100updates保存完整latest.pt；中斷後 `uv run python scripts/paper_depth_study.py --resume-root results/<study>` 續訓，協定/資料摘要不合拒絕。官方作者設定未完全對齊，仍為paper-guided r3。
