# 固定10000筆，所有組統一延長至12000updates

依前輪歷史，QT seed22在3000→4000updates的validation loss2.30209→2.29507，最後batch量子gradient norm .0234→.205；seed11/33亦仍在改善。這是慢學習線索，不是已證明收斂或根因。

固定前輪10000 train、200validation索引、r3全部架構/初始化、Adam lr1e-4、有放回抽樣batch128、eval_every100。CNN與QT16各seeds11/22/33，共6runs，全部從頭跑12000updates（1536000次樣本處理）。不從best.pt接續：舊checkpoint沒有optimizer狀態。

4000/8000/12000保存當下checkpoint與validation；每100updates保留完整曲線、last-batch quantum gradient。前4000updates應重現前輪，不將其視為獨立新資料。主要比較固定節點及best/final；120次選模機會多於原40次，因此best表現不能只歸因於模型改善。

不挑seed延長、不改learning rate/資料/深度；不讀official test。這是相同資料與模型下的更新預算研究，不是完整論文MNIST重現。三seed及反覆用過的validation只供開發性觀察。

命令：uv run python scripts/paper_data_budget.py --extended。--smoke改單seed4updates。完成後用scripts/report_data_budget.py匯出；報告固定節點與逐seed變化，再決定是否需要來源追查，不自動新一輪。
