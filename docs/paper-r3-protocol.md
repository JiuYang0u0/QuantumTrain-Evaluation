# r3：只加入公開MNIST程式的機率後處理

來源固定commit d5b654116565ae9550e540e021ab0d4160a15c86，MNIST torchquantum notebook QLayer.forward：state幅度abs平方得到p，再取前M，f=(.8*tanh(.1*2^(n-1)*p))^.3，最後f減平均。

本實作QNode已輸出p，所以**不可再對p平方**。MNIST n=13，2^(n-1)=4096。raw控制組直接p；upstream組上述f。兩者均維持r2的0/1bits、首層Tanh、RY/CNOT16blocks、全域生成weights/bias中心化、theta uniform[-pi,pi]。上游其他差異仍存在，不稱精確復現。

初始尺度/梯度在固定前128筆train probe量測。未加epsilon：p=0處的0.3次冪導數可能奇異；若遇非有限梯度停止並記錄，不無聲改公式。測試涵蓋正p公式、同初始化、有限量子梯度、參數數量与匯出一致性。

配對正式pilot：r2 raw/r3 upstream、seeds11/22/33、1000train/200validation固定索引、50epochs、batch128、Adam lr1e-4、CPU float64。各457參數/6690目標（6.831%），400updates；不讀official test。記錄初始feature/weight/logits尺度、loss、兩群梯度、每epoch更新及best/final指標。

沒有改LR、資料、bit編碼、initial state、CNOT順序或mapping activation。公式改變梯度尺度，固定optimizer LR不代表有效步長不變。低準確率不能直接否定原論文完整60000train結果。

命令：uv run python scripts/paper_r3_pilot.py；--smoke為單seed2epochs。報告：--report results/<study> --output reports/paper-r3。
