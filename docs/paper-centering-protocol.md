# r2：只加入上游全域生成權重中心化

來源：固定commit d5b654116565ae9550e540e021ab0d4160a15c86的MNIST notebook，LewHybridNN.forward中mapping輸出後執行 `output - output.mean()`。這是所有6690weights與biases一起減平均，**不同於舊digits實驗只中心化W1/W2**。

初始化診斷paper-scale-5450d2a62d：相同MNIST train probe前128筆、相同theta/mapping權重，三seeds原始loss127050.36/2435.21/1545.84，全域中心化後2.42/2.61/2.34。Seed11的fc1 absmax226694.52降至6.32。這支持全域生成偏移對初始化的放大有因果作用，但不是長期學習或論文accuracy證據。

配對pilot：r1_raw、r2_global_center、direct CNN，各seeds11/22/33。其餘遵循paper-mnist-protocol.md：13qubits/16blocks、RY/CNOT、mapping249、首層Tanh、0/1bits、raw p、Adam lr1e-4、1000train/200validation、2epochs/batch128。QT仍457參數。Official test不下載不讀。

只改全域中心化；沒有同時加入機率縮放、H初态、不同activation或改LR。本版本混合主文與明確來源code處理，仍標示paper-guided reconstruction r2，不稱精確復現。

評估best-validation loss checkpoint的train/validation，驗證獨立CNN匯出輸出一致。短程16updates只能確認穩定性與進展，不期待重現81.08%或93.81%。若穩定再評估長程預算。
