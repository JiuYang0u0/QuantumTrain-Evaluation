# Update-budget完整續訓

CPU update-budget runner在每個驗證節點（正式100updates）原子保存latest.pt：先寫latest.pt.tmp，再replace。保存模型、Adam狀態、獨立抽樣器、Torch/Python/NumPy RNG、step、樣本seen mask、history、最佳loss與最佳模型副本、已完成train_seconds。

恢復以latest.pt為一致性來源，重新寫回history/best，避免中斷發生在多檔更新之間留下較新的孤立best。最多重算上次保存後的99個訓練updates；checkpoint可能在驗證中斷時落後100updates。不是每步落盤，原子replace也不是斷電耐久性保證。

續訓前驗證資料內容SHA256、模型結構與關鍵屬性、Torch版本、threads、seed、更新預算、batch、驗證間隔與milestones；不相容則拒絕。CPU同環境測試（含Dropout）確認中斷後模型、Adam moments、抽樣狀態、history、最終/最佳評估與不中斷相同；wall-clock時間不預期相同。不承諾跨硬體/版本bitwise一致。

`uv run python scripts/paper_data_budget.py --extended --recover-from results/<study>` 建立新study保留原始紀錄：完成組沿用；有latest.pt的未完成組從保存步續訓；舊版本沒有latest.pt的組別由初始化重跑。recovery.json分別記錄三種來源。

舊QT33在10200updates中斷，只有best/milestone模型，沒有Adam與抽樣狀態，不能轉成精確resume。這次從seed33重跑一次後即有新checkpoint保護。
