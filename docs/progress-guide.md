# 唯讀進度查詢與恢復

```powershell
uv run python -m qt_eval.progress --list
uv run python -m qt_eval.progress results/<study目錄>
uv run python -m qt_eval.progress results/<study目錄> --json
uv run python -m qt_eval.progress results/<study目錄> --stale-seconds 1800
```

一次性快照，沒有輪詢、不修改狀態、不自動重啟、不反序列化checkpoint。`--list`僅列study目錄，不宣稱正在執行。

## 如何判讀

| 欄位/狀態 | 含義 |
|---|---|
| completed_reported | result.json宣告完成；不是完整研究品質驗證 |
| failed_reported | 結果宣告失敗或該run有error.txt |
| not_started | 預期run尚無進度產物 |
| process_present | 命令的--resume-root精確指向study，程序存在；不是heartbeat |
| unconfirmed | 有同專案同腳本候選程序，或可見性不足，無法確定歸属 |
| interrupted_or_unowned | 可檢查的本機Python/uv程序未找到候選，已有未完成保存紀錄 |
| stale | 最後產物寫入距今超過閾值；長step、驗證、I/O也會造成，不等於已死 |

`process_evidence`：confirmed/candidate/not_found/unknown。只認`--resume-root`為明確目標；`--recover-from`是舊來源，不能當新study身份。一般首次啟動自動產生study名，命令列不帶該名，因此只標candidate，即使寫入很新也不強稱confirmed。psutil遇權限不足時保守記錄visibility_complete=false。

完成數不是耗時百分比；CNN和QT每步成本差很多。saved_step來自history或完成result，可能落後真實步數。latest.pt存在不代表可讀或相容；真正resume由訓練器核對。Best/final/latest驗證值分欄，不把最新accuracy當最好值。

## 支援范围與限制
QT106 `paper-depth-*`及`paper-data-budget-*`/`paper-data-extended-*`讀取宣告矩陣，因此未開始seed也會列出。其他舊研究僅發現現有run，matrix_support=discovered_only，分母不是完整預定矩陣；程序歸属標unknown。JSON讀取中斷或格式錯誤會顯示read_errors，不能當作完成。檔案與程序是不同時間點快照，存在競態；工具不提供保證正確的ETA。

## 恢復

QT106（原study）：
```powershell
uv run python scripts/paper_depth_study.py --resume-root results/<paper-depth目錄>
```

資料/預算矩陣（新recovery目錄、保留來源）：
```powershell
uv run python scripts/paper_data_budget.py --extended --recover-from results/<paper-data-extended目錄>
```

先確認沒有其他程序正在寫相同study，避免重複訓練。有latest.pt才可精確恢復其保存步；只有best.pt的舊run需從頭重跑。保存內容、相容性與CPU限制見[checkpoint-recovery](checkpoint-recovery.md)。工具不替你自動做重跑決策。
