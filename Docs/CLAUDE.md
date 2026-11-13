# CLAUDE.md — 專案重要資訊

## 專案說明

人體骨架分析系統，結合 YOLO 物件偵測、ViTPose 骨架姿態估計（26 關節點 HALPE 格式）、BoTSORT 多目標追蹤，以 PyQt5 GUI 呈現。適用於各類運動員動作分析。

## 執行環境

| 項目 | 內容 |
|------|------|
| Conda 環境 | `Skeleton` |
| Python | 3.8.20 |
| PyTorch | 2.0.1+cu118 |
| CUDA | 11.8 |

**啟動指令：**
```bash
conda activate Skeleton
cd Src/UI_Control
python main.py
```

## 關鍵套件版本（Skeleton 環境）

| 套件 | 版本 |
|------|------|
| torch | 2.0.1+cu118 |
| mmpose | 1.3.1（editable，來自本專案 `Src/mmpose_main`） |
| mmcv | 2.0.0 |
| mmengine | 0.10.7 |
| mmdet | 3.1.0 |
| mmpretrain | 1.2.0（editable，來自本專案 `Src/mmpretrain_main`） |
| ultralytics | 8.4.202（整合 YOLO11／YOLO26） |
| PyQt5 | 5.15.11 |
| opencv-python | 4.8.1.78 |

## 模型檔案位置

```
Db/pretrain/
├── epoch_210.pth          # ViTPose 姿態估計模型（HALPE 26 關節點）
├── yolo11m-seg.pt         # YOLO11 人體偵測模型（預設）
├── yolo11m.pt
├── yolo26m.pt
└── yolo26x.pt
```

## 程式入口與架構

```
Src/UI_Control/
├── main.py                  # 主程式入口
├── video_widget.py          # 2D 影片分析 Tab（目前唯一啟用的 tab）
├── ui/
│   ├── main_window.py       # PyQt5 主視窗 UI
│   └── video_ui.py          # 影片 Tab UI（已移除角度/關節點選擇元素）
├── utils/
│   ├── model.py             # 模型載入（YOLO + ViTPose + BoTSORT）
│   ├── analyze.py           # 關節角度分析（PoseAnalyzer）
│   ├── vis_image.py         # 骨架視覺化繪製
│   ├── vis_graph.py         # 角度圖表（GraphPlotter）
│   └── selector.py          # PersonSelector（人員選擇）
├── skeleton/detect_skeleton.py  # 骨架偵測核心
├── cv_utils/cv_control.py   # 影像控制（VideoLoader）
└── cv_utils/cv_thread.py    # QThread 執行緒（影片/攝影機/錄影）
```

## mmpretrain 安裝注意事項

mmpretrain 需從本專案 `Src/mmpretrain_main` 以 editable 方式安裝（ViTPose backbone 需要）：
```powershell
Remove-Item -Recurse -Force Src\mmpretrain_main\mmpretrain\.mim
cd Src\mmpretrain_main
python setup.py develop --no-deps
cd ..\..
```

`utils/model.py` 中已加入 `import mmpretrain` 確保 VisionTransformer 被註冊到 mmengine registry。

## 已修改的 Bug

### 1. `cv_utils/cv_thread.py` — EasyPySpin 全域 import
- **問題**：`import EasyPySpin` 放在模組最上層，導致未安裝 FLIR Spinnaker SDK 的機器無法啟動
- **修復**：改為 `try/except ImportError`，只在有安裝時才啟用

### 2. `cv_utils/cv_control.py` — 同上

### 3. 已移除的功能（2025-05）
- `BatAnalyzer`（球棒分析）— 整個 class 已從 `analyze.py` 移除
- `KptSelector`（關節點選擇）— 從 `selector.py` 移除
- `AngleGraphPlotter`、`SpeedGraphPlotter` — 從 `vis_graph.py` 移除
- UI 中的「選擇關節點」、「顯示角度」checkbox — 從 `video_ui.py` 移除
- `vis_image.py` 中所有球棒繪製邏輯（`draw_bat`、`drawBatTraj`、`drawAngleInfo`、`drawTraj`）
- 棒球打者相關廢棄檔案：`hitter_widget_dual.py`、`hitter_video_widget_dual.py`、`hitter_video_widget_dual_init.py`、`utils/analyze_dual.py`

### 4. Bug 修復（2025-05）
- `skeleton/detect_skeleton.py` `getPersonDf()`：資料為空時改回傳 `pd.DataFrame()` 而非 `None`，避免呼叫端 `.empty` AttributeError
- `skeleton/detect_skeleton.py` `getBatDF()`：將 `is None` 改為 `.empty` 檢查（bat_df 初始值為空 DataFrame，永不為 None）
- `cv_utils/cv_thread.py` `VideoToImagesThread.run()`：`_run_flag=False` 修正為 `self._run_flag = False`（原為建立局部變數，無法影響執行個體狀態）
- `cv_utils/cv_control.py` `saveVideo()`：移除已刪除的 `saveBatAnalyzeData()` 呼叫，只保留骨架 JSON 儲存
- `utils/vis_image.py` `drawBbox()`：加入 `person_df is None` 檢查，避免沒選人時點「人物框」crash

## 攝影機支援說明

| 攝影機 | 套件 | 狀態 |
|--------|------|------|
| FLIR / Point Grey 工業相機 | `EasyPySpin` + Spinnaker SDK | 已有 `VideoCaptureThread`，需安裝硬體 SDK |
| Sony AX700 / 一般相機 | `cv2.VideoCapture()` | 尚未實作，需新增 `SonyCaptureThread` |

## 目前啟用的 Tab

`main.py` 中只有 `PoseVideoTabControl`（2D 影片）是啟用的，其餘 tab 皆被 comment 掉：
- `PoseCameraTabControl` — 相機（已 comment）
- `PoseHitTabControlDual` — 雙相機打擊分析（已 comment）
- `PoseHitVideoTabControlDual` — 雙影片打擊分析（已 comment）

## mmpose 安裝注意事項

mmpose 需從本專案的 `Src/mmpose_main` 以 editable 方式安裝：
```powershell
Remove-Item -Recurse -Force Src\mmpose_main\mmpose\.mim
cd Src\mmpose_main
python setup.py develop --no-deps
cd ..\..
```
