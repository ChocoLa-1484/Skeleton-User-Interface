# Skeleton-User-Interface

人體骨架分析系統，整合 YOLOv11 物件偵測、ViTPose 骨架姿態估計（26 關節點）與 BoTSORT 多目標追蹤，以 PyQt5 GUI 呈現，可對運動影片進行骨架偵測與關節點追蹤。

---

## 目錄

1. [環境需求](#環境需求)
2. [安裝步驟](#安裝步驟)
3. [模型準備](#模型準備)
4. [啟動程式](#啟動程式)
5. [UI 操作說明](#ui-操作說明)
6. [輸出格式](#輸出格式)
7. [攝影機支援](#攝影機支援)
8. [專案結構](#專案結構)
9. [常見問題](#常見問題)

---

## 環境需求

以下流程針對 **Windows 10／11 x64、Python 3.8、NVIDIA GPU**。請依序執行，不要直接套用最新版 PyTorch 或 OpenMMLab 套件。

| 項目 | 固定組合／需求 |
|------|------|
| Python | 3.8.20（64 位元，Conda 環境） |
| PyTorch / torchvision / torchaudio | 2.0.1+cu118 / 0.15.2+cu118 / 2.0.2+cu118 |
| MMCV / MMEngine / MMDetection | 2.0.0 / 0.10.7 / 3.1.0 |
| MMPose / MMPretrain | 本專案原始碼，1.3.1 / 1.2.0 |
| NumPy / PyQt5 | 1.24.4 / 5.15.11 |
| NVIDIA GPU 與驅動 | 須支援上述 CUDA 11.8 套件；安裝後必須確認 `torch.cuda.is_available()` 為 `True` |
| C++ 編譯工具 | Visual Studio Build Tools 的「使用 C++ 的桌面開發」，含 MSVC x64/x86 工具與 Windows SDK，供 `cython_bbox` 等原始碼套件編譯 |

本流程使用預編譯的 PyTorch／MMCV CUDA wheel，不需另外手動複製 cuDNN 或編譯 MMCV。程式目前不支援 CPU-only 推論。`faiss-cpu` 是追蹤模組的相依套件，不會把 YOLO／ViTPose 推論改成 CPU。

下載來源：[PyTorch 官方歷史版本](https://docs.pytorch.org/get-started/previous-versions/)、[MMCV 官方 CUDA 11.8／Torch 2.0 wheel 清單](https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html)、[Faiss CPU wheel](https://pypi.org/project/faiss-cpu/1.8.0/)。

---

## 安裝步驟

以下指令使用 **PowerShell**。先安裝 Conda 與上述 C++ 編譯工具；若尚未初始化 PowerShell，執行 `conda init powershell` 後重新開啟視窗。

### Step 1 — 建立環境與安裝編譯前置套件

請將專案路徑換成實際位置。若已有名為 `Skeleton` 的環境且想重建，請使用不同環境名稱，避免混入舊套件。

```powershell
cd C:/Workspace/Skeleton-User-Interface
conda create -n Skeleton python=3.8.20 -y
conda activate Skeleton
python -m pip install pip==24.2 setuptools==60.2.0 wheel==0.44.0
python -m pip install numpy==1.24.4 Cython==0.29.37
```

先安裝 NumPy／Cython，是因為部分套件在建置階段就需要它們。固定 pip／setuptools，則是為了配合本專案的舊版 `setup.py` 與 editable 安裝流程。

### Step 2 — 安裝執行主程式所需的相依套件

在專案根目錄執行：

```powershell
python -m pip install --no-build-isolation -r requirements.txt
```

`requirements.txt` 已包含 PyTorch CUDA、MMCV、MMEngine、MMDetection、PyQt5、Faiss CPU 與骨架／追蹤相依套件，不需再使用 `mim install` 或 `conda install faiss-gpu`。初次安裝會下載大型 PyTorch wheel，請等待指令成功完成後再進行下一步。

- MMCV 固定使用官方 `cu118/torch2.0.0` wheel，並禁止退回原始碼編譯；不要同時安裝 `mmcv-lite` 或舊版 `mmcv-full`。
- `lapx` 提供程式使用的 `lap` 模組，不需另外安裝 `lap`。
- CSV 使用 Python 內建模組。ONNX 匯出工具、VisPy、FLIR SDK 不屬於目前 `main.py` 的必要安裝項目。
- 不要額外安裝 `opencv-python-headless` 或其他 OpenCV 發行套件來覆蓋 GUI 使用的 `opencv-python`。

### Step 3 — 安裝本專案的 MMPose 與 MMPretrain

仍在專案根目錄執行：

```powershell
python -m pip install --no-deps --no-build-isolation -e ./Src/mmpose_main -e ./Src/mmpretrain_main
```

必須使用專案內的原始碼，其中包含 HALPE／ViTPose 設定。`--no-deps` 是因為相依套件已由 Step 2 固定安裝；不要省略前一步。全新 checkout 通常不需手動刪除 `.mim`；若重複安裝遇到該目錄的權限錯誤，請參考常見問題。

### Step 4 — 驗證套件與 CUDA 算子

```powershell
python -m pip check
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA unavailable'"
python -c "import mmcv, mmengine, mmdet, mmpose, mmpretrain; print(mmcv.__version__, mmengine.__version__, mmdet.__version__, mmpose.__version__, mmpretrain.__version__)"
python -c "import torch; from mmcv.ops import nms; boxes=torch.tensor([[0.,0.,10.,10.]],device='cuda'); scores=torch.ones(1,device='cuda'); print(nms(boxes,scores,0.5)); print('MMCV CUDA OK')"
python -c "import faiss, lap; from cython_bbox import bbox_overlaps; from PyQt5.QtCore import PYQT_VERSION_STR; print('Tracking imports OK; PyQt', PYQT_VERSION_STR)"
```

預期版本為 `2.0.1+cu118 True`、`2.0.0 0.10.7 3.1.0 1.3.1 1.2.0`、PyQt `5.15.11`，且 `pip check` 顯示 `No broken requirements found`。若任何一步失敗，先修正環境再啟動 GUI。

驗證紀錄（2026-10-29）：在本機 Windows 的全新 Python 3.8.20 venv 中，不繼承既有環境套件，依上述 pip 安裝順序完成建置。已通過相依性檢查、MMCV CUDA NMS、專案根目錄啟動的模型與主視窗初始化、Qt 事件迴圈，以及 16 項回歸測試；另以實際模型完成兩支短片的資料夾批次分析。主視窗驗證採用下方的無視窗模式。其他電腦仍需備妥相容 GPU／驅動、C++ 編譯工具與模型權重。

---

## 模型準備

模型權重檔因體積過大（合計約 1.2 GB）無法上傳至 GitHub，請從以下 Google Drive 下載：

**[點此下載模型檔案](https://drive.google.com/drive/folders/1FJDGG1D4qhj-ivjHMgmz0nLkM_zeKDjX?usp=drive_link)**

| 檔案名稱 | 用途 | 大小（約）|
|---------|------|---------|
| `epoch_210.pth` | ViTPose 姿態估計模型（HALPE 26 關節點） | ~1.1 GB |
| `yolo11m-seg.pt` | YOLOv11 人體偵測模型 | ~45 MB |

下載後，將兩個檔案放至專案的 `Db/pretrain/` 資料夾（若不存在請手動建立）：

```
Skeleton-User-Interface/
├── Db/
│   ├── pretrain/
│   │   ├── epoch_210.pth        ← 下載後放這裡
│   │   └── yolo11m-seg.pt       ← 下載後放這裡
│   └── Record/                  ← 輸出資料夾（自動建立）
└── Src/
```

---

## 啟動程式

先完成上方模型準備，再使用同一個 Conda 環境啟動：

```powershell
conda activate Skeleton
cd C:/Workspace/Skeleton-User-Interface
python Src/UI_Control/main.py
```

模型、設定檔、字型與輸出位置依程式檔案位置解析，不需切換到 `Src/UI_Control`。從其他工作目錄也可用 `main.py` 的絕對路徑啟動；輸出固定放在此專案的 `Db/Record`。在 IDE 中選用已安裝相依套件的 Python 解譯器即可，程式不再依賴 `CONDA_PREFIX` 設定 Qt 路徑。

如需自動檢查主視窗、模型權重與 Qt 事件迴圈是否能初始化，可在相同目錄執行以下指令；它不顯示視窗，約一秒後結束事件迴圈（模型載入另需時間）：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -c "import sys; sys.path.insert(0,'Src/UI_Control'); from PyQt5.QtWidgets import QApplication; from PyQt5.QtCore import QTimer; from main import Main; app=QApplication([]); window=Main(); QTimer.singleShot(1000, app.quit); app.exec_(); print('main.py startup OK')"
Remove-Item Env:QT_QPA_PLATFORM
```

回歸測試可從專案根目錄執行 `python -m unittest discover -s tests -v`。測試使用臨時資料，不覆寫 `Db/Record`。

---

## UI 操作說明

### 介面總覽

```
┌─────────────────────────────────────┬──────────────────────────┐
│                                     │  2D 關節點               │
│                                     │  ┌──────────────────┐    │
│                                     │  │  檔案            │    │
│         影片顯示區                   │  │  檔名: xxx       │    │
│         (FrameView)                 │  │  FPS: 280        │    │
│                                     │  │ [載入原始影片] [載入資料夾] │
│                                     │  │ [處理和儲存]      │    │
│                                     │  │ [載入處理過的影片] │    │
│                                     │  └──────────────────┘    │
│                                     │                          │
│                                     │  顯示資訊                │
│                                     │  ☑ 人體骨架  □ 人物框    │
│                                     │  □ 選擇人                │
├─────────────────────────────────────┤                          │
│  [<<]  [▶]  [>>]   ────────●────   │                          │
│                      394 / 580      │                          │
└─────────────────────────────────────┴──────────────────────────┘
```

---

### 按鈕與功能說明

#### 檔案區

| 按鈕 | 功能 |
|------|------|
| **載入原始影片** | 開啟檔案選擇對話框，載入未處理的 MP4 影片。影片載入後會自動逐幀解析並顯示第 0 幀。 |
| **載入資料夾** | 選擇資料夾後，依檔名順序自動分析第一層所有支援的影片，顯示進度並支援取消。 |
| **處理和儲存** | 同時勾選「人體骨架」與「人物框」，從第 0 幀開始處理；抵達最後一幀後補做尚未處理的影格，再將原畫面影片、人物框＋骨架影片（`_Sk26.mp4`）、JSON 與 CSV 儲存至 `Db/Record/{影片名稱}/`。 |
| **載入處理過的影片** | 載入影片及對應的骨架 JSON，直接顯示人物框與骨架，不重新執行模型；支援原畫面影片與 `_Sk26.mp4`。 |

#### 播放控制區

| 按鈕 | 功能 |
|------|------|
| **`<<`** | 退回上一幀 |
| **`▶` / `\|\|`** | 播放 / 暫停 |
| **`>>`** | 前進下一幀 |
| **進度條（Slider）** | 拖曳跳至任意幀，每移動一格就會觸發該幀的骨架偵測 |

> **鍵盤快捷鍵**：按住 `A` 退後一幀，按 `D` 前進一幀（需視窗取得焦點）。

#### 顯示資訊區（Checkbox）

| Checkbox | 功能 |
|----------|------|
| **人體骨架** | 顯示／隱藏預覽骨架，同時啟用／停用預覽時的骨架偵測；不影響儲存前補做未處理影格 |
| **人物框** | 立即顯示／隱藏預覽中的人物框與 ID，暫停時也會更新；需已有該影格的偵測資料 |
| **選擇人** | 勾選後顯示可選人物框，左鍵點選人物可指定追蹤 ID；原始影片只對該人物執行骨架估計，取消勾選則恢復全部人物。已處理影片則篩選已儲存的人物資料。 |

輸出的 `_Sk26.mp4` 固定啟用人物框與骨架，不受預覽顯示開關影響。未選人時對所有有效追蹤人物估計骨架；指定人物後，儲存前會以目前選定的追蹤 ID 補齊全片的骨架估計，影片、JSON 與 CSV 都只保留該人物的結果。

---

### 操作流程：對影片進行骨架偵測

以下為完整的操作步驟（以跑步影片為例）：

**① 載入影片**

點擊「**載入原始影片**」→ 在對話框中選擇 MP4 影片 → 等待讀取完成（上方會顯示檔案名稱與解析度）。

**② 開始偵測**

點擊「**處理和儲存**」→ 程式會自動從第 0 幀開始播放並逐幀進行骨架偵測。

「人體骨架」與「人物框」會一併勾選，因此預覽會同時顯示兩者。

> 偵測速度取決於 GPU 效能、人物數與影片解析度。FPS 顯示骨架處理計時結果，已處理影片顯示「—」；不代表原影片幀率或完整輸出速度。

**③ 選取追蹤目標（選擇人）**

1. 載入原始影片，勾選「**選擇人**」，程式會啟用骨架顯示並顯示可選人物框
2. 用滑鼠左鍵點擊想分析的人物框，該影格會立即改為只估計此人物的骨架
3. 點擊「**處理和儲存**」，從頭處理並輸出此人物的結果；可點選其他人物切換目標
4. 取消「**選擇人**」可恢復對所有有效追蹤人物進行骨架估計

> 人物 ID 使用 BoTSORT 追蹤結果，拖曳進度條時會依時間順序補做前面的偵測與追蹤，以維持 ID 對應。遮擋或離開畫面仍可能導致追蹤中斷；選定 ID 未出現時，不會改以其他人物或整張畫面估計骨架。

**④ 儲存結果**

- 點擊「**處理和儲存**」後抵達最後一幀會自動儲存；一般預覽或選人不會自行觸發儲存
- 儲存前會補做尚未處理的影格，這個階段可能需要等待
- 輸出檔案儲存至 `Db/Record/{影片名稱}/` 資料夾，同名檔案會覆寫

**⑤ 檢視輸出與重新載入**

可直接使用影片播放器開啟 `{影片名稱}_Sk26.mp4` 檢視人物框與骨架，或使用 Excel 開啟 CSV 分析座標。

使用「**載入處理過的影片**」選擇輸出資料夾中的 `{影片名稱}.mp4` 或 `{影片名稱}_Sk26.mp4`，兩者都會對應到同資料夾的 `{影片名稱}.json`，不再需要舊版球棒／揮棒 JSON。建議選擇未繪製標記的 `{影片名稱}.mp4`，以便自由切換人物框與骨架顯示；`_Sk26.mp4` 本身已包含的標記無法移除。

載入後會直接顯示儲存結果，缺少人物的影格也不會重新偵測。若 JSON 不存在或缺少必要欄位，程式會顯示錯誤訊息。此模式的「選擇人」只篩選 JSON 中已有的人物；若要重新估計或分析未儲存的人物，請以「**載入原始影片**」開啟未繪製標記的影片。CSV 目前僅供匯出分析，不作為重新載入來源。

---

### 資料夾批次分析

點擊「載入原始影片」旁的「**載入資料夾**」，選擇資料夾後就會開始逐支分析，不需另外按「處理和儲存」。

- 分析資料夾第一層的 `.mp4`、`.avi`、`.mov`、`.mkv`、`.webm`、`.m4v`、`.mpg`、`.mpeg`、`.wmv`，副檔名不區分大小寫；不包含子資料夾。
- 每支影片使用獨立的追蹤與骨架資料，對所有有效追蹤人物做 ViTPose 分析，不沿用單片預覽的選人設定。
- 每支影片輸出至 `Db/Record/{影片名稱}/`，包含原畫面 MP4、人物框＋骨架 MP4、JSON 與 CSV。輸入檔若同名但副檔名不同，會在輸出名稱加上副檔名及必要的流水號，避免互相覆寫。
- 批次期間會顯示影片與影格進度；預覽操作暫時停用，按「取消」可停止剩餘處理。已完成的影片會保留，當前尚未完成的暫存結果會清除。
- 單支影片失敗後會繼續下一支，結束時顯示成功／失敗數及錯誤詳細資料。影片完整完成後才更新同名輸出檔。

批次逐幀讀取與寫入影片，保留來源 FPS，不會一次把整個資料夾的影像載入記憶體。完成後會恢復原先單片預覽的追蹤狀態。

### 骨架關節點對照（HALPE 26 點）

| ID | 部位 | ID | 部位 |
|----|------|----|------|
| 0 | 鼻子 | 13 | 左膝 |
| 1 | 左眼 | 14 | 右膝 |
| 2 | 右眼 | 15 | 左踝 |
| 3 | 左耳 | 16 | 右踝 |
| 4 | 右耳 | 17 | 頭部 |
| 5 | 左肩 | 18 | 頸部 |
| 6 | 右肩 | 19 | 臀部 |
| 7 | 左肘 | 20 | 左大腳趾 |
| 8 | 右肘 | 21 | 右大腳趾 |
| 9 | 左腕 | 22 | 左小腳趾 |
| 10 | 右腕 | 23 | 右小腳趾 |
| 11 | 左髖 | 24 | 左腳跟 |
| 12 | 右髖 | 25 | 右腳跟 |

---

## 輸出格式

分析完成後，結果儲存至 `Db/Record/{影片名稱}/`：

| 檔案 | 說明 |
|------|------|
| `{影片名稱}.mp4` | 由載入的影格重新編碼的原畫面影片，未繪製骨架與人物框 |
| `{影片名稱}_Sk26.mp4` | 逐幀繪製人物框與骨架；指定人物時只輸出該人物，不受預覽顯示開關影響 |
| `{影片名稱}.json` | 逐幀骨架資料（含 Bounding Box + 26 關節點座標） |
| `{影片名稱}.csv` | 每列一個影格／人物／關節點，含人物框、座標、信心值與人工修改標記 |

CSV 與 JSON 同時儲存。CSV 採用 UTF-8 BOM 編碼，可直接以 Excel 開啟；欄位為 `frame_number`（從 0 開始）、`person_id`、`bbox_x1`、`bbox_y1`、`bbox_x2`、`bbox_y2`、`keypoint_id`（0–25）、`keypoint_name`、`x`、`y`、`confidence`、`is_modified`（0：未修改；1：人工修改）。座標以原始影片像素為單位；沒有偵測資料時僅輸出標題列。

兩份 MP4 都由 OpenCV 逐幀寫入，不保留原影片音軌。原畫面影片也不是來源檔案的逐位元複製。

「處理和儲存」會從第 0 幀開始處理，儲存前會補做尚未處理的影格。輸出的骨架影片固定啟用人物框與骨架；模型未偵測到人物的影格仍保留原畫面，不補造座標。

JSON 使用 records 陣列，每筆為一個影格中的一個人物。以下範例僅節錄前兩個關節點，完整資料為 26 點；每點依序為 `[x, y, confidence, is_modified]`：
```json
[
  {
    "person_id": 1,
    "bbox": [100.0, 50.0, 300.0, 450.0],
    "keypoints": [
      [200.0, 80.0, 0.98, 0.0],
      [195.0, 75.0, 0.95, 1.0]
    ],
    "frame_number": 0
  }
]
```

---

## 攝影機支援

| 攝影機類型 | 狀態 | 說明 |
|-----------|------|------|
| 影片檔（MP4 等）| ✅ 支援 | 主要使用方式 |
| FLIR / Point Grey 工業相機 | ⚠️ 僅有底層擷取程式 | 需安裝 Spinnaker SDK + `EasyPySpin`；目前主視窗未啟用相機 Tab |
| Sony AX700 / 一般相機（HDMI 擷取卡）| 🔧 規劃中 | 透過 `cv2.VideoCapture()` 讀取，需新增 `SonyCaptureThread` |

---

## 專案結構

```
Skeleton-User-Interface/
├── Db/
│   ├── pretrain/              # 模型權重檔
│   └── Record/                # 分析輸出（自動建立）
├── Src/
│   ├── UI_Control/            # 主程式
│   │   ├── main.py            # 入口點
│   │   ├── video_widget.py    # 影片分析 Tab 邏輯
│   │   ├── ui/                # PyQt5 UI 定義（.py + .ui）
│   │   ├── skeleton/          # 骨架偵測核心（ViTPose + YOLO + tracker）
│   │   ├── cv_utils/          # 影像控制與執行緒
│   │   ├── utils/             # 分析、濾波、視覺化工具
│   │   └── camera_config/     # 攝影機設定
│   ├── mmengine_main/        # OpenMMLab mmengine 原始碼
│   ├── mmpretrain_main/      # ViTPose backbone（editable 安裝）
│   ├── mmpose_main/           # OpenMMLab mmpose（本地版本，editable 安裝）
│   ├── mmyolo_main/           # OpenMMLab mmyolo
│   ├── tracker/               # BoTSORT 多目標追蹤器
│   └── cython_bbox-0.1.3/     # Bounding Box 計算工具
├── requirements.txt           # Python 套件依賴清單
├── CLAUDE.md                  # 開發環境紀錄（給 Claude Code 使用）
└── README.md                  # 本文件
```

---

## 常見問題

**Q：找不到 `mmpose`、`mmpretrain` 或 `VisionTransformer`**

A：確認使用正確的 Python 環境，並在專案根目錄重跑 Step 3 的兩個本地套件安裝。`utils/model.py` 會匯入 `mmpretrain` 來註冊 ViTPose backbone；僅安裝 PyPI 的 MMPose 不足以取代本專案原始碼。

**Q：找不到 `ultralytics`、`faiss`、`lap` 或 `cython_bbox`**

A：在啟用的 `Skeleton` 環境中，從專案根目錄重新執行 Step 2，並確認安裝成功。`faiss` 對應的套件名稱為 `faiss-cpu`，`lap` 對應 `lapx`。不要在另外一個 Python 環境單獨補裝。

**Q：`cython_bbox`／COCO 套件編譯失敗，或要求 Microsoft Visual C++**

A：安裝 Visual Studio Build Tools 的「使用 C++ 的桌面開發」（含 MSVC 與 Windows SDK），重開終端機，再依序執行 Step 1 的前置套件安裝與 Step 2。若仍找不到編譯器，可從 Visual Studio 的 x64 Native Tools 終端機啟用 Conda 後安裝。請保留 `--no-build-isolation`，讓建置使用環境中已固定的 NumPy／Cython。

**Q：找不到相符的 MMCV wheel，或出現 `No module named mmcv._ext`**

A：確認 Python 是 Windows x64 的 3.8、PyTorch 是 `2.0.1+cu118`，以及能連線到 requirements 中的官方下載來源。不要把 `mmcv-lite` 當成替代品。若環境已混裝多種 MMCV，建議建立新環境後依文件安裝，再執行 Step 4 的 CUDA NMS 檢查。

**Q：本地套件安裝時，`.mim` 出現 `PermissionError [WinError 5]`**

A：先關閉使用此環境的程式，確認目前位於專案根目錄。只清理對應套件產生的 `.mim` 目錄後重跑 Step 3；不要刪除來源 `configs`、`tools` 或整個套件目錄。全新安裝不需預先清理。

**Q：找不到模型權重**

A：確認兩份模型權重位於此專案的 `Db/pretrain`，檔名與模型準備章節一致。從專案根目錄執行 `python Src/UI_Control/main.py`，或使用程式的絕對路徑；不需手動設定 `CONDA_PREFIX`。

**Q：出現 Qt platform plugin／`qwindows.dll` 錯誤**

A：確認使用 `Skeleton` 的 Python，並完成 PyQt5 安裝。移除終端機中手動指定、指向其他 Python／Qt 安裝的 `QT_PLUGIN_PATH`，不要混用其他環境的 Qt DLL。自動啟動檢查結束後，需清除 `QT_QPA_PLATFORM=offscreen` 才會顯示正常視窗。

**Q：出現 `libiomp5md.dll` 衝突警告**

A：通常是同一個程序載入多份 OpenMP runtime。請在新環境使用本文件的 pip 套件組合，避免混裝 Conda 的 Faiss GPU／數值套件與 pip 版本；也要檢查 PATH 是否指向其他環境。不要直接刪除不明 DLL，或以 `KMP_DUPLICATE_LIB_OK` 略過衝突。

**Q：出現 `EasyPySpin` 相關錯誤**

A：目前影片分析不需要 FLIR 相機 SDK，未安裝 `EasyPySpin` 時程式會略過其匯入。若另行開發 FLIR 擷取功能，才需安裝匹配的 Spinnaker SDK 與 Python 套件。

**Q：CUDA 不可用或影片偵測很慢**

A：先執行 `nvidia-smi` 與 Step 4 的 CUDA 檢查。若 `torch.cuda.is_available()` 為 `False`，確認 GPU 驅動、硬體相容性，以及是否誤裝 CPU 版 PyTorch。安裝 CUDA Toolkit 本身不會把 CPU 版 PyTorch 變成 CUDA 版。
