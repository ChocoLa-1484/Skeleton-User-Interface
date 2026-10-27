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

| 項目 | 版本 |
|------|------|
| OS | Windows 10 / 11 |
| Python | 3.8 |
| CUDA | 11.8 |
| cuDNN | 對應 CUDA 11.8 |
| Anaconda | 任意版本 |

> GPU 為必要條件，CPU-only 模式未支援。

---

## 安裝步驟

> **注意**：所有指令請在 **Anaconda Prompt** 或已初始化 conda 的 PowerShell 中執行。
> PowerShell 需先執行 `conda init powershell` 並重新開啟視窗，才能使用 `conda activate`。

---

### Step 1 — 建立 Conda 環境

```bash
conda create -n Skeleton python=3.8 -y
conda activate Skeleton
```

---

### Step 2 — 安裝 PyTorch（CUDA 11.8）

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

---

### Step 3 — 安裝基本套件

在專案根目錄執行，包含 PyQt5 GUI 所需套件。CSV 匯出使用 Python 內建的 `csv` 模組，不需額外安裝套件。

```bash
pip install -r requirements.txt
```

---

### Step 4 — 驗證 PyQt5

PyQt5 已包含在 `requirements.txt`，完成 Step 3 後可確認安裝版本：

```bash
python -c "from PyQt5.QtCore import PYQT_VERSION_STR; print(PYQT_VERSION_STR)"
# 應輸出: 5.15.11
```

---

### Step 5 — 安裝 OpenMMLab 套件

```bash
pip install -U openmim
mim install "mmcv==2.0.0"
mim install "mmdet==3.1.0"
mim install mmengine
```

---

### Step 6 — 安裝 mmpose（從本專案原始碼）

Windows 上 `.mim` 資料夾有權限問題，**必須先刪除再安裝**：

```powershell
# 回到專案根目錄（若不在根目錄請先 cd 過去）
cd C:\Workspace\Skeleton-User-Interface

# 刪除 .mim 資料夾（若不存在會顯示錯誤可忽略）
Remove-Item -Recurse -Force Src\mmpose_main\mmpose\.mim

# 安裝
cd Src\mmpose_main
python setup.py develop --no-deps
cd ..\..
```

---

### Step 7 — 安裝 mmpretrain（從本專案原始碼）

ViTPose 的 Vision Transformer backbone 需要此套件，同樣需先刪除 `.mim`：

```powershell
# 回到專案根目錄
cd C:\Workspace\Skeleton-User-Interface

# 刪除 .mim 資料夾（若不存在會顯示錯誤可忽略）
Remove-Item -Recurse -Force Src\mmpretrain_main\mmpretrain\.mim

# 安裝
cd Src\mmpretrain_main
python setup.py develop --no-deps
cd ..\..
```

---

### Step 8 — 安裝 faiss 與 cython_bbox

```bash
conda install -c conda-forge faiss-gpu -y
pip install cython_bbox
```

---

### Step 9 — 驗證安裝

執行以下指令，確認所有關鍵套件正常：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 應輸出: 2.0.1+cu118 True

python -c "import mmpose; print(mmpose.__version__)"
# 應輸出: 1.3.1

python -c "import mmpretrain; print(mmpretrain.__version__)"
# 應輸出: 1.2.0

python -c "from ultralytics import YOLO; print('YOLO OK')"
# 應輸出: YOLO OK
```

> **libiomp5md.dll 衝突**：若出現 OpenMP 相關警告，請刪除：
> `C:\Users\<你的帳號>\anaconda3\envs\Skeleton\Library\bin\libiomp5md.dll`

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

```bash
conda activate Skeleton
cd Src\UI_Control
python main.py
```

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
│                                     │  │ [載入原始影片]    │    │
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
| **處理和儲存** | 從第 0 幀開始處理，抵達最後一幀後補做尚未處理的影格，再將原畫面影片、人物框＋骨架影片（`_Sk26.mp4`）、JSON 與 CSV 儲存至 `Db/Record/{影片名稱}/`。 |
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

**Q：程式啟動時出現 `ModuleNotFoundError: No module named 'mmpose'`**

A：mmpose 需從本專案原始碼安裝。執行：
```bash
conda activate Skeleton
cd Src\mmpose_main
python setup.py develop --no-deps
```
若出現 `PermissionError [WinError 5]`，先刪除 `Src\mmpose_main\mmpose\.mim` 資料夾。

---

**Q：程式啟動時出現 `ModuleNotFoundError: No module named 'ultralytics'`**

A：執行：
```bash
conda activate Skeleton
pip install ultralytics==8.3.96
```

---

**Q：出現 `EasyPySpin` 相關錯誤**

A：`EasyPySpin` 是 FLIR 工業相機的 SDK，非必要套件，已透過 `try/except` 處理。若仍出錯，代表程式碼有未預期的路徑直接使用它，請回報 issue。

---

**Q：出現 `libiomp5md.dll` 衝突警告**

A：刪除 `C:\Users\<帳號>\anaconda3\envs\Skeleton\Library\bin\libiomp5md.dll` 即可解決。

---

**Q：影片偵測速度很慢**

A：確認 `torch.cuda.is_available()` 回傳 `True`。若為 `False`，代表 PyTorch 使用 CPU 模式，需重新安裝 CUDA 版本的 PyTorch（見 Step 2）。

---

**Q：到另一台電腦安裝時，mmpose setup.py 失敗**

A：常見原因是 `.mim` 資料夾的 Windows 權限問題。步驟：
1. 手動刪除 `Src\mmpose_main\mmpose\.mim`
2. 再次執行 `python setup.py develop --no-deps`
