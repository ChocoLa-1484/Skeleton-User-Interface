"""Startup model selection; importing this dialog does not load ML frameworks."""
from pathlib import Path
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QComboBox, QLabel,
                             QLineEdit, QPushButton, QHBoxLayout, QDialogButtonBox,
                             QFileDialog, QMessageBox)
from utils.paths import PRETRAIN_ROOT

PRESETS = [
    ('YOLO11m-seg + ViTPose（原本模型，26 點）', 'yolo11m-seg.pt'),
    ('YOLO11m + ViTPose（26 點）', 'yolo11m.pt'),
    ('YOLO26m + ViTPose（26 點）', 'yolo26m.pt'),
    ('YOLO26x + ViTPose（26 點）', 'yolo26x.pt'),
]


class ModelSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('選擇姿態分析模型')
        self.resize(680, 260)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.preset = QComboBox()
        self.preset.addItems([item[0] for item in PRESETS])
        form.addRow('分析流程', self.preset)
        self.detector = QLineEdit()
        self.pose = QLineEdit(str(PRETRAIN_ROOT / 'epoch_210.pth'))
        self.pose_button = self.add_file_row(form, 'ViTPose 權重', self.pose, '模型權重 (*.pth)')
        self.add_file_row(form, 'YOLO 權重', self.detector, 'YOLO 權重 (*.pt)')
        layout.addLayout(form)
        self.note = QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText('載入模型')
        buttons.button(QDialogButtonBox.Cancel).setText('取消')
        buttons.accepted.connect(self.validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.preset.currentIndexChanged.connect(self.update_preset)
        self.update_preset(0)

    def add_file_row(self, form, label, field, file_filter):
        row = QHBoxLayout()
        row.addWidget(field)
        button = QPushButton('瀏覽…')
        def browse():
            filename, _ = QFileDialog.getOpenFileName(self, label, str(PRETRAIN_ROOT), file_filter)
            if filename:
                field.setText(filename)
        button.clicked.connect(browse)
        row.addWidget(button)
        form.addRow(label, row)
        return button

    def update_preset(self, index):
        _, filename = PRESETS[index]
        self.detector.setText(str(PRETRAIN_ROOT / filename))
        self.note.setText(
            '人體框 → ViTPose → HALPE 26 點。需要原專案的 MMPose／MMPretrain 環境。'
            '\nYOLO26 需要新版 Ultralytics；缺少官方 YOLO 權重時會自動下載。結果依模型分開儲存。')

    def validate(self):
        weights = Path(self.detector.text().strip())
        if weights.suffix.lower() != '.pt':
            QMessageBox.warning(self, '權重格式錯誤', '請指定 YOLO .pt 權重。')
            return
        if not weights.is_file() and weights.name not in {p[1] for p in PRESETS}:
            QMessageBox.warning(self, '找不到權重', '自訂權重必須是存在的檔案。')
            return
        if not Path(self.pose.text().strip()).is_file():
            QMessageBox.warning(self, '找不到 ViTPose 權重', '請選擇 epoch_210.pth 或相容的 HALPE26 權重。')
            return
        self.accept()

    def create_model(self):
        weights = self.detector.text().strip()
        if Path(weights).name.startswith('yolo26'):
            import ultralytics
            if not (Path(ultralytics.__file__).parent / 'cfg/models/26').is_dir():
                raise RuntimeError('目前 Ultralytics 不支援 YOLO26。請切換到已安裝新版的環境。\n'
                                   'YOLO26 + ViTPose 另外需要 MMPose、MMPretrain 及原專案依賴。')
        from utils.model import Model
        return Model(detector_weights=weights, pose_checkpoint=self.pose.text().strip(), argv=[])
