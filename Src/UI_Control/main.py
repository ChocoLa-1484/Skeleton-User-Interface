import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QMessageBox, QAction
from PyQt5.QtCore import Qt, QTimer
from video_widget import PoseVideoTabControl
from ui.main_window import Ui_MainWindow
from ui.model_selection import ModelSelectionDialog


class Main(QMainWindow):
    def __init__(self, model=None):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        if model is None:
            from utils.model import Model
            model = Model()
        self.model = model
        self.setWindowTitle('骨架分析 — ' + self.model.display_name)
        self.init_tabs()
        toolbar = self.addToolBar('模型')
        toolbar.setMovable(False)
        self.switch_action = QAction('切換模型…', self)
        self.switch_action.triggered.connect(self.switch_model)
        toolbar.addAction(self.switch_action)

    def switch_model(self):
        old_tab = self.video_tab
        if old_tab.batch_worker is not None or old_tab.video_loader.is_loading:
            QMessageBox.information(self, '目前正在處理', '請先等待影片讀取完成，或取消資料夾批次分析，再切換模型。')
            return
        old_tab.is_play = False
        old_tab.ui.playBtn.setText('▶︎')
        dialog = ModelSelectionDialog(self)
        dialog.note.setText(dialog.note.text() + '\n切換後清除目前分析與人物選取；已儲存的檔案保留。')
        while dialog.exec_() == QDialog.Accepted:
            candidate = None
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    model = dialog.create_model()
                    candidate = PoseVideoTabControl(model)
                    # Reuse decoded raw video, but never carry skeletons across models.
                    if old_tab.video_loader.video_frames and not old_tab.is_processed:
                        for field in ('video_path', 'folder_path', 'video_size', 'video_fps',
                                      'video_frames', 'total_frames', 'video_name'):
                            setattr(candidate.video_loader, field, getattr(old_tab.video_loader, field))
                        candidate.updateVideoInfo()
                finally:
                    QApplication.restoreOverrideCursor()
            except Exception as error:
                if candidate is not None:
                    candidate.deleteLater()
                QMessageBox.warning(self, '模型載入失敗', str(error))
                continue
            old_tab.is_processing = False
            self.ui.Two_d_Tab.removeTab(self.ui.Two_d_Tab.indexOf(old_tab))
            self.model = model
            self.video_tab = candidate
            self.ui.Two_d_Tab.addTab(candidate, '2D 影片')
            old_tab.deleteLater()
            self.setWindowTitle('骨架分析 — ' + model.display_name)
            self.statusBar().showMessage(model.display_name + ' ｜ 輸出：' + str(model.output_root))
            return

    def init_tabs(self):
        self.video_tab = PoseVideoTabControl(self.model)
        self.ui.Two_d_Tab.addTab(self.video_tab, "2D 影片")
        self.statusBar().showMessage(self.model.display_name + ' ｜ 輸出：' + str(self.model.output_root))

    def closeEvent(self, event):
        self.video_tab.is_play = False
        worker = self.video_tab.batch_worker
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            if not self.video_tab.closing_after_batch:
                self.video_tab.closing_after_batch = True
                worker.finished.connect(self.close)
            event.ignore()
            return
        if self.video_tab.video_loader.is_loading:
            QTimer.singleShot(100, self.close)
            event.ignore()
            return
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    selection = ModelSelectionDialog()
    while selection.exec_() == QDialog.Accepted:
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                model = selection.create_model()
                main_window = Main(model)
                del model
            finally:
                QApplication.restoreOverrideCursor()
        except Exception as error:
            QMessageBox.warning(selection, '模型載入失敗', str(error))
            continue
        main_window.show()
        sys.exit(app.exec_())
