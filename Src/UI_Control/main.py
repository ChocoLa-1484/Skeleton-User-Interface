import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from video_widget import PoseVideoTabControl
from ui.main_window import Ui_MainWindow
from utils.model import Model


class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.model = Model()
        self.init_tabs()

    def init_tabs(self):
        self.video_tab = PoseVideoTabControl(self.model)
        self.ui.Two_d_Tab.addTab(self.video_tab, "2D 影片")

    def closeEvent(self, event):
        worker = self.video_tab.batch_worker
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            if not self.video_tab.closing_after_batch:
                self.video_tab.closing_after_batch = True
                worker.finished.connect(self.close)
            event.ignore()
            return
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec_())
