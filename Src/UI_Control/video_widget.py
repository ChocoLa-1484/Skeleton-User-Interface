from PyQt5.QtWidgets import *
from PyQt5.QtGui import QColor, QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QPointF, QTimer
import numpy as np
import sys
import cv2
import os
from ui.video_ui import Ui_video_widget
import pandas as pd
from utils.vis_image import ImageDrawer
from utils.selector import PersonSelector
from cv_utils.cv_control import VideoLoader, JsonLoader
from skeleton.detect_skeleton import PoseEstimater
from utils.model import Model
from utils.paths import PROJECT_ROOT
from cv_utils.batch_process import BatchAnalysisThread, folder_jobs

class PoseVideoTabControl(QWidget):
    def __init__(self, model:Model, parent = None):
        super(PoseVideoTabControl, self).__init__(parent)
        self.ui = Ui_video_widget()
        self.ui.setupUi(self)
        self.model = model
        self.setupComponents()
        self.init_var()
        self.bind_ui()

    def bind_ui(self):
        self.ui.loadOriginalVideoBtn.clicked.connect(
            lambda: self.loadVideo(is_processed=False))
        self.ui.loadProcessedVideoBtn.clicked.connect(
            lambda: self.loadVideo(is_processed=True))
        self.ui.loadFolderBtn.clicked.connect(self.loadFolder)

        self.ui.playBtn.clicked.connect(self.play_btn_clicked)
        self.ui.backKeyBtn.clicked.connect(
            lambda: self.ui.frameSlider.setValue(self.ui.frameSlider.value() - 1)
        )
        self.ui.forwardKeyBtn.clicked.connect(
            lambda: self.ui.frameSlider.setValue(self.ui.frameSlider.value() + 1)
        )
        self.ui.frameSlider.valueChanged.connect(self.analyzeFrame)
        self.ui.FrameView.mousePressEvent = self.mousePressEvent
        self.ui.startCodeBtn.clicked.connect(self.toggleDetect)
        self.ui.selectCheckBox.stateChanged.connect(self.toggleSelect)
        self.ui.showSkeletonCheckBox.stateChanged.connect(self.toggleShowSkeleton)
        self.ui.showBboxCheckBox.stateChanged.connect(self.toggleShowBbox)

    def mousePressEvent(self, event):
        view_rect = self.ui.FrameView.rect()
        pos = event.pos()

        if not view_rect.contains(pos):
            return
        search_person_df = self.pose_estimater.getSelectablePersons(self.ui.frameSlider.value())
        scene_pos = self.ui.FrameView.mapToScene(pos)
        x, y = scene_pos.x(), scene_pos.y()

        if self.ui.selectCheckBox.isChecked():
            if event.button() == Qt.LeftButton:
                selected_id = self.person_selector.select(x, y, search_person_df)
                if selected_id is not None:
                    self.pose_estimater.setPersonId(selected_id)
                    self.analyzeFrame()

        if self.label_kpt:
            if event.button() == Qt.LeftButton:
                self.sendtoTable(x, y, 1)
            elif event.button() == Qt.RightButton:
                self.sendtoTable(0, 0, 0)
            self.label_kpt = False

        self.updateFrame(self.ui.frameSlider.value())

    def keyPressEvent(self, event):
        if self.batch_worker is not None:
            return
        if event.key() == ord('D') or event.key() == ord('d'):
            self.ui.frameSlider.setValue(self.ui.frameSlider.value() + 1)
        elif event.key() == ord('A') or event.key() == ord('a'):
            self.ui.frameSlider.setValue(self.ui.frameSlider.value() - 1)
        else:
            super().keyPressEvent(event)

    def setupComponents(self):
        self.person_selector = PersonSelector()
        self.pose_estimater = PoseEstimater(self.model)
        self.kpt_dict = self.pose_estimater.joints["haple"]["keypoints"]
        self.image_drawer = ImageDrawer(self.pose_estimater)
        self.video_loader = VideoLoader(self.image_drawer)

    def init_var(self):
        self.is_play = False
        self.is_processing = False
        self.video_scene = QGraphicsScene()
        self.video_scene.clear()
        self.correct_kpt_idx = 0
        self.label_kpt = False
        self.is_processed = False
        self.batch_worker = None
        self.batch_progress = None
        self.closing_after_batch = False

    def reset(self):
        self.is_processing = False
        self.person_selector.reset()
        self.pose_estimater.reset()
        self.image_drawer.reset()
        self.video_scene.clear()
        for checkbox in (self.ui.selectCheckBox, self.ui.showSkeletonCheckBox, self.ui.showBboxCheckBox):
            blocked = checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(blocked)

    def loadVideo(self, is_processed:bool = False):
        if self.batch_worker is not None:
            return
        if self.is_play:
            self.ui.playBtn.click()
        if not self.video_loader.loadVideo():
            return
        self.is_processed = is_processed
        self.checkVideoLoad()

    def loadFolder(self):
        if self.batch_worker is not None:
            return
        if self.video_loader.is_loading:
            QMessageBox.information(self, "影片讀取中", "請等待目前影片讀取完成。")
            return
        folder = QFileDialog.getExistingDirectory(self, "選擇要批次分析的影片資料夾", str(PROJECT_ROOT))
        if not folder:
            return
        try:
            jobs = folder_jobs(folder)
        except OSError as error:
            QMessageBox.warning(self, "無法讀取資料夾", str(error))
            return
        if not jobs:
            QMessageBox.information(self, "沒有影片", "資料夾第一層沒有支援的影片。")
            return
        if self.is_play:
            self.play_btn_clicked()
        self.is_processing = False
        self.setBatchControlsEnabled(False)
        self.batch_progress = QProgressDialog("準備批次分析…", "取消", 0, len(jobs) * 100, self)
        self.batch_progress.setWindowTitle("ViTPose 資料夾分析")
        self.batch_progress.setWindowModality(Qt.WindowModal)
        self.batch_progress.setAutoClose(False)
        self.batch_progress.setAutoReset(False)
        self.batch_progress.setMinimumDuration(0)
        self.batch_worker = BatchAnalysisThread(self.model, jobs, parent=self)
        self.batch_worker.progress.connect(self.updateBatchProgress)
        self.batch_worker.finished.connect(self.finishBatch)
        self.batch_progress.canceled.connect(self.batch_worker.requestInterruption)
        self.batch_progress.show()
        self.batch_worker.start()

    def setBatchControlsEnabled(self, enabled):
        for control in (self.ui.FileGroupbox, self.ui.DisplayGroupBox, self.ui.FrameView,
                        self.ui.frameSlider, self.ui.playBtn, self.ui.backKeyBtn, self.ui.forwardKeyBtn):
            control.setEnabled(enabled)

    def updateBatchProgress(self, index, total, filename, frame, frame_count):
        detail = f"影格 {frame}/{frame_count}" if frame_count else "讀取影片中"
        self.batch_progress.setLabelText(f"{index}/{total}：{filename}\n{detail}")
        percent = min(100, int(frame * 100 / frame_count)) if frame_count else 0
        self.batch_progress.setValue((index - 1) * 100 + percent)

    def finishBatch(self):
        result = self.batch_worker.result
        self.batch_progress.close()
        self.batch_progress.deleteLater()
        self.setBatchControlsEnabled(True)
        self.batch_worker.deleteLater()
        self.batch_worker = None
        if self.closing_after_batch:
            return
        message = f"完成 {len(result['completed'])} 支，失敗 {len(result['errors'])} 支。"
        if result['cancelled']:
            message += "\n已取消剩餘影片，已完成的輸出會保留。"
        if result['errors']:
            dialog = QMessageBox(QMessageBox.Warning, "批次分析結果", message, QMessageBox.Ok, self)
            dialog.setDetailedText('\n'.join(f'{path}: {error}' for path, error in result['errors']))
            dialog.exec_()
        else:
            QMessageBox.information(self, "批次分析結果", message)

    def checkVideoLoad(self):
        if not self.video_loader.video_name:
            return
        if self.video_loader.is_loading:
            self.ui.videoNameLabel.setText("讀取影片中")
            QTimer.singleShot(100, self.checkVideoLoad)
            return
        self.updateVideoInfo()

    def updateVideoInfo(self):
        self.reset()
        self.initFrameSlider()
        self.model.setImageSize(self.video_loader.video_size)
        self.ui.videoNameLabel.setText(self.video_loader.video_name)
        video_size = self.video_loader.video_size
        self.ui.ResolutionLabel.setText(f"(0,0) - {video_size[0]} x {video_size[1]}")
        if self.is_processed:
            self.loadProcessedData()
        self.updateFrame(0)

    def loadProcessedData(self):
        json_loader = JsonLoader(self.video_loader.folder_path, self.video_loader.video_name)
        try:
            person_df, _, _ = json_loader.load()
            self.pose_estimater.setProcessedData(person_df)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "無法載入骨架資料", str(error))
            return
        self.ui.showSkeletonCheckBox.setChecked(True)
        self.ui.showBboxCheckBox.setChecked(True)

    def initFrameSlider(self):
        total_frames = self.video_loader.total_frames
        blocked = self.ui.frameSlider.blockSignals(True)
        self.ui.frameSlider.setMinimum(0)
        self.ui.frameSlider.setMaximum(total_frames - 1)
        self.ui.frameSlider.setValue(0)
        self.ui.frameSlider.blockSignals(blocked)
        self.ui.frameNumLabel.setText(f'0/{total_frames - 1}')

    def showImage(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView):
        scene.clear()
        image = image.copy()
        image = cv2.circle(image, (0, 0), 10, (0, 0, 255), -1)
        w, h = image.shape[1], image.shape[0]
        bytesPerline = 3 * w
        qImg = QImage(image, w, h, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        scene.addPixmap(pixmap)
        GraphicsView.setScene(scene)
        GraphicsView.setAlignment(Qt.AlignLeft)
        GraphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def playFrame(self, start_num=0):
        for i in range(start_num, self.video_loader.total_frames):
            if not self.is_play:
                break
            self.ui.frameSlider.setValue(i)
            if i == self.video_loader.total_frames - 1 and self.is_play:
                self.play_btn_clicked()
            cv2.waitKey(15)

    def play_btn_clicked(self):
        if not self.video_loader.video_name:
            QMessageBox.warning(self, "無法播放影片", "請讀取影片!")
            return
        if self.video_loader.is_loading:
            QMessageBox.warning(self, "影片讀取中", "請稍等!")
            return
        self.is_play = not self.is_play
        self.ui.playBtn.setText("||" if self.is_play else "▶︎")
        if self.is_play:
            self.playFrame(self.ui.frameSlider.value())

    def analyzeFrame(self):
        if self.batch_worker is not None:
            return
        if self.video_loader.is_loading or not self.video_loader.video_frames:
            return
        frame_num = self.ui.frameSlider.value()
        self.ui.frameNumLabel.setText(f'{frame_num}/{len(self.video_loader.video_frames) - 1}')
        frame = self.video_loader.getVideoImage(frame_num)
        if not self.is_processed and self.pose_estimater.is_detect:
            self.pose_estimater.ensureTrackedThrough(self.video_loader.video_frames, frame_num)
        _, _, fps, _ = self.pose_estimater.detectKpt(frame, frame_num, is_video=True, is_processed=self.is_processed)
        self.ui.FPSInfoLabel.setText(str(fps) if not self.is_processed else "—")
        if self.is_processing and not self.is_processed and frame_num == self.video_loader.total_frames - 1:
            self.is_processing = False
            try:
                self.video_loader.saveVideo(False)
                self.video_loader.saveVideo(True)
            except (OSError, ValueError) as error:
                self.is_play = False
                self.ui.playBtn.setText("▶︎")
                QMessageBox.warning(self, "儲存失敗", str(error))
        self.updateFrame(frame_num)

    def updateFrame(self, frame_num:int):
        frames = self.video_loader.video_frames
        if frames is None or not 0 <= frame_num < len(frames):
            return
        image = self.video_loader.getVideoImage(frame_num)
        drawed_img = self.image_drawer.drawInfo(image, frame_num, self.pose_estimater.kpt_buffer)
        if self.ui.selectCheckBox.isChecked():
            drawed_img = self.image_drawer.drawBbox(
                drawed_img, self.pose_estimater.getSelectablePersons(frame_num))
        self.showImage(drawed_img, self.video_scene, self.ui.FrameView)

    def toggleDetect(self):
        if self.video_loader.is_loading or not self.video_loader.video_frames:
            QMessageBox.warning(self, "無法處理影片", "請先讀取影片並等待讀取完成！")
            return
        if self.is_play:
            self.play_btn_clicked()
        if self.is_processed:
            QMessageBox.information(self, "已載入骨架資料", "目前使用已儲存的骨架。若要重新偵測與儲存，請以「載入原始影片」開啟影片。")
            return
        self.ui.showSkeletonCheckBox.setChecked(True)
        self.ui.showBboxCheckBox.setChecked(True)
        self.pose_estimater.setDetect(True)
        self.is_processing = True
        if self.ui.frameSlider.value() == 0:
            self.analyzeFrame()
        else:
            self.ui.frameSlider.setValue(0)
        self.ui.playBtn.click()

    def toggleSelect(self, state:int):
        if self.video_loader.is_loading or not self.video_loader.video_frames:
            self.ui.selectCheckBox.setChecked(False)
            return
        if state == 2:
            if not self.ui.showSkeletonCheckBox.isChecked():
                self.ui.showSkeletonCheckBox.setChecked(True)
        else:
            self.person_selector.reset()
            self.pose_estimater.setPersonId(None)
        self.analyzeFrame()

    def toggleShowSkeleton(self, state:int):
        if state == 2:
            self.pose_estimater.setDetect(True)
            self.image_drawer.setShowSkeleton(True)
        else:
            self.pose_estimater.setDetect(False)
            self.image_drawer.setShowSkeleton(False)
        self.updateFrame(self.ui.frameSlider.value())

    def toggleShowBbox(self, state:int):
        if state == 2:
            self.image_drawer.setShowBbox(True)
        else:
            self.image_drawer.setShowBbox(False)
        frame_num = self.ui.frameSlider.value()
        frames = self.video_loader.video_frames
        if frames is not None and 0 <= frame_num < len(frames):
            self.updateFrame(frame_num)

    def clearTableView(self):
        self.ui.KptTable.clear()
        self.ui.KptTable.setColumnCount(4)
        title = ["關節點", "X", "Y", "有無更改"]
        self.ui.KptTable.setHorizontalHeaderLabels(title)
        header = self.ui.KptTable.horizontalHeader()
        for i in range(4):
            header.setDefaultAlignment(Qt.AlignLeft)

    def importDatatoTable(self, frame_num:int):
        self.clearTableView()
        person_id = self.pose_estimater.person_id
        if person_id is None:
            return
        person_data = self.pose_estimater.getPersonDf(frame_num=frame_num, is_select=True)
        if person_data.empty:
            self.clearTableView()
            self.ui.selectCheckBox.setChecked(False)
            return

        num_keypoints = len(self.kpt_dict)
        if self.ui.KptTable.rowCount() < num_keypoints:
            self.ui.KptTable.setRowCount(num_keypoints)

        for kpt_idx, kpt in enumerate(person_data['keypoints'].iloc[0]):
            kptx, kpty, kpt_label = kpt[0], kpt[1], kpt[3]
            kpt_name = self.kpt_dict[kpt_idx]
            kpt_name_item = QTableWidgetItem(str(kpt_name))
            kptx_item = QTableWidgetItem(str(np.round(kptx,1)))
            kpty_item = QTableWidgetItem(str(np.round(kpty,1)))
            if kpt_label :
                kpt_label_item = QTableWidgetItem("Y")
            else:
                kpt_label_item = QTableWidgetItem("N")
            kpt_name_item.setTextAlignment(Qt.AlignRight)
            kptx_item.setTextAlignment(Qt.AlignRight)
            kpty_item.setTextAlignment(Qt.AlignRight)
            kpt_label_item.setTextAlignment(Qt.AlignRight)
            self.ui.KptTable.setItem(kpt_idx, 0, kpt_name_item)
            self.ui.KptTable.setItem(kpt_idx, 1, kptx_item)
            self.ui.KptTable.setItem(kpt_idx, 2, kpty_item)
            self.ui.KptTable.setItem(kpt_idx, 3, kpt_label_item)

    def onCellClicked(self, row, column):
        self.correct_kpt_idx = row
        self.label_kpt = True

    def sendtoTable(self, kptx:float, kpty:float, kpt_label:int):
        kptx_item = QTableWidgetItem(str(kptx))
        kpty_item = QTableWidgetItem(str(kpty))
        if kpt_label :
            kpt_label_item = QTableWidgetItem("Y")
        else:
            kpt_label_item = QTableWidgetItem("N")
        kptx_item.setTextAlignment(Qt.AlignRight)
        kpty_item.setTextAlignment(Qt.AlignRight)
        kpt_label_item.setTextAlignment(Qt.AlignRight)
        self.ui.KptTable.setItem(self.correct_kpt_idx, 1, kptx_item)
        self.ui.KptTable.setItem(self.correct_kpt_idx, 2, kpty_item)
        self.ui.KptTable.setItem(self.correct_kpt_idx, 3, kpt_label_item)

        self.pose_estimater.update_person_df(kptx, kpty, self.ui.frameSlider.value(), self.correct_kpt_idx)
        self.updateFrame(frame_num=self.ui.frameSlider.value())

    def correctId(self):
        before_correctId = self.ui.beforeCorrectId.value()
        after_correctId = self.ui.afterCorrectId.value()
        self.pose_estimater.correct_person_id(before_correctId, after_correctId)
        self.updateFrame(self.ui.frameSlider.value())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PoseVideoTabControl()
    window.show()
    sys.exit(app.exec_())
