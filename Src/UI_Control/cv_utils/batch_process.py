"""Sequential folder analysis without retaining decoded videos in memory."""
import math
import os
import tempfile
from pathlib import Path

import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from cv_utils.cv_control import VideoLoader
from skeleton.detect_skeleton import PoseEstimater
from utils.paths import RECORD_ROOT
from utils.vis_image import ImageDrawer
from tracker.basetrack import BaseTrack


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.mpg', '.mpeg', '.wmv'}


def folder_jobs(folder):
    videos = sorted((path for path in Path(folder).iterdir()
                     if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
                    key=lambda path: (path.name.casefold(), path.name))
    stems = [path.stem.casefold() for path in videos]
    used = set()
    jobs = []
    for path in videos:
        name = path.stem if stems.count(path.stem.casefold()) == 1 else f'{path.stem}_{path.suffix[1:].lower()}'
        base = name
        index = 2
        while name.casefold() in used:
            name = f'{base}_{index}'
            index += 1
        used.add(name.casefold())
        jobs.append((path, name))
    return jobs


class BatchCancelled(Exception):
    pass


class BatchAnalysisThread(QThread):
    progress = pyqtSignal(int, int, str, int, int)

    def __init__(self, model, jobs, output_root=None, parent=None):
        super().__init__(parent)
        self.model = model
        self.jobs = list(jobs)
        self.output_root = Path(output_root) if output_root is not None else RECORD_ROOT
        self.result = {'completed': [], 'errors': [], 'cancelled': False}

    def check_cancelled(self):
        if self.isInterruptionRequested():
            raise BatchCancelled()

    def run(self):
        # The GUI's loaded video can resume with its original tracker afterwards.
        original_tracker = self.model.tracker
        original_size = self.model.image_size
        original_track_count = BaseTrack._count
        try:
            for index, (path, name) in enumerate(self.jobs, 1):
                try:
                    self.check_cancelled()
                    self.progress.emit(index, len(self.jobs), path.name, 0, 0)
                    output = self.analyze_video(path, name, index)
                    self.result['completed'].append(str(output))
                except BatchCancelled:
                    self.result['cancelled'] = True
                    break
                except Exception as error:
                    self.result['errors'].append((str(path), str(error)))
        finally:
            self.model.tracker = original_tracker
            self.model.image_size = original_size
            BaseTrack._count = original_track_count

    def analyze_video(self, path, name, index):
        self.output_root.mkdir(parents=True, exist_ok=True)
        # Publish only complete videos; cancellation leaves earlier results intact.
        with tempfile.TemporaryDirectory(prefix='.batch-', dir=str(self.output_root)) as staging:
            staging = Path(staging)
            capture = cv2.VideoCapture(str(path))
            writers = []
            try:
                if not capture.isOpened():
                    raise OSError('無法開啟影片')
                fps = capture.get(cv2.CAP_PROP_FPS)
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if not math.isfinite(fps) or fps <= 0:
                    raise ValueError('影片 FPS 無效')
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('影片沒有可讀取的影格')
                size = (frame.shape[1], frame.shape[0])
                self.model.setImageSize(size)
                estimator = PoseEstimater(self.model)
                estimator.setDetect(True)
                drawer = ImageDrawer(estimator)
                for suffix in ('', '_Sk26'):
                    writer = cv2.VideoWriter(str(staging / f'{name}{suffix}.mp4'),
                                             cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
                    writers.append(writer)
                    if not writer.isOpened():
                        raise OSError('無法建立輸出影片')
                frame_num = 0
                while ok:
                    self.check_cancelled()
                    estimator.detectKpt(frame, frame_num, is_video=True)
                    annotated = drawer.drawInfo(frame, frame_num, show_bbox=True,
                                                show_skeleton=True, is_select=False)
                    writers[0].write(frame)
                    writers[1].write(annotated)
                    frame_num += 1
                    if frame_num % 10 == 0 or frame_num == count:
                        self.progress.emit(index, len(self.jobs), path.name, frame_num, count)
                    ok, frame = capture.read()
                if count > 0 and frame_num < count:
                    raise OSError(f'影片讀取中斷：只讀取 {frame_num}/{count} 個影格')
                self.check_cancelled()
                loader = VideoLoader(drawer, self.output_root)
                loader.video_name = name
                loader.saveSkeletonData(staging)
            finally:
                capture.release()
                for writer in writers:
                    writer.release()
            self.check_cancelled()
            output = self.output_root / name
            output.mkdir(parents=True, exist_ok=True)
            for file in staging.iterdir():
                os.replace(str(file), str(output / file.name))
            self.progress.emit(index, len(self.jobs), path.name, frame_num, frame_num)
            return output
