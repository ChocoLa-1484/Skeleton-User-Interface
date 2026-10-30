"""Run with: conda run -n Skeleton python -m unittest discover -s tests -v"""
import os
import sys
import tempfile
import unittest
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
_yolo_config = tempfile.TemporaryDirectory()
os.environ['YOLO_CONFIG_DIR'] = _yolo_config.name
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Src' / 'UI_Control'))

import numpy as np
import pandas as pd
import cv2
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from cv_utils.cv_control import JsonLoader, VideoLoader
from skeleton.detect_skeleton import PoseEstimater
from utils.selector import PersonSelector
from utils.model import Model
from tracker.mc_bot_sort import BoTSORT
from tracker.basetrack import BaseTrack
from video_widget import PoseVideoTabControl
from cv_utils.batch_process import BatchAnalysisThread, BatchCancelled, folder_jobs
from utils.paths import PROJECT_ROOT, PRETRAIN_ROOT, RECORD_ROOT


def person(person_id=17, frame=0):
    return {'person_id': person_id, 'bbox': [10, 20, 100, 200],
            'keypoints': [[30, 40, 0.95, 0]] * 26, 'frame_number': frame}


class VideoWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_json_alias_and_no_legacy_files(self):
        with tempfile.TemporaryDirectory() as folder:
            pd.DataFrame([person()]).to_json(Path(folder) / 'test.json', orient='records')
            for name in ('test', 'test_Sk26'):
                data, bat, swing = JsonLoader(folder, name).load()
                self.assertEqual(data.iloc[0]['person_id'], 17)
                self.assertTrue(bat.empty and swing.empty)
            with self.assertRaises(FileNotFoundError):
                JsonLoader(folder, 'missing').load()

    def test_empty_and_invalid_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.json'
            path.write_text('[]', encoding='utf-8')
            data, _, _ = JsonLoader(folder, 'test').load()
            self.assertIn('frame_number', data.columns)
            path.write_text('[{"wrong": 1}]', encoding='utf-8')
            with self.assertRaises(ValueError):
                JsonLoader(folder, 'test').load()

    def test_loaded_frames_never_infer_including_empty_frames(self):
        estimator = PoseEstimater(SimpleNamespace())
        estimator.setProcessedData(pd.DataFrame([person()]))
        estimator.processImage = Mock(side_effect=AssertionError('Unexpected inference'))
        estimator.setDetect(True)
        estimator.setPersonId(17)
        for frame in (0, 1, 30):
            estimator.detectKpt(None, frame, is_video=True, is_processed=True)
        estimator.processImage.assert_not_called()

    def test_selected_id_filters_inference_and_missing_target_does_not_fallback(self):
        estimator = PoseEstimater(SimpleNamespace(pose_estimator=object()))
        estimator.tracked_frames[0] = pd.DataFrame([
            {'person_id': 17, 'bbox': [10, 20, 100, 200]},
            {'person_id': 42, 'bbox': [200, 20, 300, 200]},
        ])
        with patch('skeleton.detect_skeleton.inference_topdown', return_value=[]) as infer, \
                patch('skeleton.detect_skeleton.merge_data_samples', return_value={'pred_instances': 'pose'}):
            _, ids, _, _ = estimator.processImage(estimator.model, None, select_id=42, frame_num=0)
            self.assertEqual(ids, [42])
            np.testing.assert_array_equal(infer.call_args[0][2], [[200, 20, 300, 200]])
            infer.reset_mock()
            self.assertIsNone(estimator.processImage(estimator.model, None, select_id=99, frame_num=0)[0])
            infer.assert_not_called()

    def test_selection_change_reprocesses_same_frame_without_duplicate_rows(self):
        estimator = PoseEstimater(SimpleNamespace())
        estimator.setDetect(True)
        estimator.smoothKpt = Mock()
        estimator.processImage = Mock(return_value=(None, [], [], None))
        estimator.setProcessedData(pd.DataFrame([person(17), person(42)]))
        estimator.setPersonId(42)
        estimator.detectKpt(None, 0, is_video=True)
        self.assertEqual(estimator.processImage.call_count, 1)
        self.assertTrue(estimator.person_df.empty)
        estimator.detectKpt(None, 0, is_video=True)
        self.assertEqual(estimator.processImage.call_count, 1)
        estimator.setPersonId(None)
        estimator.detectKpt(None, 0, is_video=True)
        self.assertEqual(estimator.processImage.call_count, 2)

    def test_tracks_are_chronological_cached_and_keep_real_ids(self):
        estimator = PoseEstimater(SimpleNamespace())
        seen = []
        def track(model, image, frame):
            seen.append(frame)
            estimator.tracked_frames[frame] = pd.DataFrame()
        estimator.trackFrame = track
        estimator.ensureTrackedThrough([None] * 5, 3)
        estimator.ensureTrackedThrough([None] * 5, 1)
        estimator.ensureTrackedThrough([None] * 5, 4)
        self.assertEqual(seen, [0, 1, 2, 3, 4])
        targets = [SimpleNamespace(track_id=42, tlwh=[20, 30, 100, 200])]
        boxes, ids = estimator.filterValidTargets(targets, select_id=42)
        self.assertEqual(ids, [42])
        self.assertEqual(boxes, [[20, 30, 120, 230]])

    def test_real_tracker_keeps_ids_when_detection_order_changes(self):
        with patch.object(sys, 'argv', ['test']):
            args = Model.setTrackerParser(None)
        args.cmc_method = 'none'
        tracker = BoTSORT(args, frame_rate=30)
        image = np.zeros((240, 640, 3), dtype=np.uint8)
        first = np.array([[20, 20, 100, 200, 0.95, 0], [400, 20, 480, 200, 0.95, 0]])
        targets = tracker.update(first, image)
        initial = {int(target.tlwh[0] > 200): target.track_id for target in targets}
        second = first[::-1].copy()
        second[:, [0, 2]] += 2
        targets = tracker.update(second, image)
        actual = {int(target.tlwh[0] > 200): target.track_id for target in targets}
        self.assertEqual(len(initial), 2)
        self.assertEqual(actual, initial)

    def test_export_keeps_selected_target_and_rechecks_every_frame(self):
        estimator = SimpleNamespace(is_detect=False, person_id=42,
                                    ensureTrackedThrough=Mock(), detectKpt=Mock())
        estimator.setDetect = lambda value: setattr(estimator, 'is_detect', value)
        loader = VideoLoader(SimpleNamespace(pose_estimater=estimator))
        loader.video_frames = [None, None, None]
        loader.ensureVideoProcessed()
        self.assertEqual(estimator.person_id, 42)
        self.assertFalse(estimator.is_detect)
        self.assertEqual([call.args[1] for call in estimator.detectKpt.call_args_list], [0, 1, 2])

    def test_selector_no_hit_does_not_reuse_previous_id(self):
        selector = PersonSelector()
        data = pd.DataFrame([person(42)])
        self.assertEqual(selector.select(50, 50, data), 42)
        self.assertIsNone(selector.select(500, 500, data))

    def test_widget_loads_saved_skeleton_and_selects_without_model(self):
        widget = PoseVideoTabControl(SimpleNamespace())
        widget.video_loader.video_frames = [np.zeros((240, 320, 3), dtype=np.uint8)] * 2
        widget.video_loader.total_frames = 2
        widget.video_loader.video_name = 'test_Sk26'
        widget.is_processed = True
        with tempfile.TemporaryDirectory() as folder:
            pd.DataFrame([person()]).to_json(Path(folder) / 'test.json', orient='records')
            widget.video_loader.folder_path = folder
            widget.loadProcessedData()
            self.assertTrue(widget.ui.showSkeletonCheckBox.isChecked())
            self.assertTrue(widget.ui.showBboxCheckBox.isChecked())
            widget.ui.selectCheckBox.setChecked(True)
            widget.pose_estimater.setPersonId(17)
            widget.analyzeFrame()
            widget.ui.frameSlider.setValue(1)
            self.assertTrue(widget.pose_estimater.getPersonDf(frame_num=1).empty)
        widget.close()

    def test_widget_click_changes_pose_target_and_uncheck_restores_all(self):
        widget = PoseVideoTabControl(SimpleNamespace())
        widget.video_loader.video_frames = [np.zeros((240, 320, 3), dtype=np.uint8)]
        widget.video_loader.total_frames = 1
        widget.pose_estimater.tracked_frames[0] = pd.DataFrame([
            {'person_id': 17, 'bbox': [10, 20, 100, 200]},
            {'person_id': 42, 'bbox': [200, 20, 300, 200]},
        ])
        widget.pose_estimater.processImage = Mock(return_value=(None, [], [], None))
        widget.ui.selectCheckBox.setChecked(True)
        widget.mousePressEvent(SimpleNamespace(
            pos=lambda: widget.ui.FrameView.mapFromScene(250, 100),
            button=lambda: Qt.LeftButton))
        self.assertEqual(widget.pose_estimater.person_id, 42)
        self.assertEqual(widget.pose_estimater.processImage.call_args.kwargs['select_id'], 42)
        widget.ui.selectCheckBox.setChecked(False)
        self.assertIsNone(widget.pose_estimater.person_id)
        self.assertIsNone(widget.pose_estimater.processImage.call_args.kwargs['select_id'])
        widget.close()

    def test_process_button_enables_both_overlays(self):
        widget = PoseVideoTabControl(SimpleNamespace())
        widget.video_loader.video_frames = [np.zeros((240, 320, 3), dtype=np.uint8)]
        with patch.object(widget, 'analyzeFrame'), patch.object(widget.ui.playBtn, 'click'):
            widget.toggleDetect()
        self.assertTrue(widget.ui.showSkeletonCheckBox.isChecked())
        self.assertTrue(widget.ui.showBboxCheckBox.isChecked())
        self.assertTrue(widget.image_drawer.show_bbox)
        self.assertTrue(widget.image_drawer.show_skeleton)
        widget.close()

    def test_default_paths_do_not_depend_on_working_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as folder, patch.object(sys, 'argv', ['test']):
            try:
                os.chdir(folder)
                yolo = Model.setDetectParserYolo(None)
                pose = Model.setPoseParser(None)
                tracker = Model.setTrackerParser(None)
                loader = VideoLoader()
            finally:
                os.chdir(original)
        self.assertEqual(Path(yolo.model), PRETRAIN_ROOT / 'yolo11m-seg.pt')
        self.assertTrue(Path(pose.pose_config).is_file())
        self.assertTrue(Path(pose.pose_config).is_absolute())
        self.assertTrue(Path(tracker.fast_reid_config).is_absolute())
        self.assertEqual(loader.output_root, RECORD_ROOT)
        self.assertEqual(RECORD_ROOT, PROJECT_ROOT / 'Db' / 'Record')

    def test_folder_discovery_ignores_subfolders_and_avoids_name_collisions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ['clip.mp4', 'clip.AVI', 'clip_mp4.mp4', 'other.MOV', 'notes.txt']:
                (root / name).touch()
            (root / 'nested').mkdir()
            (root / 'nested' / 'hidden.mp4').touch()
            jobs = folder_jobs(root)
            self.assertEqual(len(jobs), 4)
            self.assertEqual(len({name.casefold() for _, name in jobs}), 4)
            self.assertEqual([path.name for path, _ in jobs], ['clip.AVI', 'clip.mp4', 'clip_mp4.mp4', 'other.MOV'])

    @staticmethod
    def write_clip(path):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 24, (320, 240))
        if not writer.isOpened():
            raise RuntimeError('Test video codec unavailable')
        try:
            for _ in range(3):
                writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        finally:
            writer.release()

    @staticmethod
    def batch_model():
        model = SimpleNamespace(tracker=object(), image_size=(10, 10))
        def set_size(size):
            model.tracker = object()
            model.image_size = size
            BaseTrack.clear_count()
        model.setImageSize = set_size
        return model

    @staticmethod
    def batch_estimator(model):
        estimator = PoseEstimater(model)
        def detect(frame, frame_num, **kwargs):
            estimator.person_df = pd.concat([
                estimator.person_df, pd.DataFrame([person(17, frame_num)])
            ], ignore_index=True)
        estimator.detectKpt = detect
        return estimator

    def test_batch_writes_all_outputs_and_continues_after_bad_video(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root / 'inputs'
            inputs.mkdir()
            self.write_clip(inputs / 'one.mp4')
            self.write_clip(inputs / 'two.mp4')
            (inputs / 'bad.mp4').write_bytes(b'not a video')
            model = self.batch_model()
            original_tracker = model.tracker
            original_track_count = BaseTrack.next_id()
            worker = BatchAnalysisThread(model, folder_jobs(inputs), root / 'outputs')
            with patch('cv_utils.batch_process.PoseEstimater', side_effect=self.batch_estimator):
                worker.run()
            self.assertEqual(len(worker.result['completed']), 2)
            self.assertEqual(len(worker.result['errors']), 1)
            self.assertIs(model.tracker, original_tracker)
            self.assertEqual(BaseTrack._count, original_track_count)
            self.assertEqual(model.image_size, (10, 10))
            for name in ['one', 'two']:
                output = root / 'outputs' / name
                data = json.loads((output / f'{name}.json').read_text())
                self.assertEqual([row['frame_number'] for row in data], [0, 1, 2])
                with (output / f'{name}.csv').open(encoding='utf-8-sig', newline='') as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 78)
                for suffix in ['', '_Sk26']:
                    capture = cv2.VideoCapture(str(output / f'{name}{suffix}.mp4'))
                    try:
                        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
                        self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 24, places=1)
                        ok, frame = capture.read()
                        self.assertTrue(ok)
                        if suffix:
                            self.assertGreater(int(frame.sum()), 0)
                    finally:
                        capture.release()
            self.assertEqual(list((root / 'outputs').glob('.batch-*')), [])

    def test_batch_cancel_preserves_completed_results_and_old_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_clip(root / 'one.mp4')
            self.write_clip(root / 'two.mp4')
            outputs = root / 'outputs'
            (outputs / 'two').mkdir(parents=True)
            old = outputs / 'two' / 'two.json'
            old.write_text('old result')
            model = self.batch_model()
            original_tracker = model.tracker
            worker = BatchAnalysisThread(model, [(root / 'one.mp4', 'one'), (root / 'two.mp4', 'two')], outputs)
            def estimator_for_video(model):
                estimator = self.batch_estimator(model)
                if worker.result['completed']:
                    estimator.detectKpt = Mock(side_effect=BatchCancelled())
                return estimator
            with patch('cv_utils.batch_process.PoseEstimater', side_effect=estimator_for_video):
                worker.run()
            self.assertTrue(worker.result['cancelled'])
            self.assertEqual(len(worker.result['completed']), 1)
            self.assertFalse(worker.result['errors'])
            self.assertEqual(old.read_text(), 'old result')
            self.assertTrue((outputs / 'one' / 'one_Sk26.mp4').is_file())
            self.assertEqual(list(outputs.glob('.batch-*')), [])
            self.assertIs(model.tracker, original_tracker)


if __name__ == '__main__':
    unittest.main()
