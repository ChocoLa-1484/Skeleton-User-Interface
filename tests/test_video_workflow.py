"""Run with: conda run -n Skeleton python -m unittest discover -s tests -v"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
_yolo_config = tempfile.TemporaryDirectory()
os.environ['YOLO_CONFIG_DIR'] = _yolo_config.name
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Src' / 'UI_Control'))

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from cv_utils.cv_control import JsonLoader, VideoLoader
from skeleton.detect_skeleton import PoseEstimater
from utils.selector import PersonSelector
from utils.model import Model
from tracker.mc_bot_sort import BoTSORT
from video_widget import PoseVideoTabControl


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


if __name__ == '__main__':
    unittest.main()
