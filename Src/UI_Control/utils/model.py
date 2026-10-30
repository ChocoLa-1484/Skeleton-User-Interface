from argparse import ArgumentParser
import argparse
import sys
from utils.paths import SRC_ROOT, PRETRAIN_ROOT
from ultralytics import YOLO
sys.path.append(str(SRC_ROOT / 'tracker'))
from tracker.mc_bot_sort import BoTSORT
import mmpretrain  # noqa: F401 — registers mmpretrain.VisionTransformer into mmengine registry
from mmpose.apis import init_model as init_pose_estimator


class Model(object):
    def __init__(self):
        self.detect_args_yolo = self.setDetectParserYolo()
        self.pose_args = self.setPoseParser()
        self.tracker_args = self.setTrackerParser()
        self.detector = YOLO(self.detect_args_yolo.model)
        self.pose_estimator = init_pose_estimator(
            self.pose_args.pose_config,
            self.pose_args.pose_checkpoint
        )
        self.tracker = BoTSORT(self.tracker_args, frame_rate=30.0)
        self.image_size = (0, 0, 0)

    def setDetectParserYolo(self) -> ArgumentParser:
        parser = argparse.ArgumentParser(description="YOLO Model Argument Parsing")
        parser.add_argument('--model', type=str, default=str(PRETRAIN_ROOT / 'yolo11m-seg.pt'))
        parser.add_argument('--conf-thres', type=float, default=0.3)
        parser.add_argument('--iou-thres', type=float, default=0.3)
        return parser.parse_args()

    def setPoseParser(self) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument('--pose-config', default=str(SRC_ROOT / 'mmpose_main/configs/body_2d_keypoint/topdown_heatmap/haple/ViTPose_base_simple_halpe_256x192.py'))
        parser.add_argument('--pose-checkpoint', default=str(PRETRAIN_ROOT / 'epoch_210.pth'))
        parser.add_argument('--device', default='cuda:0')
        parser.add_argument('--kpt-thr', type=float, default=0.3)
        parser.add_argument('--show-kpt-idx', action='store_true', default=False)
        parser.add_argument('--skeleton-style', default='mmpose', type=str, choices=['mmpose', 'openpose'])
        parser.add_argument('--radius', type=int, default=3)
        return parser.parse_args()

    def setTrackerParser(self) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument("--track_high_thresh", type=float, default=0.3)
        parser.add_argument("--track_low_thresh", default=0.05, type=float)
        parser.add_argument("--new_track_thresh", default=0.4, type=float)
        parser.add_argument("--track_buffer", type=int, default=360)
        parser.add_argument("--match_thresh", type=float, default=0.8)
        parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6)
        parser.add_argument('--min_box_area', type=float, default=10)
        parser.add_argument("--fuse-score", dest="mot20", default=True, action='store_true')
        parser.add_argument("--cmc-method", default="sparseOptFlow", type=str)
        parser.add_argument("--with-reid", dest="with_reid", default=False)
        parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=str(SRC_ROOT / 'tracker/fast_reid/configs/MOT17/sbs_S50.yml'), type=str)
        parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=str(SRC_ROOT / 'tracker/fast_reid/mot17_sbs_S50.pth'), type=str)
        parser.add_argument('--proximity_thresh', type=float, default=0.5)
        parser.add_argument('--appearance_thresh', type=float, default=0.25)
        tracker_args = parser.parse_args()
        tracker_args.jde = False
        tracker_args.ablation = False
        return tracker_args

    def init_tracker(self):
        self.tracker = BoTSORT(self.tracker_args, frame_rate=30.0)

    def reset_tracker(self):
        self.tracker = BoTSORT(self.tracker_args, frame_rate=30.0)

    def setImageSize(self, image_size: tuple):
        self.reset_tracker()
        self.image_size = image_size
