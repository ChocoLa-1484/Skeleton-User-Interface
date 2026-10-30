import numpy as np
import pandas as pd
from utils.one_euro_filter import OneEuroFilter
import numpy as np
from utils.model import Model
from mmengine.logging import print_log
from scipy.signal import savgol_filter
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from utils.timer import FPSTimer
from ultralytics import YOLO
import torch
import torch.autograd.profiler as profiler
import cv2

try:
    from mmdet.apis import inference_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

class PoseEstimater:
    def __init__(self, model: Model =None):
        
        self.model = model
        self.img_shape = (0,0,0)
        self.person_df = pd.DataFrame()
        self.pre_person_df = pd.DataFrame()

        self.bat_df = pd.DataFrame()
        self.has_pre_bat_df = False
        self.pre_bat_df = pd.DataFrame()
        self.bat_data = []
        self.person_id = None
        self.kpt_id = None
        self.pitch_hand_id = 10
        self.fps = None
        self.person_data = []
        self.processed_frames = set()
        self.processed_targets = {}
        self.tracked_frames = {}
        self.fps_timer = FPSTimer()
        self.smooth_filter = OneEuroFilter()
        self.is_detect = False
        self.kpt_buffer = []
        self.joints = {
            "coco": {
                "keypoints": {
                    0: "鼻子",
                    1: "左眼",
                    2: "右眼",
                    3: "左耳",
                    4: "右耳",
                    5: "左肩",
                    6: "右肩",
                    7: "左手肘",
                    8: "右手肘",
                    9: "左手腕",
                    10: "右手腕",
                    11: "左髋",
                    12: "右髋",
                    13: "左膝",
                    14: "右膝",
                    15: "左腳踝",
                    16: "右腳踝"
                },
                "skeleton_links": [
                    [0, 1], [0, 2], [1, 3], [2, 4], # 頭
                    #軀幹
                    [5, 7], [7, 9],                 #左手
                    [6, 8], [8, 10],                #右手
                    [11, 13], [13, 15],   #左腿
                    [12, 14], [14, 16],   #右腿
                ],
                "left_points_indices": [[5, 7], [7, 9], [11, 13], [13, 15]],  # Indices of left hand, leg, and foot keypoints
                "right_points_indices": [[6, 8], [8, 10], [12, 14], [14, 16]]  # Indices of right hand, leg, and foot keypoints
            },
            "haple":{
                "keypoints": {
                    0: "鼻子",
                    1: "左眼",
                    2: "右眼",
                    3: "左耳",
                    4: "右耳",
                    5: "左肩",
                    6: "右肩",
                    7: "左肘",
                    8: "右肘",
                    9: "左腕",
                    10: "右腕",
                    11: "左髖",
                    12: "右髖",
                    13: "左膝",
                    14: "右膝",
                    15: "左踝",
                    16: "右踝",
                    17: "頭部",
                    18: "頸部",
                    19: "臀部",
                    20: "左大腳趾",
                    21: "右大腳趾",
                    22: "左小腳趾",
                    23: "右小腳趾",
                    24: "左腳跟",
                    25: "右腳跟"
                },
                "skeleton_links":[
                    [0, 1], [0, 2], [1, 3], [2, 4], # 頭
                    [5, 18], [6, 18], [17, 18],[18, 19],#軀幹
                    [5, 7], [7, 9],                 #左手
                    [6, 8], [8, 10],                #右手
                    [19, 11], [11, 13], [13, 15],   #左腿
                    [19, 12], [12, 14], [14, 16],   #右腿
                    [20, 24], [22, 24], [15, 24],   #左腳
                    [21, 25], [23, 25], [16, 25]    #右腳
                ],
                
                "left_points_indices": [[5, 18], [5, 7], [7, 9],[19, 11], [11, 13], [13, 15], [20, 24], [22, 24], [15, 24]],  # Indices of left hand, leg, and foot keypoints
                "right_points_indices": [[6, 18], [6, 8], [8, 10], [19, 12], [12, 14], [14, 16], [21, 25], [23, 25], [16, 25]],  # Indices of right hand, leg, and foot keypoints
                "angle_dict":{
                    # 'l_elbow_angle': [5, 7, 9],
                    '右手肘': [6, 8, 10]
                    # 'l_shoulder_angle': [18, 5, 7],
                    # 'r_shoulder_angle': [18, 6, 8],
                    # 'l_knee_angle': [11, 13, 15],
                    # 'r_knee_angle': [12, 14, 16]
                }
            },
        }

    def mergePersonData(self, pred_instances, person_ids: list, frame_num: int = None):
        if frame_num is None:
            self.person_data = []
        else:
            self.person_data = [row for row in self.person_df.to_dict('records')
                                if row['frame_number'] != frame_num]
        if pred_instances is None:
            return pd.DataFrame(self.person_data, columns=['person_id', 'bbox', 'keypoints', 'frame_number'])
        person_bboxes = pred_instances['bboxes']

        for person, pid, bbox in zip(pred_instances, person_ids, person_bboxes):
            keypoints_data = np.hstack((
                np.round(person['keypoints'][0], 2),
                np.round(person['keypoint_scores'][0], 2).reshape(-1, 1),
                np.full((len(person['keypoints'][0]), 1), False, dtype=bool)
            ))

            new_kpts = np.full((len(self.joints['haple']['keypoints']), keypoints_data.shape[1]), 0.9)
            new_kpts[:26] = keypoints_data
            person_info = {
                'person_id': pid,
                'bbox': bbox,
                'keypoints': new_kpts
            }
            if frame_num is not None:
                person_info['frame_number'] = frame_num

            self.person_data.append(person_info)

        return pd.DataFrame(self.person_data)
    
    def mergeBatData(self, bbox, mask,  frame_num:int = None):
        if mask == None:
            self.has_pre_bat_df = False
        else: # 有detect到東西
            self.has_pre_bat_df = True
            if frame_num is None:
                return pd.DataFrame()
            else:
                bat_info = {
                    'bbox': bbox,
                    'mask': mask,
                    'frame_number': frame_num
                }
                self.pre_bat_df.copy(bat_info)
                self.bat_data.append(bat_info)
                return pd.DataFrame(self.bat_data)
        
  
    def detectKpt(self, image:np.ndarray, frame_num:int = None, is_video:bool = False, is_processed:bool=False):
        if is_processed:
            return image, self.person_df, 0, self.bat_df
        if not self.is_detect:
            return image, pd.DataFrame(), 0, pd.DataFrame()
        self.fps_timer.tic()
        fps = 0
        if is_video: 
            #影片處理方式
            if (frame_num not in self.processed_frames or
                    self.processed_targets.get(frame_num) != self.person_id):
                pred_instances, person_ids, bat_bbox, bat_mask= self.processImage(
                    self.model, image, select_id=self.person_id, frame_num=frame_num)
                self.person_df = self.mergePersonData(pred_instances, person_ids, frame_num)
                self.bat_df = self.mergeBatData(bat_bbox, bat_mask, frame_num)
                self.smoothKpt(person_ids, frame_num)
                self.processed_frames.add(frame_num)
                self.processed_targets[frame_num] = self.person_id
            if self.kpt_id is not None:
                self.kpt_buffer = self.updateKptBuffer(frame_num)
        else:
            #real time 處理方式
            pred_instances, person_ids, bat_bbox, bat_mask = self.processImage(self.model, image, select_id=self.person_id)
            self.person_df = self.mergePersonData(pred_instances, person_ids)
            self.bat_df = self.mergeBatData(bat_bbox, bat_mask, frame_num)
            self.smoothKpt(person_ids, frame_num)
            self.pre_person_df = self.person_df.copy()
            if self.kpt_id:
                person_data = self.getPersonDf(is_select=True, is_kpt=True)
                if person_data is not None:
                    keypoint = person_data[self.kpt_id][:2]  # 確保取出的是有效的 [x, y] 數據
                    self.kpt_buffer.append(keypoint)

        average_time = self.fps_timer.toc()
        fps = int(1/max(average_time, 0.00001))

        return image, self.person_df, fps, self.bat_df

    def smoothKpt(self, person_ids: list, frame_num=None):
        # 如果是即時處理，則初始化前一幀的數據，否則依賴 frame_slider 進行處理
        if frame_num is not None:
            curr_frame = frame_num
            if curr_frame == 0:
                return  # 初始幀，無需處理
            pre_frame_num = curr_frame - 1
        
        # 用於即時處理時的前一幀數據
        if frame_num is None and self.pre_person_df.empty:
            self.pre_person_df = self.person_df.copy()

        # 當前幀無數據時，跳過處理
        if self.person_df.empty:
            return
        
        for person_id in person_ids:
            # 如果使用 frame_slider，根據前後幀數據進行處理
            if frame_num is not None:
                pre_person_data = self.person_df.loc[(self.person_df['frame_number'] == pre_frame_num) &
                                                    (self.person_df['person_id'] == person_id)]
                curr_person_data = self.person_df.loc[(self.person_df['frame_number'] == curr_frame) &
                                                    (self.person_df['person_id'] == person_id)]
            # 即時處理時，使用 self.pre_person_df 作為前幀數據
            else:
                pre_person_data = self.pre_person_df.loc[self.pre_person_df['person_id'] == person_id]
                curr_person_data = self.person_df.loc[self.person_df['person_id'] == person_id]
            
            if curr_person_data.empty or pre_person_data.empty:
                continue  # 當前幀或前幀沒有該 person_id 的數據，跳過
            
            pre_kpts = torch.tensor(pre_person_data.iloc[0]['keypoints'], device='cuda')
            curr_kpts = torch.tensor(curr_person_data.iloc[0]['keypoints'], device='cuda')
            smoothed_kpts = []

            # 使用張量運算來進行平滑
            for pre_kpt, curr_kpt in zip(pre_kpts, curr_kpts):
                pre_kptx, pre_kpty = pre_kpt[0], pre_kpt[1]
                curr_kptx, curr_kpty, curr_conf, curr_label = curr_kpt[0], curr_kpt[1], curr_kpt[2], curr_kpt[3]
                
                if all([pre_kptx.item() != 0, pre_kpty.item() != 0, curr_kptx.item() != 0, curr_kpty.item() != 0]):
                    curr_kptx = self.smooth_filter(curr_kptx, pre_kptx)
                    curr_kpty = self.smooth_filter(curr_kpty, pre_kpty)
                
                smoothed_kpts.append([curr_kptx.cpu().item(), curr_kpty.cpu().item(), curr_conf.item(), curr_label.item()])
            
            # 更新當前幀的數據
            self.person_df.at[curr_person_data.index[0], 'keypoints'] = smoothed_kpts
           
    def trackFrame(self, model, image, frame_num=None):
        if frame_num is not None and frame_num in self.tracked_frames:
            return self.tracked_frames[frame_num]
        detections = model.detector.predict(image, task='segment', classes=[0])[0]
        boxes = detections.boxes.xyxy.cpu().numpy()
        scores = detections.boxes.conf.cpu().numpy().reshape(-1, 1)
        tracker_input = np.hstack((boxes, scores, np.zeros((len(boxes), 1))))
        targets = model.tracker.update(tracker_input, image.copy())
        boxes, person_ids = self.filterValidTargets(targets)
        persons = pd.DataFrame([
            {'person_id': person_id, 'bbox': bbox}
            for bbox, person_id in zip(boxes, person_ids)
        ], columns=['person_id', 'bbox'])
        if frame_num is not None:
            self.tracked_frames[frame_num] = persons
        return persons

    def ensureTrackedThrough(self, video_frames, frame_num):
        # Feed the tracker in chronological order, even when the slider skips frames.
        for index in range(len(self.tracked_frames), frame_num + 1):
            self.trackFrame(self.model, video_frames[index], index)

    def getSelectablePersons(self, frame_num):
        if frame_num in self.tracked_frames:
            return self.tracked_frames[frame_num]
        return self.getPersonDf(frame_num=frame_num)

    def processImage(self, model, img, select_id=None, frame_num=None):
        persons = self.trackFrame(model, img, frame_num)
        if select_id is not None:
            persons = persons.loc[persons['person_id'] == select_id]
        if persons.empty:
            # Empty boxes would make inference_topdown estimate the entire image.
            return None, [], [], None
        boxes = np.asarray(persons['bbox'].tolist())
        person_ids = persons['person_id'].tolist()
        pose_results = inference_topdown(model.pose_estimator, img, boxes)
        data_samples = merge_data_samples(pose_results)
        return data_samples.get('pred_instances', None), person_ids, [], None

    def filterBatBiggestTarget(self, online_targets, masks):
        # 看有沒有內容
        if not online_targets: 
            return [], None
        # 如果有偵測到
        tlwhs = []
        track_ids = []
        # 把當前frame偵測到的bat加進去list
        for target in online_targets:
            tlwhs.append(target.tlwh)
            track_ids.append(target.track_id)

        # 將列表轉換為 NumPy array
        tlwhs = np.array(tlwhs)
        track_ids = np.array(track_ids)

        # 將數據轉為張量並放到 GPU 上
        tlwhs = torch.tensor(tlwhs, device='cuda')  # shape: (n, 4)
        track_ids = torch.tensor(track_ids, device='cuda')  # shape: (n,)

        # 計算面積 w * h
        areas = tlwhs[:, 2] * tlwhs[:, 3]  # w * h
        max_area_idx = torch.argmax(areas) #get max index

        # 找出最大的bat
        biggest_bbox = tlwhs[max_area_idx]  # 對應的邊界框 (x, y, w, h)
        biggest_bbox_xyxy = torch.cat([biggest_bbox[:2], biggest_bbox[:2] + biggest_bbox[2:4]], dim=0) # 轉乘xyxy
        biggest_mask = None

        # 找masks中對應biggest_bbox的mask
        biggest_mask_area = 0
        for mask in enumerate(masks):
            if (np.all(mask[1][0][:, 0] >= (biggest_bbox_xyxy.cpu().tolist()[0] -50)) and
                np.all(mask[1][0][:, 0] <= (biggest_bbox_xyxy.cpu().tolist()[2] + 50)) and
                np.all(mask[1][0][:, 1] >= (biggest_bbox_xyxy.cpu().tolist()[1] - 50)) and
                np.all(mask[1][0][:, 1] <= (biggest_bbox_xyxy.cpu().tolist()[3] + 50)) ):

                current_mask_area = cv2.contourArea(mask[1][0]) 
                if current_mask_area > biggest_mask_area:
                    biggest_mask = mask
                    biggest_mask_area = current_mask_area
        
        # 如果pre為空，則直接回傳
        if self.pre_bat_df.empty :
            return biggest_bbox_xyxy.cpu().tolist(), biggest_mask
        
        # 如果pre不為空，確認是否在有效範圍
        else: 
            pre_bat_bbox = self.pre_bat_df['bbox'].iloc[-1]
            pre_w = (pre_bat_bbox[2] + pre_bat_bbox[0]) / 2
            pre_h = (pre_bat_bbox[3] + pre_bat_bbox[1]) / 2

            valid_x_min = pre_bat_bbox[0] - (pre_w * 0.25)
            valid_x_max = pre_bat_bbox[2] + (pre_w * 0.25)
            valid_y_min = pre_bat_bbox[1] - (pre_h * 0.25)
            valid_y_max = pre_bat_bbox[3] + (pre_h * 0.25)
            
            bbox_x_min = biggest_bbox_xyxy.cpu().tolist()[0]
            bbox_x_max = biggest_bbox_xyxy.cpu().tolist()[2]
            bbox_y_min = biggest_bbox_xyxy.cpu().tolist()[1]
            bbox_y_max = biggest_bbox_xyxy.cpu().tolist()[3]

            # 選出最大的看是否在有效範圍內
            if (bbox_x_min >= valid_x_min) and (bbox_x_max <= valid_x_max) and (bbox_y_min >= valid_y_min) and (bbox_y_max <= valid_y_max):
                return biggest_bbox_xyxy.cpu().tolist(), biggest_mask
            else: 
                # 若不在有效範圍內不存
                return [], None


    def filterValidTargets(self, online_targets, select_id: int = None):
        boxes, person_ids = [], []
        for target in online_targets:
            x, y, width, height = target.tlwh
            if width * height <= 10:
                continue
            if select_id is not None and target.track_id != select_id:
                continue
            boxes.append([float(x), float(y), float(x + width), float(y + height)])
            person_ids.append(int(target.track_id))
        return boxes, person_ids

    def filterBiggestTargets(self, online_targets, select_id: int = None):
        if not online_targets:
            return [], []

        tlwhs = []
        track_ids = []

        for target in online_targets:
            tlwhs.append(target['tlwh'])  # 使用字典的 'tlwh' 键
            track_ids.append(target['track_id'])  # 使用字典的 'track_id' 键

        # 轉換為 NumPy array
        tlwhs = np.array(tlwhs)
        track_ids = np.array(track_ids)

        # 將數據轉為張量並放到 GPU 上
        tlwhs = torch.tensor(tlwhs, device='cuda')  # shape: (n, 4)
        track_ids = torch.tensor(track_ids, device='cuda')  # shape: (n,)

        # 計算面積 w * h
        areas = tlwhs[:, 2] * tlwhs[:, 3]  # w * h

        # 過濾掉面積過小的
        valid_mask = areas > 10

        if select_id is not None:
            valid_mask &= (track_ids == select_id)  # 確保符合 select_id

        # 過濾有效數據
        valid_tlwhs = tlwhs[valid_mask].cpu()
        valid_track_ids = track_ids[valid_mask].cpu()
        valid_areas = areas[valid_mask].cpu()

        if len(valid_tlwhs) == 0:
            return [], []

        # 找出面積最大的索引
        max_idx = torch.argmax(valid_areas).item()


        # 轉換 (x, y, w, h) -> (x1, y1, x2, y2)
        max_tlwh = valid_tlwhs[max_idx]
        max_bbox = [max_tlwh[0], max_tlwh[1], max_tlwh[0] + max_tlwh[2], max_tlwh[1] + max_tlwh[3]]
        max_id = valid_track_ids[max_idx]

        return [max_bbox], [int(max_id)]





    def correct_person_id(self, before_correctId:int, after_correctId:int):
        if self.person_df.empty:
            return
    
        if (before_correctId not in self.person_df['person_id'].unique()) or (after_correctId not in self.person_df['person_id'].unique()):
            return

        if (before_correctId in self.person_df['person_id'].unique()) and (after_correctId in self.person_df['person_id'].unique()):
            for i in range(0, max(self.processed_frames)):
                condition_1 = (self.person_df['frame_number'] == i) & (self.person_df['person_id'] == before_correctId)
                self.person_df.loc[condition_1, 'person_id'] = after_correctId

    def setPitchHandId(self,kpt_id):
        self.pitch_hand_id=kpt_id

    def setPersonId(self, person_id):
        self.person_id = person_id
        print(self.person_id)

    def setKptId(self, kpt_id):
        self.kpt_id = kpt_id
        print(self.kpt_id)

    def setDetect(self, status:bool):
        self.is_detect = status

    def updateKptBuffer(self, frame_num:int, window_length=17, polyorder=2):
        filtered_df = self.person_df[
            (self.person_df['person_id'] == self.person_id) & 
            (self.person_df['frame_number'] < frame_num)
        ]
        if filtered_df.empty:
            return None
        filtered_df = filtered_df.sort_values(by='frame_number')
        kpt_buffer = []
        for kpts in filtered_df['keypoints']:
            kpt = kpts[self.kpt_id]
            if kpt is not None and len(kpt) >= 2:
                kpt_buffer.append((kpt[0], kpt[1]))
        
        # 如果緩衝區長度大於等於窗口長度，則應用Savgol濾波器進行平滑
        if len(kpt_buffer) >= window_length:
            # 確保窗口長度為奇數且不超過緩衝區長度
            if window_length > len(kpt_buffer):
                window_length = len(kpt_buffer) if len(kpt_buffer) % 2 == 1 else len(kpt_buffer) - 1
            # 確保多項式階數小於窗口長度
            current_polyorder = min(polyorder, window_length - 1)
            
            # 分別提取x和y座標
            x = np.array([point[0] for point in kpt_buffer])
            y = np.array([point[1] for point in kpt_buffer])
            
            # 應用Savgol濾波器
            x_smooth = savgol_filter(x, window_length=window_length, polyorder=current_polyorder)
            y_smooth = savgol_filter(y, window_length=window_length, polyorder=current_polyorder)
            
            # 將平滑後的座標重新打包
            smoothed_points = list(zip(x_smooth, y_smooth))
        else:
            # 緩衝區長度不足，直接使用原始座標
            smoothed_points = kpt_buffer

        return smoothed_points
    
    def getPersonDf(self, frame_num=None, is_select=False, is_kpt=False):
        if self.person_df.empty:
            return pd.DataFrame()
        condition = pd.Series([True] * len(self.person_df))  # 初始條件設為全為 True
        if frame_num is not None:
            condition &= (self.person_df['frame_number'] == frame_num)
        
        if is_select and self.person_id is not None:
            condition &= (self.person_df['person_id'] == self.person_id)
        data = self.person_df.loc[condition].copy()
        
        if data.empty:
            return pd.DataFrame()

        if is_kpt:
            data = data['keypoints'].iloc[0]
        return data
    
    def getPrePersonDf(self):
        if self.person_df.empty:
            return None, None, None
        condition = pd.Series([True] * len(self.person_df))  # 初始條件設為全為 True
        condition &= (self.pre_person_df['person_id']==self.person_id)
        data = self.person_df.loc[condition].copy()
        
        if data.empty:
            return None, None, None
        
        data1 = data['keypoints'].iloc[0][9]
        data2 = data['keypoints'].iloc[0][10]
        data3 = data['keypoints'].iloc[0][18]
        return data1, data2, data3
    
    def getBatDF(self, frame_num=None):
        if self.bat_df.empty:
            return pd.DataFrame()
        return self.bat_df[self.bat_df['frame_number'] == frame_num]


    def setProcessedData(self, person_df:pd.DataFrame, bat_df=None):
        self.person_df = person_df.copy()
        self.person_data = self.person_df.to_dict('records')
        self.bat_df = bat_df if bat_df is not None else pd.DataFrame()
        self.processed_frames = set(self.person_df['frame_number'])
        self.processed_targets = {frame_num: None for frame_num in self.processed_frames}

    def update_person_df(self, x:float, y:float,frame_num:int, correct_kpt_idx:int):
        self.person_df.loc[(self.person_df['frame_number'] == frame_num) &
                            (self.person_df['person_id'] == self.person_id), 'keypoints'].iloc[0][correct_kpt_idx] = [x, y, 0.9, 1]

    def clearKptBuffer(self):
        self.kpt_buffer = []

    def reset(self):
        self.person_df = pd.DataFrame()
        self.pre_person_df = pd.DataFrame()
        self.person_id = None
        self.kpt_id = None
        self.fps = None
        self.person_data = []
        self.processed_frames = set()
        self.processed_targets = {}
        self.tracked_frames = {}
        self.fps_timer = FPSTimer()
        self.smooth_filter = OneEuroFilter()
        self.is_detect = False
        self.kpt_buffer = []

        self.bat_df = pd.DataFrame()
        self.has_pre_bat_df = False
        self.pre_bat_df = pd.DataFrame()
        self.bat_data = []
