import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skeleton.detect_skeleton import PoseEstimater
import os
import sys


try:
    colors = np.round(
        np.array(plt.get_cmap('gist_rainbow').colors) * 255
    ).astype(np.uint8)[:, ::-1].tolist()
except AttributeError:  # if palette has not pre-defined colors
    colors = np.round(
        np.array(plt.get_cmap('gist_rainbow')(np.linspace(0, 1, 10))) * 255
    ).astype(np.uint8)[:, -2::-1].tolist()

if getattr(sys, 'frozen', False):
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(__file__)

class ImageDrawer():
    def __init__(self, pose_estimater: PoseEstimater=None):
        self.font_path = os.path.join(application_path, 'R-PMingLiU-TW-2.ttf')
        self.fontStyle = ImageFont.truetype(self.font_path, 20)
        self.pose_estimater = pose_estimater

        self.show_grid = False
        self.show_bbox = False
        self.show_skeleton = False
        self.show_region = False
        self.show_countdown = False
        self.region = [(100, 250), (450, 600)]


    def drawInfo(self, img:np.ndarray, frame_num:int=None, kpt_buffer:list = None, countdown_time:int = None,
                 show_bbox=None, show_skeleton=None, is_select:bool=True):
        if img is None:
            return
        image = img.copy()
        curr_person_df = self.pose_estimater.getPersonDf(frame_num = frame_num, is_select=is_select)
        if self.show_region:
            image = self.drawRegion(image)

        if countdown_time is not None:
            self.show_countdown = True

        if self.show_countdown:
            image = self.drawCountdown(image, countdown_time)

        if self.show_grid :
            image = self.drawGrid(image)

        if (self.show_bbox if show_bbox is None else show_bbox):
            image = self.drawBbox(image, curr_person_df)

        if (self.show_skeleton if show_skeleton is None else show_skeleton):
            image = self.drawPointsandSkeleton(image, curr_person_df, self.pose_estimater.joints['haple']['skeleton_links'], points_palette_samples=10)

        return image

    def drawCountdown(self, img:np.ndarray, countdown_time:int):
        height, width, _ = img.shape
        text = str(countdown_time)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 20
        color = (0, 0, 255)
        thickness = 15
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv2.putText(img, text, (text_x,text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        if countdown_time == 0:
            self.show_countdown = False
        return img

    def drawCross(self, image, x, y, length=5, color=(0, 0, 255), thickness=2):
        cv2.line(image, (x, y - length), (x, y + length), color, thickness)
        cv2.line(image, (x - length, y), (x + length, y), color, thickness)

    def drawGrid(self, image:np.ndarray):
        height, width = image.shape[:2]
        self.drawCross(image,int(width/2),int(height/2),length=20,color=(0,0,255),thickness = 3)
        vertical_interval = width // 5
        vertical_lines = [vertical_interval * i for i in range(1, 5)]
        horizontal_interval = height // 5
        horizontal_lines = [horizontal_interval * i for i in range(1, 5)]
        for x in vertical_lines:
            cv2.line(image, (x, 0), (x, height), (0, 255, 0), 2)
        for y in horizontal_lines:
            cv2.line(image, (0, y), (width, y), (0, 255, 0), 2)
        return image

    def drawBbox(self, image:np.ndarray, person_df:pd.DataFrame):
        if person_df is None or person_df.empty:
            return image
        person_ids = person_df['person_id']
        person_bbox = person_df['bbox']
        for id, bbox in zip(person_ids, person_bbox):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            color = (0,255,0)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
            image = cv2.putText(image, str(id), (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 1.5, color, 2)
        return image

    def drawRegion(self, img:np.ndarray):
        cv2.rectangle(img, self.region[0], self.region[1], (0, 255, 0), -1)
        return img

    def drawPoints(self, image, points, person_idx, color_palette='gist_rainbow', palette_samples=10, confidence_threshold=0.3):
        try:
            colors = np.round(
                np.array(plt.get_cmap(color_palette).colors) * 255
            ).astype(np.uint8)[:, ::-1].tolist()
        except AttributeError:
            colors = np.round(
                np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
            ).astype(np.uint8)[:, -2::-1].tolist()

        circle_size = max(1, min(image.shape[:2]) // 160)
        for i, pt in enumerate(points):
            unlabel = False if pt[0] != 0 and pt[1] != 0 else True
            if pt[2] > confidence_threshold and not unlabel:
                image = cv2.circle(image, (int(pt[1]), int(pt[0])), circle_size, tuple(colors[person_idx % len(colors)]), -1)

        return image

    def drawSkeleton(self, image, points, skeleton, color_palette='Set2', palette_samples='jet', person_index=0,
                    confidence_threshold=0.5):
        try:
            colors = np.round(
                np.array(plt.get_cmap(color_palette).colors) * 255
            ).astype(np.uint8)[:, ::-1].tolist()
        except AttributeError:
            colors = np.round(
                np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
            ).astype(np.uint8)[:, -2::-1].tolist()
        right_skeleton = self.pose_estimater.joints['haple']['right_points_indices']
        left_skeleton = self.pose_estimater.joints['haple']['left_points_indices']

        for i, joint in enumerate(skeleton):
            pt1, pt2 = points[joint]
            pt1_unlabel = False if pt1[0] != 0 and pt1[1] != 0 else True
            pt2_unlabel = False if pt2[0] != 0 and pt2[1] != 0 else True
            skeleton_color = (0, 165, 255)
            if joint in right_skeleton:
                skeleton_color = (240, 176, 0)
            elif joint in left_skeleton:
                skeleton_color = (0, 0, 255)
            if pt1[2] > confidence_threshold and not pt1_unlabel and pt2[2] > confidence_threshold and not pt2_unlabel:
                image = cv2.line(
                    image, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])),
                    skeleton_color , 6
                )
        return image

    def drawPointsandSkeleton(self, image, person_df, skeleton, points_color_palette='gist_rainbow', points_palette_samples=10,
                                skeleton_color_palette='Set2', skeleton_palette_samples='jet', confidence_threshold=0.3):
        if person_df is None:
            return image
        if person_df.empty:
            return image
        person_data = self.df_to_points(person_df)
        for person_id, points in person_data.items():
            image = self.drawSkeleton(image, points, skeleton, person_index=person_id)
            image = self.drawPoints(image, points, person_idx=person_id)
        return image

    def df_to_points(self, person_df):
        person_data = {}
        person_ids = person_df['person_id']
        person_kpts = person_df['keypoints']
        for id, kpts in zip(person_ids, person_kpts):
            person_data[id] = np.array(self.swapValues(kpts))
        return person_data

    def swapValues(self, kpts):
        return [[item[1], item[0], item[2]] for item in kpts]

    def setShowBbox(self, status:bool):
        self.show_bbox = status

    def setShowSkeleton(self, status:bool):
        self.show_skeleton = status

    def setShowGrid(self, status:bool):
        self.show_grid = status

    def setShowRegion(self, status:bool):
        self.show_region = status

    def reset(self):
        self.show_grid = False
        self.show_bbox = False
        self.show_skeleton = False
