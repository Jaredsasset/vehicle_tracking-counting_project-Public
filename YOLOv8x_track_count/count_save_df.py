import os
import ultralytics
import yolox
import pickle
import torch
import sys
import supervision
import torchvision
import torch
from tqdm.notebook import tqdm
from typing import Dict, Optional
import numpy as np
import cv2
import pandas as pd
import datetime
import copy
from ultralytics import YOLO
from supervision.draw.color import ColorPalette
from supervision.geometry.core import Point, Rect, Vector
# from supervision.geometry.dataclasses import Point, Rect, Vector
from supervision.video import VideoInfo
from supervision.video import get_video_frames_generator
from supervision.video import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
# from supervision.detection import Detections, BoxAnnotator
from supervision.detection.core import Detections, BoxAnnotator
from line_counter import LineCounter, LineCounterAnnotator
# from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass

model = YOLO('./yolov8x.pt')  # load model
model.fuse
HOME = os.getcwd()

@dataclass(frozen = True)
class BYTETrackerArgs:
    track_thresh : float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False
from typing import List

import numpy as np

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

CLASS_NAMES_DICT = model.model.names
CLASS_ID = [2,3,5,7]
import cv2
from IPython import display
display.clear_output()
line_start = Point(500,600)
line_end = Point(1750, 600)
# target_video_path_list = ['./datasets/대연교차로_오전_1_vehicle_counting_result.mp4',
#                      './datasets/대연교차로_오전_2_vehicle_counting_result.mp4']
source_video_path_list = ['./datasets/한양대_영상데이터2/대연교차로_오전_1.mp4',
                     './datasets/한양대_영상데이터2/대연교차로_오전_2.mp4']

# create BYTETracker instance
# create LineCounter instance
# create instance of BoxAnnotator and LineCounterAnnotator

# create VideoInfo instance

# open target video file
for flag, source_video_path in enumerate(source_video_path_list):
    count_per_class_for_every_10 = []
    # create VideoInfo instance
    video_info = VideoInfo.from_video_path(source_video_path)
    # create frame generator
    generator = get_video_frames_generator(source_video_path)
    target_video_path = source_video_path[:-4] + '_counting_result.mp4'
    line_counter = LineCounter(start=line_start, end=line_end)
    byte_tracker = BYTETracker(BYTETrackerArgs())
    box_annotator = BoxAnnotator(thickness=4, text_thickness=4, text_scale=1.5)
    line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=1.5)
    tmp = copy.deepcopy(line_counter.class_count)
    tmp['IN'] = 0
    tmp['OUT'] = 0
    tmp['TOTAL'] = 0
    frames_sofar = 0
    # duration = video_info.total_frames / video_info.fps
    with VideoSink(target_video_path, video_info) as sink:
        # loop over video frames
        
        
        for frame in tqdm(generator, total=video_info.total_frames):
            frames_sofar += 1
            current_time = frames_sofar / video_info.fps
            vid_minutes = current_time // 60
            vid_seconds = current_time % 60
            vid_time = datetime.time(minute=int(vid_minutes), second=int(vid_seconds))
            
            
            # model prediction on single frame and conversion to supervision Detections
            results = model(frame, device = 'mps')
            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int)
            )
            # filtering out detections with unwanted classes
            mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            # tracking detections
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)
            # filtering out detections without trackers
            mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            # format custom labels
            labels = [
                f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]

            # updating line counter
            line_counter.update(detections=detections)

            class_count = line_counter.class_count
            class_count['IN'] = line_counter.in_count
            class_count['OUT'] = line_counter.out_count
            class_count['TOTAL'] = line_counter.in_count + line_counter.out_count
            if ((current_time % 300 == 0)) or (frames_sofar == video_info.total_frames):
                count_per_class_for_every_10.append(
                    {f'{vid_minutes-5}m ~ {vid_time.minute}m {vid_time.second}s' :
                     {x : class_count[x] - tmp[x] for x in class_count.keys()}})
                tmp = copy.deepcopy(class_count)


            # annotate and display frame
            frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

            for num, key in enumerate(list(class_count.keys())[:4]):
                frame = cv2.putText(frame, (CLASS_NAMES_DICT[int(key)])+ ' :' + str(class_count[key]),
                                   (50,50*(num+1)), cv2.FONT_HERSHEY_COMPLEX, 1.5,(0,255,255), 2 ) 
            line_annotator.annotate(frame=frame, line_counter=line_counter)
            sink.write_frame(frame)

    df = pd.DataFrame()
    for each in count_per_class_for_every_10:
        df = pd.concat([df, pd.DataFrame(each)], axis = 1)
    df.rename(index={
    x : model.model.names[int(x)] for x in df.index[:4]
    })
    df.to_csv(f'./df_result for video_counting/video{flag+1}.csv', sep = ',')


