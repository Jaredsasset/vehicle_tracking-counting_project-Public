import time
import cv2
import supervision as sv
import cv2
import datetime
import copy
from ultralytics import YOLO
from supervision.geometry.core import Point
from line_counter import LineCounterAnnotator, LineCounter
from supervision.detection.core import BoxAnnotator,Detections
from supervision.video import VideoInfo
# from supervision.draw.color import ColorPalette
model = YOLO('./yolov8x.pt')
# model.fuse()
video_path = './datasets/한양대_영상데이터2/대연교차로_오전_1.mp4'
video_info = VideoInfo.from_video_path(video_path)

count_per_class_for_every_10 = []

start = Point(500, 600)
end = Point(1700, 600)
box_annotator = BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=1.5,
    )
line_start = Point(500,600)
line_end = Point(1800, 600)
line_counter = LineCounter(start = line_start, end = line_end)
line_annotator = LineCounterAnnotator(thickness = 2, text_thickness=1, text_scale=1.5)
cap = cv2.VideoCapture(video_path)
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

frames_sofar = 0
former_count = copy.deepcopy(line_counter.class_count)
former_count['IN'] = 0
former_count['OUT'] = 0
former_count['TOTAL'] = 0
results = model.track(video_path, device= 'mps', conf = 0.5, stream = True,
                      classes = [2,3,5,7], agnostic_nms = True, verbose = False)
for result in results:
    frame = result.orig_img
    frames_sofar += 1
    current_time = frames_sofar / fps
    vid_minutes = current_time // 60
    vid_seconds = current_time % 60
    vid_time = datetime.time(minute=int(vid_minutes), second=int(vid_seconds))
    detections = Detections.from_yolov8(result)
    if result.boxes.id is not None:
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
    labels = [
        f'{model.model.names[class_id]} {confidence:0.2f}'
        for _, confidence, class_id, tracker_id in detections
    ]
    frame = box_annotator.annotate(scene=frame, detections=detections, labels = labels)
            
    line_counter.update(detections=detections)
    line_annotator.annotate(frame=frame, line_counter = line_counter)
    class_count = line_counter.class_count
    class_count['IN'] = line_counter.in_count
    class_count['OUT'] = line_counter.out_count
    class_count['TOTAL'] = line_counter.in_count + line_counter.out_count

    if ((current_time % 5 == 0) & (current_time !=0)) or (frames_sofar == video_info.total_frames):
        count_per_class_for_every_10.append(
            {f'{vid_minutes-1}m ~ {vid_time.minute}m {vid_time.second}s' : {x : class_count[x] - former_count[x] for x in class_count.keys()}}                        
            )
        former_count = copy.deepcopy(class_count)
        
    for num, key in enumerate(list(class_count.keys())[:4]):
        frame = cv2.putText(frame, (model.model.names[int(key)])+ ' :' + str(class_count[key]),
                           (50,50*(num+1)), cv2.FONT_HERSHEY_COMPLEX, 1.5,(0,255,255), 2 ) 
    cv2.imshow(' ', frame)
#         annotated_frame = results[0].plot()
#         cv2.imshow(' ', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cv2.waitKey(1)
