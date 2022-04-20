# -*- coding:utf-8 -*-
import cv2

in_path = "videos/road1.mp4"  # road1 road2
out_path = "output_multiple_road1.avi"
cap = cv2.VideoCapture(in_path)
if not cap.isOpened():
    print("Could not open video")
    sys.exit()
# fourcc = -1
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("in_path", in_path)
print("fourcc", fourcc)
print("fps", fps)
print("width", width)
print("height", height)

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}
w_video = cv2.VideoWriter(out_path, fourcc, fps, (int(width), int(height)))
ok_init = False
ok_track = False
init_boxes = []
while cap.isOpened():
    # Read a new frame
    ok, frame = cap.read()
    if not ok:
        w_video.release()
        break
    if ok_init:
        ok_track, new_boxes = trackers.update(frame)
        # print(ok_track, new_boxes)
    if ok_track:
        for box in new_boxes:
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else: # Tracking failure
        cv2.putText(frame, "Tracking failure or No ROI", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.imshow('MultiTracker', frame)
    w_video.write(frame)
    k = cv2.waitKey(10) & 0xff
    if k == 27: break
    if k == ord("p"): # 暂停视频
        while True :
            k = cv2.waitKey(10) & 0xff
            if k == ord("a"): # 每按一次a，追加一个box
                box = cv2.selectROI("MultiTracker", frame, showCrosshair=True, fromCenter=False)
                init_boxes.append(box)
            if k == ord("e"): # 最后按e，打印所有boxes，初始化每个tracker
                print('Selected boxes {}'.format(init_boxes))
                trackers = cv2.MultiTracker_create()
                for box in init_boxes :
                    ok_init = trackers.add(OPENCV_OBJECT_TRACKERS['csrt'](), frame, box)
                init_boxes = []
                print("MultiTracker created")
                break
            if k == 27: break
