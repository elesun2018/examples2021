import cv2
import sys

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print("cv2 version : ",cv2.__version__)

if __name__ == '__main__' :
    in_path = "videos/road1.mp4"  # road1 road2
    out_path = "output_single_road1.avi"
    # Read video
    cap = cv2.VideoCapture(in_path)
    # Exit if video not opened.
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
    # Read first frame.
    ok, frame = cap.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    w_video = cv2.VideoWriter(out_path,
                              fourcc,
                              fps,
                              (int(width), int(height)))
    ok_init = False
    ok_track = False
    box_cnt = 0
    while True:
        # Read a new frame
        ok, frame = cap.read()
        if not ok:
            w_video.release()
            break
        # Update tracker
        if ok_init :
            ok_track, bbox = tracker.update(frame)
        # Draw bounding box
        if ok_track:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            cv2.putText(frame, str(box_cnt), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure or No ROI", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        # Display result
        cv2.imshow("Tracking", frame)
        w_video.write(frame)
        # Exit if ESC pressed
        k = cv2.waitKey(10) & 0xff
        if k == 27 : break
        if k == ord("p"):
            # 选择一个区域，按s
            # Uncomment the line below to select a different bounding box
            bbox = cv2.selectROI("Tracking", frame, showCrosshair=True, fromCenter=False)
            # Set up tracker.
            tracker = OPENCV_OBJECT_TRACKERS['kcf']()
            # Initialize tracker with first frame and bounding box
            ok_init = tracker.init(frame, bbox)
            box_cnt += 1

    cap.release()
    cv2.destroyAllWindows()