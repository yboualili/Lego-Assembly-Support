from flask import Flask, Response, render_template
import cv2
from object_detection import ObjectDetection
import time
import json
import os
import helper

# set global variables
min_mae = 10000
mae_threshold = 30
mae_sufficient = 0

app = Flask(__name__)

detection_engine = ObjectDetection()
CLASS_NAME_DICT = detection_engine.get_class_name_dict()
current_stage = -1

# Bounding Boxes at each stage
root = os.path.realpath(os.path.dirname(__file__))
json_url = os.path.join(root, "static/", "bounding_boxes.json")
bboxes_json_file = open(json_url)
bboxes_json = json.load(bboxes_json_file)

def csi_pipeline(
    sensor_id=1,
    capture_width=1920,
    capture_height=1080,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    """
    create gstreamer pipeline for csi camera
    """
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def usb_camera_pipeline():
    """
    create gstreamer pipeline for usb camera
    """
    pipeline = " ! ".join(["v4l2src device=/dev/video1",
                           "video/x-raw, width=1280, height=720, framerate=10/1",
                           "videoconvert",
                           "video/x-raw, format=(string)BGR",
                           "appsink"
                           ])
    return pipeline

def camera_stream():
    """
    This function sends images to the frontend and applies the yolo model
    """
    last_frame = None
    sufficient_mae_counter = 0
    insufficient_mae_counter = 0

    global min_mae, mae_threshold, mae_sufficient, detection_engine, bboxes_json

    camera = cv2.VideoCapture(usb_camera_pipeline(), cv2.CAP_GSTREAMER)
    while True:
        # capture camera stream
        _, frame = camera.read()
        mae = 0

        if last_frame is not None:
            # compute mse
            mae = helper.compare_images(last_frame, frame)

            if min_mae is not None:
                if mae < min_mae:
                    min_mae = mae

            if mae_threshold is not None:
                mae_sufficient = 1 if mae <= mae_threshold else 0

        last_frame = frame

        if mae <= mae_threshold:
            sufficient_mae_counter += 1
        # mae has to be lower than threshold for 30 frames
        if sufficient_mae_counter >= 30:
            
            if mae > mae_threshold:
                insufficient_mae_counter += 1

            mae_sufficient = 1

            result = detection_engine.predict(frame)
            # decode results
            class_ids = result[0].cpu().boxes.cls.numpy().astype(int)
            names = [CLASS_NAME_DICT[class_ids[i]] for i in range(len(class_ids))]
            filtered_names = get_current_bounding_boxes(names)
            frame = detection_engine.plot_bboxes(frame, result, filtered_names)
            _, frame = cv2.imencode('.jpg', frame)
            frame = frame.tobytes()

            if insufficient_mae_counter >= 10:
                sufficient_mae_counter = 0
                insufficient_mae_counter = 0
                mae_sufficient = 0

            check_latest_stage(names)
            
            time.sleep(0.25)

        else:
            _, frame = cv2.imencode('.jpg', frame)
            frame = frame.tobytes()

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

def get_current_bounding_boxes(names):
    """
    only return bounding boxes for the relevant parts for the next assembly step
    """
    global bboxes_json

    current_boxes = bboxes_json[str(current_stage)]
    filtered_names = []

    for name in names:

        if name in current_boxes["parts"]:
            filtered_names.append(name)

    return filtered_names


def check_latest_stage(names):
    """
    check if the detected stage is possible based on the last stage
    """
    global current_stage

    next_stage = "stage_" + str(current_stage + 1)

    if (next_stage in names) and not current_stage in names:
        
        current_stage += 1
        

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_stream')
def video_stream():
    return Response(camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/mae_stream')
def mae_stream():
    """
    return mae to the frontend
    """

    def stream():

        global mae_sufficient

        while True:

            yield "data: %d\n\n" % mae_sufficient

            time.sleep(0.5)

    return Response(stream(), content_type='text/event-stream')


@app.route('/current_stage_stream')
def current_stage_stream():
    """
    return current stage to the frontend
    """
    def current_stage():

        global current_stage

        while True:

            yield "data: %d\n\n" % current_stage

            time.sleep(0.5)

    return Response(current_stage(), content_type='text/event-stream')


if __name__ == "__main__":
    # run app
    app.run(debug=False, host="0.0.0.0")
