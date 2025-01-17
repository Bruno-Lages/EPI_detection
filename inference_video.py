import os
import json
from datetime import timedelta

import cv2
from ultralytics import YOLO

MODEL_PATH = "/home/bc/tcx/EPI_detection/runs/detect/train/weights/best.pt"
VIDEO_PATH = "/home/bc/tcx/EPI_detection/data/ch5-cut.mp4"
VEHICLE_ROI_FOLDER = "/home/bc/tcx/EPI_detection/ROI/vehicle"
PEOPLE_ROI_FOLDER = "/home/bc/tcx/EPI_detection/ROI/people"
LOG_FILE_PATH = "/home/bc/tcx/EPI_detection/alertas.log"

# Load YOLO model
model = YOLO(MODEL_PATH)

# Dictionary to store trackers and their labels
object_trackers = {}
tracker_id = 0

def save_roi(image, label, folder, prefix):
    """
    Save a specific ROI from an image to a specified folder.
    """
    x1, y1, x2, y2 = (int(label["box"][key]) for key in ["x1", "y1", "x2", "y2"])
    roi = image[y1:y2, x1:x2]
    file_name = f"{prefix}_{label['name']}.jpg"
    cv2.imwrite(os.path.join(folder, file_name), roi)

def log_alert(timestamp, message):
    """
    Log an alert with a timestamp to the specified log file.
    """
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(f"{timestamp}: {message}\n")

def initialize_tracker(frame, label):
    """
    Initialize an OpenCV tracker for a detected object.
    """
    global tracker_id
    x1, y1, x2, y2 = (int(label["box"][key]) for key in ["x1", "y1", "x2", "y2"])
    # tracker = cv2.legacy.TrackerCSRT.create()
    tracker = cv2.TrackerKCF.create()
    tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
    tracker_id += 1
    return tracker_id, tracker

def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def match_detections_to_trackers(frame, detections, trackers, iou_threshold=0.3):
    """
    Match detected bounding boxes to existing trackers using IoU.
    """
    matches = []
    unmatched_detections = list(range(len(detections)))
    unmatched_trackers = list(trackers.keys())

    for i, det in enumerate(detections):
        det_box = [int(det["box"][key]) for key in ["x1", "y1", "x2", "y2"]]
        best_iou = 0
        best_tracker_id = None

        for tracker_id in trackers:
            # Access the tracker object correctly
            tracker = trackers[tracker_id][0]  # The first element is the label dictionary
            tracker_label = trackers[tracker_id][1]  # The first element is the label dictionary
            success, bbox = tracker.update(frame)
            if success:
                tracker_box = [int(coord) for coord in bbox]
                iou = compute_iou(det_box, tracker_box)

                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_tracker_id = tracker_id

        if best_tracker_id is not None:
            matches.append((i, best_tracker_id))
            unmatched_detections.remove(i)
            unmatched_trackers.remove(best_tracker_id)

    return matches, unmatched_detections, unmatched_trackers


def update_trackers_with_detection(frame, detections):
    """
    Update trackers with new detections and reinitialize when needed.
    """
    global object_trackers

    # Step 1: Match detections with existing trackers
    matches, unmatched_detections, unmatched_trackers = match_detections_to_trackers(frame, detections, object_trackers)

    # Step 2: Update matched trackers
    for det_idx, tracker_id in matches:
        det = detections[det_idx]
        x1, y1, x2, y2 = (int(det["box"][key]) for key in ["x1", "y1", "x2", "y2"])
        object_trackers[tracker_id][0].init(frame, (x1, y1, x2 - x1, y2 - y1))
        object_trackers[tracker_id][1] = det["name"]

    # Step 3: Remove unmatched trackers
    for tracker_id in unmatched_trackers:
        del object_trackers[tracker_id]

    # Step 4: Add new trackers for unmatched detections
    for det_idx in unmatched_detections:
        det = detections[det_idx]
        if det["name"] in ["vehicle", "person"]:
            obj_id, tracker = initialize_tracker(frame, det)
            object_trackers[obj_id] = (tracker, det["name"])
            save_roi(frame, det, PEOPLE_ROI_FOLDER, "obj_id")


def process_frame(frame, timestamp):
    global object_trackers

    results = model(frame)[0]
    labels = json.loads(results.to_json())

    # Process detections and update trackers
    update_trackers_with_detection(frame, labels)

    # Draw bounding boxes from trackers
    for obj_id, (tracker, label_name) in object_trackers.items():
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            color = (0, 255, 0) if label_name == "vehicle" else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Get the current timestamp in the video
        milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = str(timedelta(milliseconds=milliseconds)).replace(":", "-")

        processed_frame = process_frame(frame, timestamp)
        cv2.imshow('Output', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

'''
1- façam a inferência no vídeo com o modelo treinado
2- identificar todas as classes presentes na imagem
3- caso exista pessoa sem epi, salva a imagem do roi da pessoa numa pasta
4- caso identifique um carro, salva uma imagem do roi do carro na pasta 
5- gerar *um* arquivo "alertas.log" contendo o timestamp do video e o alerta identificado
'''
