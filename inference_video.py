import os
import json
from datetime import timedelta
import math
from collections import deque

import cv2
from ultralytics import YOLO

MODEL_PATH = "/home/bc/tcx/EPI_detection/runs/detect/train3/weights/best.pt"
VIDEO_PATH = "/home/bc/tcx/EPI_detection/data/teste.mp4"
VEHICLE_ROI_FOLDER = "/home/bc/tcx/EPI_detection/ROI/vehicle"
PEOPLE_ROI_FOLDER = "/home/bc/tcx/EPI_detection/ROI/people"
LOG_FILE_PATH = "/home/bc/tcx/EPI_detection/alertas.log"

model = YOLO(MODEL_PATH)

detected_objects = 0
alerts_history = []
obj_history = []

def calculate_center(box):
    """
    Calculate the center of a bounding box.
    Args:
        box: Dictionary containing x1, y1, x2, y2 coordinates.
    Returns:
        Tuple with (cx, cy) coordinates of the center.
    """
    cx = (box["x1"] + box["x2"]) / 2
    cy = (box["y1"] + box["y2"]) / 2
    return cx, cy

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    Args:
        point1: Tuple with (x, y) coordinates of the first point.
        point2: Tuple with (x, y) coordinates of the second point.
    Returns:
        Euclidean distance between the points.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def match_and_update_history(current_frame_objects, threshold=100):
    """
    Match detected objects with the history based on Euclidean distance and update the history.
    Args:
        current_frame_objects: List of objects detected in the current frame.
        label_class: The class of objects to process (e.g., "person").
        threshold: Distance threshold to consider objects as the same.
    """
    # current_frame_objects = []
    global obj_history
    global detected_objects
    
    for prev_obj in obj_history:
        prev_obj["frames_remaining"] -= 1
    
    for obj in current_frame_objects:
        current_center = calculate_center(obj["box"])
        label = obj["name"]
        
        matched = False
        closest_distance = math.inf

        # Check against history
        for prev_obj in obj_history:
            prev_center = prev_obj["center"]
            prev_label = prev_obj["name"]
            
            
            distance = euclidean_distance(current_center, prev_center)
            if label == prev_label and distance < threshold and distance < closest_distance:
                
                # obj["id"] = prev_obj["id"]
                
                # Update the historical object
                # prev_obj["center"] = current_center
                prev_obj["box"] = obj["box"]
                prev_obj["frames_remaining"] = 30
                matched = True

        if not matched:            
            # Add a new object to the history
            obj_history.append({
                "id": detected_objects,
                "center": current_center,
                "name": label,
                "box": obj["box"],
                "frames_remaining": 30,
            })

            detected_objects += 1

        # current_frame_objects.append(obj)

    # Decrement frames_remaining for unmatched historical objects
    for prev_obj in obj_history:
        if prev_obj["frames_remaining"] <= 0:
            obj_history.remove(prev_obj)

def save_roi(image, label, folder, prefix):
    x1, y1, x2, y2 = (int(label["box"][key]) for key in ["x1", "y1", "x2", "y2"])
    roi = image[y1:y2, x1:x2]
    file_name = f"{prefix}_{label['name']}.jpg"
    cv2.imwrite(os.path.join(folder, file_name), roi)

def log_alert(timestamp, message):
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(f"{timestamp}: {message}\n")

def detect_people_without_helmets(people, helmets):
    """
    Detect people without helmets and save their ROIs.
    Args:
        image: The current frame being processed.
        people: List of detected people objects.
        helmets: List of detected helmet objects.
        timestamp: Timestamp of the current frame.
    """
    
    people_without_helmets = []

    for person in people:
        x1, y1, x2, y2 = (int(person["box"][key]) for key in ["x1", "y1", "x2", "y2"])
        using_helmet = False

        # Check for overlap with helmets
        for helmet in helmets:
            hx1, hy1, hx2, hy2 = (int(helmet["box"][key]) for key in ["x1", "y1", "x2", "y2"])
            if (x1 < hx2 and x2 > hx1 and y1 < hy2 and y2 > hy1):
                using_helmet = True
                break

        if not using_helmet:
            # save_roi(image, person, PEOPLE_ROI_FOLDER, timestamp)
            # log_alert(timestamp, "Person without helmet detected")
            people_without_helmets.append(person)

    return people_without_helmets

def process_frame(frame, timestamp):
    results = model(frame)[0]
    labels = json.loads(results.to_json())
    log_alert(timestamp, f"Number of objects detected: {len(labels)}")
    
    helmets, people, vehicles = [], [], []

    # Match and update history
    match_and_update_history(labels)

    for label in obj_history:
        if label["frames_remaining"] != 30:
            continue

        if label["name"] == "vehicle":
            vehicles.append(label)
            # save_roi(frame, label, VEHICLE_ROI_FOLDER, timestamp)
            # log_alert(timestamp, "Vehicle detected")
        elif label["name"] == "helmet":
            helmets.append(label)
        elif label["name"] == "person":
            people.append(label)


    for label in vehicles:
        if label["id"] in alerts_history:
            continue

        save_roi(frame, label, VEHICLE_ROI_FOLDER, timestamp)
        log_alert(timestamp, "Vehicle detected")
        alerts_history.append(label["id"])

    people_without_helmets = detect_people_without_helmets(people, helmets)
    
    for label in people_without_helmets:
        # if label["id"] in alerts_history:
        #     continue

        save_roi(frame, label, PEOPLE_ROI_FOLDER, timestamp)
        log_alert(timestamp, "Person without helmet detected")
        alerts_history.append(label["id"])


    return results.plot()

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = str(timedelta(milliseconds=milliseconds)) #.replace(":", "-")

        processed_frame = process_frame(frame, timestamp)
        cv2.imshow('Output', processed_frame)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
