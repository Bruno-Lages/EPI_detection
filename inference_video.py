import argparse
import os
import json
from datetime import timedelta
import math

import cv2
from ultralytics import YOLO

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

def save_roi(image, label, folder, prefix):
    x1, y1, x2, y2 = (int(label["box"][key]) for key in ["x1", "y1", "x2", "y2"])
    
    roi = image[y1:y2, x1:x2]
    
    file_name = f"{prefix}_{label['name']}.jpg"
    
    # Ensure folder exists
    os.makedirs(folder, exist_ok=True)
    
    cv2.imwrite(os.path.join(folder, file_name), roi)

def log_alert(log_file_path, timestamp, message):
    """
    Write an alert message to the log file.
    Args:
        log_file_path: Path to the log file.
        timestamp: Timestamp of the alert.
        message: Message to log.
    """
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{timestamp}: {message}\n")

def match_and_update_history(
        current_frame_objects, 
        obj_history,
        max_occlusion_frames,
        threshold=150
    ):
    """
    Match detected objects with the history based on Euclidean distance and update the history.
    Args:
        current_frame_objects: List of objects detected in the current frame.
        obj_history: List of previous objects.
        max_occlusion_frames: Maximum number of frames an object can be occluded.
        threshold: Distance threshold to consider objects as the same.
    """
    global num_detected_objects

    for prev_obj in obj_history:
        prev_obj["frames_remaining"] -= 1
    
    for obj in current_frame_objects:
        current_center = calculate_center(obj["box"])
        label = obj["name"]
        
        matched = False
        closest_distance = math.inf

        for prev_obj in obj_history:
            prev_center = prev_obj["center"]
            prev_label = prev_obj["name"]
            
            distance = euclidean_distance(current_center, prev_center)
            if label == prev_label and distance < threshold and distance < closest_distance:
                
                # Update the object
                prev_obj["center"] = current_center
                prev_obj["box"] = obj["box"]
                prev_obj["frames_remaining"] = max_occlusion_frames
                matched = True

        if not matched:
            # Add a new object to the history
            obj_history.append({
                "id": num_detected_objects,
                "center": current_center,
                "name": label,
                "box": obj["box"],
                "frames_remaining": max_occlusion_frames,
            })

            num_detected_objects += 1


    # Decrement frames_remaining for unmatched historical objects
    for prev_obj in obj_history:
        if prev_obj["frames_remaining"] <= 0:
            obj_history.remove(prev_obj)

def detect_people_without_helmets(people, helmets):
    """
    Return a list of people without helmets.
    Args:
        people: List of detected people objects.
        helmets: List of detected helmet objects.
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
            people_without_helmets.append(person)

    return people_without_helmets

def process_frame(
        model,
        alerts_history,
        obj_history,
        log_file_path,
        vehicle_roi_folder,
        people_roi_folder,
        frame, 
        max_occlusion_frames,
        timestamp
    ):

    
    """
    Process a frame and detect vehicles and people without helmets.

    Args:
        model: YOLO model.
        alerts_history: Set of IDs of objects that have already triggered an alert.
        obj_history: List of previous objects.
        log_file_path: Path to the log file.
        vehicle_roi_folder: Folder to save vehicle ROIs.
        people_roi_folder: Folder to save people ROIs.
        frame: The current frame being processed.
        max_occlusion_frames: Maximum number of frames an object can be occluded.
        timestamp: Timestamp of the current frame.

    Returns:
        The processed frame with bounding boxes.
    """
    results = model(frame)[0]
    labels = json.loads(results.to_json())
    
    helmets, people, vehicles = [], [], []

    match_and_update_history(
        current_frame_objects=labels,
        obj_history=obj_history,
        max_occlusion_frames=max_occlusion_frames
    )

    # Extract objects from current frame for analysis
    for label in obj_history:
        if label["frames_remaining"] != max_occlusion_frames:
            continue

        if label["name"] == "vehicle":
            vehicles.append(label)
        elif label["name"] == "helmet":
            helmets.append(label)
        elif label["name"] == "person":
            people.append(label)


    for label in vehicles:
        if label["id"] in alerts_history:
            continue

        save_roi(frame, label, vehicle_roi_folder, timestamp)
        log_alert(log_file_path, timestamp, "Vehicle detected")
        alerts_history.add(label["id"])

    people_without_helmets = detect_people_without_helmets(people, helmets)
    
    for label in people_without_helmets:
        if label["id"] in alerts_history:
            continue

        save_roi(frame, label, people_roi_folder, timestamp)
        log_alert(log_file_path, timestamp, "Person without helmet detected")
        alerts_history.add(label["id"])

    return results.plot()

def main(
        model_path: str,
        video_path: str,
        log_file_path: str,
        vehicle_roi_folder: str,
        people_roi_folder: str,
        max_occlusion_frames: int
    ):

    model = YOLO(model_path)

    global num_detected_objects
    num_detected_objects = 0
    alerts_history = set()

    # List of objs wiTh {name, id, center, box, frames_remaining}
    obj_history = []

    if not os.path.isfile(video_path):
        print(f"Error: The file '{video_path}' does not exist.")
        return

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = str(timedelta(milliseconds=milliseconds))

        processed_frame = process_frame(
            model=model,
            alerts_history=alerts_history,
            obj_history=obj_history,
            log_file_path=log_file_path,
            vehicle_roi_folder=vehicle_roi_folder,
            people_roi_folder=people_roi_folder,
            frame=frame, 
            max_occlusion_frames=max_occlusion_frames,
            timestamp=timestamp
        )
        cv2.imshow('Output', processed_frame)

        if cv2.waitKey(60) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video and detect vehicles and people without helmets.")
    parser.add_argument("--model_path", type=str, default="runs/detect/train3/weights/best.pt", help="Path to the YOLO model.")
    parser.add_argument("--video_path", type=str, default="data/teste.mp4", help="Path to the video file.")
    parser.add_argument("--log_file_path", type=str, default="alertas.log", help="Path to the log file.")
    parser.add_argument("--vehicle_roi_folder", type=str, default="ROI/vehicle", help="Folder to save vehicle ROIs.")
    parser.add_argument("--people_roi_folder", type=str, default="ROI/people", help="Folder to save people ROIs.")
    parser.add_argument("--max_occlusion_frames", type=int, default=15, help="Maximum occlusion frames before object is removed.")

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        video_path=args.video_path,
        log_file_path=args.log_file_path,
        vehicle_roi_folder=args.vehicle_roi_folder,
        people_roi_folder=args.people_roi_folder,
        max_occlusion_frames=args.max_occlusion_frames
    )
