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

# Dictionary to store previous detections and their labels
previous_detections = {}  # Format: {object_id: {'box': [x1, y1, x2, y2], 'label': 'vehicle/person'}}
object_id = 0

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

def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    """
    print(boxA, boxB)
    xA = max(int(boxA[0]), int(boxB[0]))
    yA = max(int(boxA[1]), int(boxB[1]))
    xB = min(int(boxA[2]), int(boxB[2]))
    yB = min(int(boxA[3]), int(boxB[3]))
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def match_detections_to_previous(frame, detections, iou_threshold=0.3):
    """
    Match detected bounding boxes to the previous ones using IoU.
    """
    matches = []
    unmatched_detections = list(range(len(detections)))
    unmatched_object_ids = list(previous_detections.keys())
    
    # Track which detections have been matched, along with their IoU
    matched_detections = {}

    # This dictionary will track which previous object is being matched to
    object_detections_map = {obj_id: [] for obj_id in previous_detections}

    for i, det in enumerate(detections):

        det_box = [int(det["box"][key]) for key in ["x1", "y1", "x2", "y2"]]
        best_iou = 0
        best_object_id = None

        for obj_id in previous_detections:
            obj_label = previous_detections[obj_id]["label"]
            if obj_label != det["name"]:
                continue

            prev_box = previous_detections[obj_id]["box"]
            iou = compute_iou(det_box, prev_box)

            # Find the best match based on IoU and label
            if iou > best_iou and iou > iou_threshold and det["name"] == previous_detections[obj_id]["label"]:
                best_iou = iou
                best_object_id = obj_id

        if best_object_id is not None:
            # Track the object ID this detection is matched to
            object_detections_map[best_object_id].append(i)
            matched_detections[i] = best_iou
            print(f"Matched detection {i} to object {best_object_id} with IoU {best_iou}")

    # Now, assign the best object ID to each unmatched detection in the object_detections_map
    for obj_id, matched_indices in object_detections_map.items():
        if matched_indices:
            # If there are multiple detections matched to the same object, pick the one with the best IoU
            best_detection = max(matched_indices, key=lambda idx: matched_detections[idx])
            matches.append((best_detection, obj_id))
            unmatched_detections = [i for i in unmatched_detections if i not in matched_indices]  # Remove matched detections
            unmatched_object_ids.remove(obj_id)

    return matches, unmatched_detections, unmatched_object_ids


def update_previous_detections_with_new_info(frame, detections):
    global previous_detections

    # Step 1: Match detections with previous bounding boxes
    matches, unmatched_detections, unmatched_object_ids = match_detections_to_previous(frame, detections)

    # Step 2: Update matched detections with new bounding boxes
    for det_idx, obj_id in matches:
        det = detections[det_idx]
        previous_detections[obj_id]["box"] = [int(det["box"][key]) for key in ["x1", "y1", "x2", "y2"]]
        previous_detections[obj_id]["label"] = det["name"]

    # Step 3: Remove unmatched previous detections
    for obj_id in unmatched_object_ids:
        del previous_detections[obj_id]

    # Step 4: Add new detections as new objects
    for det_idx in unmatched_detections:
        det = detections[det_idx]
        if det["name"] in ["vehicle", "person"]:
            global object_id
            object_id += 1
            previous_detections[object_id] = {
                "box": [int(det["box"][key]) for key in ["x1", "y1", "x2", "y2"]],
                "label": det["name"]
            }
            save_roi(frame, det, PEOPLE_ROI_FOLDER, str(object_id))


def process_frame(frame, timestamp):
    global previous_detections

    results = model(frame)[0]
    labels = json.loads(results.to_json())

    # Process detections and update previous detections
    update_previous_detections_with_new_info(frame, labels)

    # Draw bounding boxes from previous detections
    for obj_id, detection in previous_detections.items():
        x1, y1, x2, y2 = detection["box"]
        label_name = detection["label"]
        color = (0, 255, 0) if label_name == "vehicle" else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
