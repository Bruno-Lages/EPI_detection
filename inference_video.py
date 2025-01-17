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

model = YOLO(MODEL_PATH)

def save_roi(image, label, folder, prefix):
    """
    Save a specific ROI from an image to a specified folder.
    Args:
        image: The input frame containing the detected objects.
        label: Dictionary containing the bounding box and object name.
        folder: Directory where the ROI image will be saved.
        prefix: Timestamp or unique identifier for the file name.
    """
    x1, y1, x2, y2 = (int(label["box"][key]) for key in ["x1", "y1", "x2", "y2"])
    roi = image[y1:y2, x1:x2]
    file_name = f"{prefix}_{label['name']}.jpg"
    cv2.imwrite(os.path.join(folder, file_name), roi)

def log_alert(timestamp, message):
    """
    Log an alert with a timestamp to the specified log file.
    Args:
        timestamp: Timestamp of the detected event.
        message: Description of the alert.
    """
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(f"{timestamp}: {message}\n")

def detect_people_without_helmets(image, people, helmets, timestamp):
    """
    Detect people without helmets and save their ROIs.
    Args:
        image: The current frame being processed.
        people: List of detected people objects.
        helmets: List of detected helmet objects.
        timestamp: Timestamp of the current frame.
    """
    for person in people:
        x1, y1, x2, y2 = (int(person["box"][key]) for key in ["x1", "y1", "x2", "y2"])
        person_roi = image[y1:y2, x1:x2]

        # Check for overlap with helmets
        for helmet in helmets:
            hx1, hy1, hx2, hy2 = (int(helmet["box"][key]) for key in ["x1", "y1", "x2", "y2"])
            if (x1 < hx2 and x2 > hx1 and y1 < hy2 and y2 > hy1):
                person_roi = None
                break

        if person_roi is not None:
            save_roi(image, person, PEOPLE_ROI_FOLDER, timestamp)
            log_alert(timestamp, "Person without helmet detected")

def process_frame(frame, timestamp):
    """
    Process a single video frame to detect objects and log alerts.
    Args:
        frame: The video frame to process.
        timestamp: Timestamp of the current frame.
    """
    results = model(frame)[0]
    labels = json.loads(results.to_json())
    
    helmets, people = [], []

    for label in labels:
        if label["name"] == "vehicle":
            save_roi(frame, label, VEHICLE_ROI_FOLDER, timestamp)
            log_alert(timestamp, "Vehicle detected")
        elif label["name"] == "helmet":
            helmets.append(label)
        elif label["name"] == "person":
            people.append(label)

    detect_people_without_helmets(frame, people, helmets, timestamp)

    return results.plot()

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
