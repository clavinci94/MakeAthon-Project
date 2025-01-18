import os
import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# Implementierung von Deepsort für eine stabilere Objekterkennung 
def initialize_deepsort_tracker():
    return DeepSort(max_age=30,  # Hält Objekte länger in der Verfolgung, wenn sie vorübergehend verschwinden
                    nn_budget=200,  # Mehr historische Informationen speichern, um die Zuverlässigkeit der ID-Zuordnung zu erhöhen
                    max_iou_distance=0.7,  # Toleranter gegenüber kleinen Bewegungen und Größenänderungen
                    nms_max_overlap=1.0,
                    max_cosine_distance=0.4)  # Merkmalsabgleich mit einem weniger restriktiven Schwellenwert

def process_video(video_path, model, tracker):
    if not os.path.exists(video_path):
        print(f"Fehler: Das Video '{video_path}' wurde nicht gefunden.")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Fehler beim Öffnen des Videos.")
        return

    # Anzeigen der Objekterkennung im Video
    cv2.namedWindow("YOLOv5 + DeepSort Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv5 + DeepSort Tracking", 800, 600)

    # Objektklassen, die erkannt werden sollen (COCO-Klassen-IDs)
    target_classes = [0, 1, 2, 3, 4, 5, 6, 7]  # Person, Fahrrad, Auto, Motorrad, Flugzeug, Bus, Zug, LKW

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fehler beim Lesen eines Frames oder Videoende erreicht.")
            break

        # Perform object detection using YOLOv5
        results = model(frame)

        # Extract bounding boxes and class IDs
        detections = []
        for *box, conf, cls in results.xyxy[0]:
            if int(cls) in target_classes:
                detections.append((box[0].item(), box[1].item(), box[2].item(), box[3].item(), conf.item(), int(cls)))

        # Update tracker with detections
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw bounding boxes and track IDs on the frame
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (int(ltrb[0]), int(ltrb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("YOLOv5 + DeepSort Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize DeepSort tracker
tracker = initialize_deepsort_tracker()

# Process video
process_video('path_to_your_video.mp4', model, tracker)
