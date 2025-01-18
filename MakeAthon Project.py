import torch
import cv2
import numpy as np
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import time

def load_yolov5_model(save_model=False, save_path="yolov5s_saved_model.pt"):
    """Lädt das vortrainierte YOLOv5-Modell und speichert es optional."""
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Modell speichern
        if save_model:
            torch.save(model.state_dict(), save_path)
            print(f"Modell erfolgreich unter '{save_path}' gespeichert.")
        
        return model
    except Exception as e:
        print(f"Fehler beim Laden des YOLOv5-Modells: {str(e)}")
        exit()

def initialize_deepsort_tracker():
    """Initialisiert den DeepSort-Tracker mit angepassten Parametern für eine stabilere Verfolgung."""
    return DeepSort(max_age=30, nn_budget=200, max_iou_distance=0.7, nms_max_overlap=1.0, max_cosine_distance=0.4)

def determine_field(x1, y1, x2, y2, fields):
    """Bestimmt, in welchem Feld sich ein Objekt befindet."""
    for field_id, (x_min, y_min, x_max, y_max) in fields.items():
        if x1 >= x_min and x2 <= x_max and y1 >= y_min and y2 <= y_max:
            return field_id
    return None

def process_video(video_path, model, tracker, conf_threshold=0.1):
    """Verarbeitet das Video, um Personen, Fahrzeuge, Flugzeuge und andere Objekte zu erkennen und zu verfolgen."""
    if not os.path.exists(video_path):
        print(f"Fehler: Das Video '{video_path}' wurde nicht gefunden.")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Fehler beim Öffnen des Videos.")
        return

    cv2.namedWindow("YOLOv5 + DeepSort Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv5 + DeepSort Tracking", 800, 600)

    # COCO-Klassen-IDs für relevante Objekte (z.B. Personen, Fahrzeuge, etc.)
    target_classes = [0]  # Klasse 0 = Person


    # Dictionary zur Zuordnung der stabilen IDs und Feldverfolgung
    stable_track_ids = {}
    next_stable_id = 1

    # Speicher für Aufenthaltszeiten der Personen in den Feldern
    track_field_times = defaultdict(lambda: {"field_id": None, "enter_time": None, "total_time": 0})

    # Verschiebung um 2 cm nach außen in Pixel (etwa 37,8 Pixel pro cm)
    outer_shift_pixels = 76

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fehler beim Lesen eines Frames oder Videoende erreicht.")
            break

        width = frame.shape[1]
        height = frame.shape[0]
        frame_resized = frame.copy()

        # Definiere Felder als 3x3-Gitter ohne Felder 2, 5 und 8
        line1_x = (width // 3) - outer_shift_pixels
        line3_x = (2 * (width // 3)) + outer_shift_pixels
        line1_y = height // 3
        line2_y = 2 * (height // 3)
        field_width = (line3_x - line1_x) // 2
        field_height = (line2_y - line1_y) // 2

        fields = {
            1: (line1_x, 0, line1_x + field_width, line1_y),
            3: (line1_x + field_width, 0, line3_x, line1_y),
            4: (line1_x, line1_y, line1_x + field_width, line2_y),
            6: (line1_x + field_width, line1_y, line3_x, line2_y),
            7: (line1_x, line2_y, line1_x + field_width, height),
            9: (line1_x + field_width, line2_y, line3_x, height)
        }

        # Zeichne die Felder
        for field_id, (x_min, y_min, x_max, y_max) in fields.items():
            cv2.rectangle(frame_resized, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(frame_resized, f"{field_id}", (x_min + 5, y_min + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        try:
            results = model(frame_resized)

            detections = []
            for *xyxy, conf, cls in results.xyxy[0]:
                if int(cls) in target_classes and conf > conf_threshold:
                    x1, y1, x2, y2 = map(int, xyxy)
                    bbox = [x1, y1, x2 - x1, y2 - y1]

                    field_id = determine_field(x1, y1, x2, y2, fields)
                    if field_id:
                        print(f"Person erkannt im Feld {field_id}")
                        cv2.putText(frame_resized, f"Field: {field_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    detections.append((bbox, float(conf), int(cls)))

        except Exception as e:  # Sicherstellen, dass ein except-Block vorhanden ist
            print(f"Fehler bei der Objekterkennung: {str(e)}")
            continue

        if len(detections) > 0:
            try:
                tracks = tracker.update_tracks(detections, frame_resized)

                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    if track.track_id not in stable_track_ids:
                        stable_track_ids[track.track_id] = next_stable_id
                        next_stable_id += 1

                    stable_id = stable_track_ids[track.track_id]
                    ltrb = track.to_ltrb()

                    # Zeichne die Bounding Box und die ID des verfolgten Objekts
                    cv2.rectangle(frame_resized, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
                    cv2.putText(frame_resized, f"ID: {stable_id}", (int(ltrb[0]), int(ltrb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Bestimme das aktuelle Feld des Tracks
                    current_field_id = determine_field(int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]), fields)
                    current_time = time.time()

                    if current_field_id != track_field_times[track.track_id]["field_id"]:
                        # Wenn das Feld gewechselt wurde, speichere die Verweildauer im alten Feld
                        if track_field_times[track.track_id]["field_id"] is not None:
                            elapsed_time = current_time - track_field_times[track.track_id]["enter_time"]
                            track_field_times[track.track_id]["total_time"] += elapsed_time
                            print(f"Track {track.track_id} hat Feld {track_field_times[track.track_id]['field_id']} verlassen. Verweildauer: {elapsed_time:.2f} Sekunden")

                        # Aktualisiere das Feld und setze die Eintrittszeit
                        track_field_times[track.track_id]["field_id"] = current_field_id
                        track_field_times[track.track_id]["enter_time"] = current_time

                    # Berechne die aktuelle Zeit im aktuellen Feld
                    if track_field_times[track.track_id]["enter_time"]:
                        time_in_field = current_time - track_field_times[track.track_id]["enter_time"]
                        cv2.putText(frame_resized, f"Time: {time_in_field:.2f}s", (int(ltrb[0]), int(ltrb[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            except Exception as e:  # Füge diesen except-Block hinzu, um Fehler zu behandeln
                print(f"Fehler beim Aktualisieren der Tracks: {str(e)}")
                continue

        # Zeige das Frame an
        try:
            cv2.imshow("YOLOv5 + DeepSort Tracking", frame_resized)
        except Exception as e:
            print(f"Fehler beim Anzeigen des Frames: {str(e)}")
            break

        # Beenden, wenn die Taste 'q' gedrückt wird
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Verwendung des Deep Learning Modells und des Trackings
    video_path = 'C:/Users/claud/Downloads/LX2254.mp4'

    # Modell laden speichern
    model = load_yolov5_model(save_model=True, save_path="C:/Users/claud/Downloads/yolov5s_saved_model.pt")
    tracker = initialize_deepsort_tracker()
    process_video(video_path, model, tracker, conf_threshold=0.1)
