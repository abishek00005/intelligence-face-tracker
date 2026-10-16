import cv2
import torch
import json
import logging
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from database import create_database, add_face, add_event
from face_recognition_module import FaceRecognizer


# 1. Load configuration
with open("config.json", "r", encoding="utf-8") as file:
    config = json.load(file)

VIDEO_PATH = config["video_path"]
MODEL_PATH = config["face_model_path"]
OUTPUT_PATH = config["output_video_path"]
CONFIDENCE = float(config["confidence_threshold"])
FRAMES_TO_SKIP = int(config["frames_to_skip"])
TRACKER = config["tracker"]

device_setting = config.get("device", "auto")

if device_setting == "auto":
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
elif device_setting == "cpu":
    DEVICE = "cpu"
else:
    DEVICE = device_setting


# 2. Create folders and database
LOG_DIR = Path("logs")
FACE_DIR = LOG_DIR / "faces"
FACE_DIR.mkdir(parents=True, exist_ok=True)

create_database()


# 3. Configure event logging
logging.basicConfig(
    filename=str(LOG_DIR / "events.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)


def log_event(event_type, track_id, image_path=None):
    logging.info(
        "event=%s | track_id=%s | image=%s",
        event_type,
        track_id,
        image_path or ""
    )


# 4. Load models
print("Using device:", DEVICE)

print("Loading YOLO face model...")
model = YOLO(MODEL_PATH)

print("Loading InsightFace recognition model...")
recognizer = FaceRecognizer()


# 5. Open input video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit(f"ERROR: Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25.0

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise SystemExit("ERROR: Could not create output video")


# 6. Track faces, save crops, and store embeddings
saved_track_ids = set()
frame_count = 0

print("Tracking faces and saving crops... Press Q to stop.")

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_count += 1
        result = None

        if (frame_count - 1) % (FRAMES_TO_SKIP + 1) == 0:
            results = model.track(
                source=frame,
                persist=True,
                tracker=TRACKER,
                device=DEVICE,
                conf=CONFIDENCE,
                verbose=False
            )

            result = results[0]

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):
                boxes = result.boxes.xyxy.cpu().tolist()
                track_ids = result.boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):

                    if track_id in saved_track_ids:
                        continue

                    x1, y1, x2, y2 = map(int, box)

                    # Keep coordinates inside the frame
                    x1 = max(0, min(x1, frame.shape[1] - 1))
                    x2 = max(0, min(x2, frame.shape[1]))
                    y1 = max(0, min(y1, frame.shape[0] - 1))
                    y2 = max(0, min(y2, frame.shape[0]))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Add padding around the detected face
                    pad_x = int((x2 - x1) * 0.35)
                    pad_y = int((y2 - y1) * 0.35)

                    crop_x1 = max(0, x1 - pad_x)
                    crop_y1 = max(0, y1 - pad_y)
                    crop_x2 = min(
                        frame.shape[1],
                        x2 + pad_x
                    )
                    crop_y2 = min(
                        frame.shape[0],
                        y2 + pad_y
                    )

                    face_crop = frame[
                        crop_y1:crop_y2,
                        crop_x1:crop_x2
                    ]

                    if face_crop.size == 0:
                        continue

                    # Save the face crop
                    file_timestamp = datetime.now().strftime(
                        "%Y%m%d_%H%M%S_%f"
                    )

                    image_path = FACE_DIR / (
                        f"track_{track_id}_{file_timestamp}.jpg"
                    )

                    if not cv2.imwrite(str(image_path), face_crop):
                        print(f"Could not save crop for track {track_id}")
                        continue

                    timestamp_text = datetime.now().isoformat(
                        timespec="seconds"
                    )

                    saved_image_path = image_path.as_posix()

                    # Generate the embedding only once
                    embedding = recognizer.get_embedding(face_crop)

                    # Save the crop and embedding in SQLite
                    add_face(
                        track_id,
                        saved_image_path,
                        timestamp_text,
                        embedding
                    )

                    # Save the detection event
                    add_event(
                        track_id,
                        "face_detected",
                        timestamp_text,
                        saved_image_path
                    )

                    log_event(
                        "face_detected",
                        track_id,
                        saved_image_path
                    )

                    if embedding is not None:
                        log_event(
                            "embedding_generated",
                            track_id,
                            saved_image_path
                        )

                        print(
                            f"Track {track_id}: embedding generated "
                            f"with shape {embedding.shape} and saved "
                            "to database"
                        )
                    else:
                        print(
                            f"Track {track_id}: crop saved, but "
                            "InsightFace could not detect a face "
                            "inside the crop. No embedding saved."
                        )

                    saved_track_ids.add(track_id)

                    print(
                        f"Saved crop and database records for "
                        f"track ID {track_id}"
                    )

        # Draw tracking boxes on the video
        if result is not None:
            annotated_frame = result.plot()
        else:
            annotated_frame = frame

        writer.write(annotated_frame)

        cv2.imshow(
            "Katomaran Face Tracking",
            annotated_frame
        )

        if frame_count % 30 == 0:
            print(
                "Frames:", frame_count,
                "| Face crops saved:", len(saved_track_ids)
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped by user.")
            break

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


# 7. Print final results
print("\nProcessing completed!")
print("Frames processed:", frame_count)
print("Face crops saved:", len(saved_track_ids))
print("Event log:", LOG_DIR / "events.log")
print("Output video:", OUTPUT_PATH)

