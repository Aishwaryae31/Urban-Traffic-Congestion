import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI(title="Urban Traffic CV Service", version="1.0.0")

INGEST_URL = os.getenv("HATCHABLE_INGEST_URL", "").rstrip("/")
INGEST_SECRET = os.getenv("CV_INGEST_SECRET", "")
MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8n.pt")
CONF = float(os.getenv("YOLO_CONF", "0.35"))
FRAME_STRIDE = max(1, int(os.getenv("FRAME_STRIDE", "2")))
PIXELS_PER_METER = float(os.getenv("PIXELS_PER_METER", "8.0"))
SPEED_LIMIT = float(os.getenv("SPEED_LIMIT_KMH", "50"))

# COCO vehicle classes used by YOLO.
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

model = YOLO(MODEL_PATH)
jobs: dict[str, dict[str, Any]] = {}


class Health(BaseModel):
    status: str
    model: str
    ingest_configured: bool


@app.get("/health", response_model=Health)
def health():
    return Health(
        status="ok",
        model=MODEL_PATH,
        ingest_configured=bool(INGEST_URL and INGEST_SECRET),
    )


@app.post("/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    job_id: str = Form(...),
    camera_id: str | None = Form(None),
    road_segment_id: str | None = Form(None),
):
    suffix = Path(video.filename or "traffic.mp4").suffix or ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    with open(path, "wb") as out:
        while chunk := await video.read(1024 * 1024):
            out.write(chunk)

    jobs[job_id] = {
        "job_id": job_id,
        "camera_id": camera_id,
        "road_segment_id": road_segment_id,
        "filename": video.filename,
        "status": "PROCESSING",
        "frames": 0,
        "detections": 0,
        "started_at": time.time(),
    }

    background_tasks.add_task(
        process_video,
        path,
        job_id,
        camera_id,
        road_segment_id,
    )
    return jobs[job_id]


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    return jobs.get(job_id, {"error": "job not found"})


def lane_from_center(cx: float, width: int) -> int:
    # Simple three-lane fallback. Replace with polygon calibration for a
    # particular camera when available.
    ratio = max(0.0, min(0.9999, cx / max(width, 1)))
    return int(ratio * 3) + 1


def estimate_speed(prev_center, center, fps: float) -> float:
    if prev_center is None:
        return 0.0
    dx = center[0] - prev_center[0]
    dy = center[1] - prev_center[1]
    pixels = (dx * dx + dy * dy) ** 0.5
    meters_per_frame = pixels / max(PIXELS_PER_METER, 0.01)
    return max(0.0, meters_per_frame * fps * 3.6)


def process_video(
    path: str,
    job_id: str,
    camera_id: str | None,
    road_segment_id: str | None,
):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        jobs[job_id]["status"] = "FAILED"
        jobs[job_id]["error"] = "Could not open video"
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_no = 0
    total_detections = 0
    track_last_center: dict[int, tuple[float, float]] = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1

            if frame_no % FRAME_STRIDE != 0:
                continue

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=CONF,
                verbose=False,
            )

            detections = []
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes):
                ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
                xyxy = boxes.xyxy.cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                classes = boxes.cls.int().cpu().tolist()

                for idx, cls_id in enumerate(classes):
                    if cls_id not in VEHICLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = xyxy[idx]
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    track_id = ids[idx] if idx < len(ids) else idx
                    speed = estimate_speed(
                        track_last_center.get(track_id),
                        (cx, cy),
                        fps * FRAME_STRIDE,
                    )
                    track_last_center[track_id] = (cx, cy)

                    # Reject implausible camera-motion/noise speeds.
                    speed = min(speed, SPEED_LIMIT * 2.5)

                    detections.append(
                        {
                            "track_id": track_id,
                            "vehicle_class": VEHICLE_CLASSES[cls_id],
                            "confidence": round(float(confs[idx]), 4),
                            "lane": lane_from_center(cx, frame.shape[1]),
                            "speed_kmh": round(float(speed), 2),
                        }
                    )

            total_detections += len(detections)

            if detections and INGEST_URL and INGEST_SECRET:
                payload = {
                    "job_id": job_id,
                    "camera_id": camera_id,
                    "road_segment_id": road_segment_id,
                    "frame_number": frame_no,
                    "timestamp": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                    "detections": detections,
                }
                try:
                    with httpx.Client(timeout=20) as client:
                        r = client.post(
                            INGEST_URL,
                            headers={"x-cv-secret": INGEST_SECRET},
                            json=payload,
                        )
                        r.raise_for_status()
                except Exception as exc:
                    jobs[job_id]["last_ingest_error"] = str(exc)

            jobs[job_id]["frames"] = frame_no
            jobs[job_id]["detections"] = total_detections

        jobs[job_id]["status"] = "COMPLETED"
        jobs[job_id]["completed_at"] = time.time()

    except Exception as exc:
        jobs[job_id]["status"] = "FAILED"
        jobs[job_id]["error"] = str(exc)
    finally:
        cap.release()
        try:
            os.remove(path)
        except OSError:
            pass
