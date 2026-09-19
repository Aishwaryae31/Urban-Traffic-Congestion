# Urban Traffic Intelligence — Real CV Service

This service is the external computer-vision worker for the Hatchable Urban Traffic Intelligence app.

## Pipeline

Video → YOLOv8 → ByteTrack → vehicle IDs → lane assignment → speed estimate → Hatchable `/api/cv/ingest` → traffic DB

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Download/use the YOLO model automatically through Ultralytics on first run.

## Required configuration

Set:

- `HATCHABLE_INGEST_URL` — the Hatchable CV ingest endpoint
- `CV_INGEST_SECRET` — exactly the same secret configured in Hatchable
- `YOLO_MODEL` — default `yolov8n.pt`
- `PIXELS_PER_METER` — camera calibration value for speed estimation

## API

`GET /health`

`POST /jobs`

Multipart fields:

- `video`
- `job_id`
- `camera_id` (optional)
- `road_segment_id` (optional)

`GET /jobs/{job_id}`

The worker uses Ultralytics tracking with `bytetrack.yaml`.

### Important accuracy note

The default lane assignment is a three-lane x-coordinate fallback and the default speed calculation is a pixel-to-meter approximation. For a final academic/demo deployment, calibrate each camera with lane polygons and a pixels-per-meter value (or homography) before presenting speed as physically accurate.

## Connecting to Hatchable

1. In Hatchable project setup, create `CV_INGEST_SECRET` and copy the same value into the CV service environment.
2. Deploy this service.
3. Verify `/health` returns `ingest_configured: true`.
4. Submit a video to `POST /jobs` using an existing Hatchable `cv_jobs.id`.
5. The worker sends real detections to `/api/cv/ingest`.

The Hatchable app intentionally retains its simulated CV fallback when no external inference service is connected.
