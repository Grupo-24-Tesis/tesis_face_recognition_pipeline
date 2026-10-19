"""Queue worker: backend video job -> two-stage pipeline -> results back to the backend.

Runs in venvs/worker (boto3 only) and never imports torch or TensorFlow: each
stage runs as a subprocess on its own venv interpreter, exactly like run.sh.

    venvs/worker/bin/python worker.py                   # poll SQS forever
    venvs/worker/bin/python worker.py --local VIDEO     # run one video, print the payload

Flow per message (sent by the backend's SqsVideoJobPublisher):
  1. download s3://<bucket>/<objectKey> into database/jobs/<jobId>/
  2. tracking.py         (venvs/track) -> candidates/ + manifest.json
  3. face_recognition.py (venvs/faces) -> faces/ + results.json
  4. upload each face to s3://<bucket>/<FACES_S3_PREFIX>/<jobId>/<trackId>.jpg
  5. POST /api/video-jobs/<jobId>/results, then delete the message

A failed job leaves its message in the queue, so SQS redelivers it once the
visibility timeout expires (attach a dead-letter queue to cap the retries).
"""
import argparse
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
TRACK_PYTHON = os.path.join(REPO_DIR, "venvs", "track", "bin", "python")
FACES_PYTHON = os.path.join(REPO_DIR, "venvs", "faces", "bin", "python")
JOBS_DIR = os.path.join(REPO_DIR, "database", "jobs")

AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
QUEUE_URL = os.environ.get("AWS_SQS_QUEUE_URL", "")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8080").rstrip("/")
FACES_S3_PREFIX = os.environ.get("FACES_S3_PREFIX", "faces").strip("/")
# Base URL the frontend loads face images from. Defaults to the bucket's own
# S3 URL, which only works if <FACES_S3_PREFIX>/ is publicly readable.
FACES_URL_BASE = os.environ.get("FACES_URL_BASE", "").rstrip("/")
# Time zone of the camera clock that stamps the exported file names.
CAMERA_TZ = ZoneInfo(os.environ.get("CAMERA_TZ", "America/Bogota"))
KEEP_JOB_FILES = os.environ.get("KEEP_JOB_FILES", "0") == "1"

EVENT_TYPE = "VIDEO_UPLOAD_READY"
VISIBILITY_TIMEOUT = 300   # seconds a received message stays hidden from other consumers
HEARTBEAT_EVERY = 120      # extend that timeout this often while a job runs
POLL_WAIT = 20             # SQS long-poll seconds (the maximum SQS allows)
POST_RETRIES = 5
# A stage running longer than this is killed and the job fails, so one stuck
# video can't hold the queue forever (the heartbeat would hide it indefinitely).
STAGE_TIMEOUT = int(os.environ.get("STAGE_TIMEOUT", "7200"))
POLL_ERROR_BACKOFF = 30    # seconds to wait after an SQS/network error before polling again

# NVR exports like Camera_ch5_20260824100831_20260824102210.mp4 carry the
# recording start as YYYYMMDDHHMMSS; the first such group is the start.
RECORDING_START_RE = re.compile(r"(?<!\d)(\d{14})(?!\d)")
# Mongo ObjectIds are hex; anything else must not become a path segment.
JOB_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")

log = logging.getLogger("worker")
stopping = threading.Event()


class JobError(Exception):
    """The message can never be processed; retrying will not help."""


def recording_start(file_name):
    for match in RECORDING_START_RE.finditer(file_name):
        try:
            start = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
        except ValueError:
            continue
        return start.replace(tzinfo=CAMERA_TZ)
    return None


def parse_instant(value):
    """Parse the message's createdAt, which Jackson may write as ISO text or epoch seconds."""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def video_timestamp(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def run_stage(python, script, *args):
    cmd = [python, os.path.join(REPO_DIR, script), *args]
    log.info("running %s", " ".join(cmd))
    # Own session: a SIGTERM/Ctrl-C aimed at the worker doesn't kill the stage
    # mid-job; subprocess.run still kills it if the worker itself errors out.
    subprocess.run(cmd, cwd=REPO_DIR, check=True,
                   timeout=STAGE_TIMEOUT, start_new_session=True)


def run_pipeline(video_path, job_dir):
    candidates_dir = os.path.join(job_dir, "candidates")
    faces_dir = os.path.join(job_dir, "faces")
    run_stage(TRACK_PYTHON, "tracking.py",
              "--video", video_path, "--candidates-dir", candidates_dir)
    run_stage(FACES_PYTHON, "face_recognition.py",
              "--candidates-dir", candidates_dir, "--faces-dir", faces_dir)
    with open(os.path.join(faces_dir, "results.json")) as f:
        return faces_dir, json.load(f)["faces"]


def build_appearances(job_id, faces, faces_dir, file_name, created_at, frame_url):
    """Turn stage 2 output into the backend's VideoResultRequest.appearances."""
    start = recording_start(file_name)
    if start is None:
        # No clock in the file name: the upload time is the best guess we have.
        log.warning("no recording start in %r; capturedAt falls back to the "
                    "upload time", file_name)

    appearances = []
    for face in faces:
        time_s = face.get("time_s")
        if start is not None and time_s is not None:
            captured_at = start + timedelta(seconds=time_s)
        else:
            # The backend keeps only the wall-clock part (toLocalDateTime), so
            # express the fallback in the camera zone like file-name times.
            captured_at = parse_instant(created_at).astimezone(CAMERA_TZ)
        appearances.append({
            "faceId": f"{job_id}_{face['track_id']}",
            "timestamp": video_timestamp(time_s) if time_s is not None else None,
            "frameUrl": frame_url(face, os.path.join(faces_dir, face["face_file"])),
            # The backend parses this with OffsetDateTime, so the offset is mandatory.
            "capturedAt": captured_at.isoformat(timespec="seconds"),
        })
    return appearances


def post_results(job_id, appearances):
    url = f"{BACKEND_URL}/api/video-jobs/{job_id}/results"
    body = json.dumps({"appearances": appearances}).encode()
    for attempt in range(1, POST_RETRIES + 1):
        request = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30):
                log.info("posted %d appearances to %s", len(appearances), url)
                return
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                # e.g. 404 "Job no encontrado"; rerunning the job won't change that
                raise JobError(f"backend rejected results: HTTP {exc.code}") from exc
            error = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            error = exc
        delay = min(2 ** attempt, 60)
        log.warning("POST %s failed (%s), attempt %d/%d, retrying in %ds",
                    url, error, attempt, POST_RETRIES, delay)
        time.sleep(delay)
    raise RuntimeError(f"could not POST results to {url} after {POST_RETRIES} attempts")


class VisibilityHeartbeat:
    """Keeps a message hidden while its (possibly long) job runs."""

    def __init__(self, sqs, receipt_handle):
        self.sqs = sqs
        self.receipt_handle = receipt_handle
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.done.wait(HEARTBEAT_EVERY):
            try:
                self.sqs.change_message_visibility(
                    QueueUrl=QUEUE_URL,
                    ReceiptHandle=self.receipt_handle,
                    VisibilityTimeout=VISIBILITY_TIMEOUT,
                )
            except Exception:
                log.exception("could not extend message visibility")

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.done.set()
        self.thread.join()


def parse_message(body):
    try:
        msg = json.loads(body)
    except json.JSONDecodeError as exc:
        raise JobError(f"message is not JSON: {exc}") from exc
    if not isinstance(msg, dict):
        raise JobError("message is not a JSON object")
    if msg.get("eventType") != EVENT_TYPE:
        raise JobError(f"unexpected eventType {msg.get('eventType')!r}")
    for field in ("jobId", "bucket", "objectKey"):
        if not isinstance(msg.get(field), str) or not msg[field]:
            raise JobError(f"message has no {field}")
    if not JOB_ID_RE.fullmatch(msg["jobId"]):
        raise JobError(f"unsafe jobId {msg['jobId']!r}")
    return msg


def process_job(s3, msg):
    job_id, bucket, key = msg["jobId"], msg["bucket"], msg["objectKey"]
    job_dir = os.path.join(JOBS_DIR, job_id)
    # A redelivered job starts clean, so crops from a failed attempt can't leak in.
    shutil.rmtree(job_dir, ignore_errors=True)
    os.makedirs(job_dir)

    try:
        video_path = os.path.join(job_dir, "video" + os.path.splitext(key)[1])
        log.info("[%s] downloading s3://%s/%s", job_id, bucket, key)
        try:
            s3.download_file(bucket, key, video_path)
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NoSuchBucket"):
                raise JobError(f"video not found: s3://{bucket}/{key}") from exc
            raise

        faces_dir, faces = run_pipeline(video_path, job_dir)

        def upload_face(face, path):
            face_key = f"{FACES_S3_PREFIX}/{job_id}/{face['track_id']}.jpg"
            s3.upload_file(path, bucket, face_key,
                           ExtraArgs={"ContentType": "image/jpeg"})
            if FACES_URL_BASE:
                return f"{FACES_URL_BASE}/{face_key}"
            return f"https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{face_key}"

        appearances = build_appearances(
            job_id, faces, faces_dir,
            msg.get("originalFileName") or os.path.basename(key),
            msg.get("createdAt"), upload_face,
        )
        post_results(job_id, appearances)
    finally:
        if not KEEP_JOB_FILES:
            shutil.rmtree(job_dir, ignore_errors=True)


def delete_message(sqs, receipt):
    try:
        sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=receipt)
    except Exception:
        # The job itself is done; at worst SQS redelivers it once more.
        log.exception("could not delete message from the queue")


def handle_message(sqs, s3, message):
    receipt = message["ReceiptHandle"]
    try:
        msg = parse_message(message["Body"])
    except JobError as exc:
        # Retrying a malformed message would fail forever; drop it loudly.
        log.error("dropping message %s: %s", message.get("MessageId"), exc)
        delete_message(sqs, receipt)
        return

    log.info("[%s] received job for %s (camera %s)",
             msg["jobId"], msg.get("originalFileName"), msg.get("cameraId"))
    started = time.monotonic()
    try:
        with VisibilityHeartbeat(sqs, receipt):
            process_job(s3, msg)
    except JobError as exc:
        log.error("[%s] dropping job: %s", msg["jobId"], exc)
        delete_message(sqs, receipt)
        return
    except Exception:
        # Leave the message alone: SQS redelivers it after the visibility timeout.
        log.exception("[%s] job failed; it will be retried", msg["jobId"])
        return

    delete_message(sqs, receipt)
    log.info("[%s] done in %.0fs", msg["jobId"], time.monotonic() - started)


def poll_forever():
    import boto3

    if not QUEUE_URL:
        raise SystemExit("AWS_SQS_QUEUE_URL is not set")
    sqs = boto3.client("sqs", region_name=AWS_REGION)
    s3 = boto3.client("s3", region_name=AWS_REGION)

    log.info("polling %s (backend %s)", QUEUE_URL, BACKEND_URL)
    while not stopping.is_set():
        try:
            resp = sqs.receive_message(
                QueueUrl=QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=POLL_WAIT,
                VisibilityTimeout=VISIBILITY_TIMEOUT,
            )
        except Exception:
            # Network blip, throttling or expired credentials: keep the worker alive.
            log.exception("receive_message failed; retrying in %ds", POLL_ERROR_BACKOFF)
            stopping.wait(POLL_ERROR_BACKOFF)
            continue
        for message in resp.get("Messages", []):
            handle_message(sqs, s3, message)
    log.info("stopped")


def run_local(video_path, job_id):
    """Run one local video end to end without AWS or the backend; print the payload."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    shutil.rmtree(job_dir, ignore_errors=True)
    os.makedirs(job_dir)
    faces_dir, faces = run_pipeline(os.path.abspath(video_path), job_dir)
    appearances = build_appearances(
        job_id, faces, faces_dir, os.path.basename(video_path),
        datetime.now(timezone.utc).isoformat(), lambda face, path: path,
    )
    print(json.dumps({"appearances": appearances}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--local", metavar="VIDEO",
                        help="process one local video and print the results payload")
    parser.add_argument("--job-id", default="local-test",
                        help="job id used with --local")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.local:
        run_local(args.local, args.job_id)
    else:
        # Finish the current job on SIGTERM instead of dying mid-upload. Under
        # systemd use KillMode=mixed and a TimeoutStopSec longer than a job.
        signal.signal(signal.SIGTERM, lambda *_: stopping.set())
        poll_forever()


if __name__ == "__main__":
    main()
