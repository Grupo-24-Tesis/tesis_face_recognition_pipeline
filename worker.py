import argparse
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
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
FACES_URL_BASE = os.environ.get("FACES_URL_BASE", "").rstrip("/")
CAMERA_TZ = ZoneInfo(os.environ.get("CAMERA_TZ", "America/Bogota"))
KEEP_JOB_FILES = os.environ.get("KEEP_JOB_FILES", "0") == "1"

EVENT_TYPE = "VIDEO_UPLOAD_READY"
VISIBILITY_TIMEOUT = 300
HEARTBEAT_EVERY = 120
POLL_WAIT = 20
POST_RETRIES = 5
STAGE_TIMEOUT = int(os.environ.get("STAGE_TIMEOUT", "7200"))
POLL_ERROR_BACKOFF = 30
BACKEND_WAIT = 15
PAYLOAD_NAME = "payload.json"

RECORDING_START_RE = re.compile(r"(?<!\d)(\d{14})(?!\d)")
JOB_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")

log = logging.getLogger("worker")
stopping = threading.Event()


class JobError(Exception):
    pass


def recording_start(file_name):
    for match in RECORDING_START_RE.finditer(file_name):
        try:
            start = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
        except ValueError:
            continue
        return start.replace(tzinfo=CAMERA_TZ)
    return None


def parse_instant(value):
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
    log.info("ejecutando %s", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_DIR, check=True, stdout=sys.stderr,
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
    start = recording_start(file_name)
    if start is None:
        log.warning("no hay inicio de grabación en %r; capturedAt usa la hora "
                    "de subida", file_name)

    appearances = []
    for face in faces:
        time_s = face.get("time_s")
        if start is not None and time_s is not None:
            captured_at = start + timedelta(seconds=time_s)
        else:
            captured_at = parse_instant(created_at).astimezone(CAMERA_TZ)
        appearances.append({
            "faceId": f"{job_id}_{face['track_id']}",
            "timestamp": video_timestamp(time_s) if time_s is not None else None,
            "frameUrl": frame_url(face, os.path.join(faces_dir, face["face_file"])),
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
                log.info("se enviaron %d apariciones a %s", len(appearances), url)
                return
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise JobError(f"el backend rechazó los resultados: HTTP {exc.code}") from exc
            error = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            error = exc
        delay = min(2 ** attempt, 60)
        log.warning("POST %s falló (%s), intento %d/%d, nuevo intento en %ds",
                    url, error, attempt, POST_RETRIES, delay)
        time.sleep(delay)
    raise RuntimeError(f"no se pudieron enviar los resultados a {url} tras {POST_RETRIES} intentos")


def backend_reachable():
    try:
        with urllib.request.urlopen(f"{BACKEND_URL}/api/health", timeout=5):
            return True
    except Exception:
        return False


class VisibilityHeartbeat:

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
                log.exception("no se pudo extender la visibilidad del mensaje")

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
        raise JobError(f"el mensaje no es JSON: {exc}") from exc
    if not isinstance(msg, dict):
        raise JobError("el mensaje no es un objeto JSON")
    if msg.get("eventType") != EVENT_TYPE:
        raise JobError(f"eventType inesperado {msg.get('eventType')!r}")
    for field in ("jobId", "bucket", "objectKey"):
        if not isinstance(msg.get(field), str) or not msg[field]:
            raise JobError(f"el mensaje no tiene {field}")
    if not JOB_ID_RE.fullmatch(msg["jobId"]):
        raise JobError(f"jobId no seguro {msg['jobId']!r}")
    return msg


def process_job(s3, msg):
    job_id, bucket, key = msg["jobId"], msg["bucket"], msg["objectKey"]
    job_dir = os.path.join(JOBS_DIR, job_id)
    payload_path = os.path.join(job_dir, PAYLOAD_NAME)

    if os.path.isfile(payload_path):
        with open(payload_path) as f:
            appearances = json.load(f)
        log.info("[%s] se reutilizan los resultados de un intento anterior", job_id)
        post_results(job_id, appearances)
        if not KEEP_JOB_FILES:
            shutil.rmtree(job_dir, ignore_errors=True)
        return

    shutil.rmtree(job_dir, ignore_errors=True)
    os.makedirs(job_dir)
    keep_for_retry = False

    try:
        video_path = os.path.join(job_dir, "video" + os.path.splitext(key)[1])
        log.info("[%s] descargando s3://%s/%s", job_id, bucket, key)
        try:
            s3.download_file(bucket, key, video_path)
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NoSuchBucket"):
                raise JobError(f"video no encontrado: s3://{bucket}/{key}") from exc
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
        with open(payload_path, "w") as f:
            json.dump(appearances, f)
        try:
            post_results(job_id, appearances)
        except JobError:
            raise
        except Exception:
            keep_for_retry = True
            raise
    finally:
        if keep_for_retry:
            for name in os.listdir(job_dir):
                if name != PAYLOAD_NAME:
                    path = os.path.join(job_dir, name)
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        os.remove(path)
        elif not KEEP_JOB_FILES:
            shutil.rmtree(job_dir, ignore_errors=True)


def delete_message(sqs, receipt):
    try:
        sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=receipt)
    except Exception:
        log.exception("no se pudo borrar el mensaje de la cola")


def handle_message(sqs, s3, message):
    receipt = message["ReceiptHandle"]
    try:
        msg = parse_message(message["Body"])
    except JobError as exc:
        log.error("se descarta el mensaje %s: %s", message.get("MessageId"), exc)
        delete_message(sqs, receipt)
        return

    log.info("[%s] trabajo recibido para %s (cámara %s)",
             msg["jobId"], msg.get("originalFileName"), msg.get("cameraId"))
    started = time.monotonic()
    try:
        with VisibilityHeartbeat(sqs, receipt):
            process_job(s3, msg)
    except JobError as exc:
        log.error("[%s] se descarta el trabajo: %s", msg["jobId"], exc)
        delete_message(sqs, receipt)
        return
    except Exception:
        log.exception("[%s] el trabajo falló; se reintentará", msg["jobId"])
        return

    delete_message(sqs, receipt)
    log.info("[%s] terminado en %.0fs", msg["jobId"], time.monotonic() - started)


def poll_forever():
    import boto3

    if not QUEUE_URL:
        raise SystemExit("AWS_SQS_QUEUE_URL no está definida")
    sqs = boto3.client("sqs", region_name=AWS_REGION)
    s3 = boto3.client("s3", region_name=AWS_REGION)

    log.info("consultando %s (backend %s)", QUEUE_URL, BACKEND_URL)
    backend_down = False
    while not stopping.is_set():
        if not backend_reachable():
            if not backend_down:
                log.warning("el backend %s no responde; se espera antes de tomar "
                            "trabajos (¿está abierto el túnel SSH y corriendo el backend?)",
                            BACKEND_URL)
                backend_down = True
            stopping.wait(BACKEND_WAIT)
            continue
        if backend_down:
            log.info("el backend %s responde de nuevo; se toman trabajos", BACKEND_URL)
            backend_down = False
        try:
            resp = sqs.receive_message(
                QueueUrl=QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=POLL_WAIT,
                VisibilityTimeout=VISIBILITY_TIMEOUT,
            )
        except Exception:
            log.exception("receive_message falló; nuevo intento en %ds", POLL_ERROR_BACKOFF)
            stopping.wait(POLL_ERROR_BACKOFF)
            continue
        for message in resp.get("Messages", []):
            handle_message(sqs, s3, message)
    log.info("detenido")


def run_local(video_path, job_id):
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
    parser = argparse.ArgumentParser(description="Worker de la cola: procesa los videos del backend con el pipeline de dos etapas")
    parser.add_argument("--local", metavar="VIDEO",
                        help="procesa un video local e imprime el payload de resultados")
    parser.add_argument("--job-id", default="local-test",
                        help="id del trabajo que se usa con --local")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.local:
        run_local(args.local, args.job_id)
    else:
        signal.signal(signal.SIGTERM, lambda *_: stopping.set())
        poll_forever()


if __name__ == "__main__":
    main()
