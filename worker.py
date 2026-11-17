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
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from presence_intervals import read_gate_settings

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
TRACK_PYTHON = os.path.join(REPO_DIR, "venvs", "track", "bin", "python")
FACES_PYTHON = os.path.join(REPO_DIR, "venvs", "faces", "bin", "python")
JOBS_DIR = os.path.join(REPO_DIR, "database", "jobs")

AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
QUEUE_URL = os.environ.get("AWS_SQS_QUEUE_URL", "")
RESULTS_QUEUE_URL = os.environ.get("AWS_SQS_RESULTS_QUEUE_URL", "")
FACES_S3_PREFIX = os.environ.get("FACES_S3_PREFIX", "faces").strip("/")
FACES_URL_BASE = os.environ.get("FACES_URL_BASE", "").rstrip("/")
CAMERA_TZ = ZoneInfo(os.environ.get("CAMERA_TZ", "America/Bogota"))
KEEP_JOB_FILES = os.environ.get("KEEP_JOB_FILES", "0") == "1"
PIPELINE_MODE = os.environ.get("PIPELINE_MODE") or "faces"
PIPELINE_MODES = ("faces", "tracks")

EVENT_TYPE = "VIDEO_UPLOAD_READY"
EVENT_STARTED = "VIDEO_PROCESSING_STARTED"
EVENT_COMPLETED = "VIDEO_PROCESSING_COMPLETED"
EVENT_FAILED = "VIDEO_PROCESSING_FAILED"
EVENT_VERSION = "1"
MAX_EVENT_BYTES = 256 * 1024
MAX_REASON_CHARS = 500
VISIBILITY_TIMEOUT = 300
HEARTBEAT_EVERY = 120
POLL_WAIT = 20
STAGE_TIMEOUT = int(os.environ.get("STAGE_TIMEOUT", "7200"))
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "3"))
POLL_ERROR_BACKOFF = 30
PAYLOAD_NAME = "payload.json"

RECORDING_START_RE = re.compile(r"(?<!\d)(\d{14})(?!\d)")
JOB_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")

log = logging.getLogger("worker")
stopping = threading.Event()


class JobError(Exception):
    pass


class StageError(RuntimeError):
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


def face_gating_enabled():
    value = os.environ.get("FACE_GATING", "").strip()
    if value not in ("", "0", "1"):
        raise SystemExit(f"FACE_GATING debe ser 0 o 1, no {value!r}")
    enabled = value != "0"
    if not enabled or PIPELINE_MODE != "faces":
        return enabled
    if os.environ.get("FACE_ADAPTIVE", "").strip() == "1":
        raise SystemExit("FACE_ADAPTIVE=1 no se puede combinar con la compuerta de presencia: "
                         "use FACE_GATING=0 o FACE_ADAPTIVE=0")
    read_gate_settings()
    return enabled


FACE_GATING = face_gating_enabled()


def run_stage(python, script, *args):
    cmd = [python, os.path.join(REPO_DIR, script), *args]
    log.info("ejecutando %s", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=REPO_DIR, check=True, stdout=sys.stderr,
                       timeout=STAGE_TIMEOUT, start_new_session=True)
    except subprocess.CalledProcessError as exc:
        raise StageError(f"{script} terminó con código {exc.returncode}") from exc
    except subprocess.TimeoutExpired as exc:
        raise StageError(f"{script} superó el límite de {STAGE_TIMEOUT} s") from exc


def check_pipeline_mode():
    if PIPELINE_MODE not in PIPELINE_MODES:
        raise SystemExit(f"PIPELINE_MODE desconocido {PIPELINE_MODE!r}: "
                         f"use {' o '.join(PIPELINE_MODES)}")


def run_pipeline(video_path, job_dir):
    faces_dir = os.path.join(job_dir, "faces")
    if PIPELINE_MODE == "faces":
        face_args = ["--video", video_path, "--faces-dir", faces_dir]
        if FACE_GATING:
            presence_path = os.path.join(job_dir, "presence.json")
            run_stage(TRACK_PYTHON, "presence_gate.py",
                      "--video", video_path, "--out", presence_path)
            face_args += ["--presence", presence_path]
        run_stage(FACES_PYTHON, "face_identities.py", *face_args)
    else:
        candidates_dir = os.path.join(job_dir, "candidates")
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


def send_event(sqs, event_type, job_id, **fields):
    body = json.dumps({
        "eventType": event_type,
        "eventVersion": EVENT_VERSION,
        "jobId": job_id,
        "sentAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        **fields,
    })
    if len(body.encode()) > MAX_EVENT_BYTES:
        raise JobError(f"los resultados ocupan {len(body.encode())} bytes y "
                       f"no caben en un mensaje de {MAX_EVENT_BYTES} bytes")
    sqs.send_message(QueueUrl=RESULTS_QUEUE_URL, MessageBody=body)
    log.info("[%s] se envió %s a la cola de resultados", job_id, event_type)


def report_started(sqs, job_id):
    try:
        send_event(sqs, EVENT_STARTED, job_id)
    except Exception:
        log.exception("[%s] no se pudo avisar que el trabajo está en curso", job_id)


def report_failure(sqs, job_id, reason):
    try:
        send_event(sqs, EVENT_FAILED, job_id, reason=reason[:MAX_REASON_CHARS])
        return True
    except Exception:
        log.exception("[%s] no se pudo avisar que el trabajo falló", job_id)
        return False


def describe_error(exc):
    if isinstance(exc, StageError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def safe_job_id(body):
    try:
        job_id = json.loads(body).get("jobId")
    except (json.JSONDecodeError, AttributeError):
        return None
    if isinstance(job_id, str) and JOB_ID_RE.fullmatch(job_id):
        return job_id
    return None


class JobHeartbeat:

    def __init__(self, sqs, receipt_handle, job_id):
        self.sqs = sqs
        self.receipt_handle = receipt_handle
        self.job_id = job_id
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def beat(self):
        try:
            self.sqs.change_message_visibility(
                QueueUrl=QUEUE_URL,
                ReceiptHandle=self.receipt_handle,
                VisibilityTimeout=VISIBILITY_TIMEOUT,
            )
        except Exception:
            log.exception("no se pudo extender la visibilidad del mensaje")
        report_started(self.sqs, self.job_id)

    def _run(self):
        while not self.done.wait(HEARTBEAT_EVERY):
            self.beat()

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


def process_job(sqs, s3, msg):
    job_id, bucket, key = msg["jobId"], msg["bucket"], msg["objectKey"]
    job_dir = os.path.join(JOBS_DIR, job_id)
    payload_path = os.path.join(job_dir, PAYLOAD_NAME)

    if os.path.isfile(payload_path):
        with open(payload_path) as f:
            appearances = json.load(f)
        log.info("[%s] se reutilizan los resultados de un intento anterior", job_id)
        send_event(sqs, EVENT_COMPLETED, job_id, appearances=appearances)
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
            send_event(sqs, EVENT_COMPLETED, job_id, appearances=appearances)
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


def discard_job_files(job_id):
    if not KEEP_JOB_FILES:
        shutil.rmtree(os.path.join(JOBS_DIR, job_id), ignore_errors=True)


def handle_message(sqs, s3, message):
    receipt = message["ReceiptHandle"]
    attempt = int(message.get("Attributes", {}).get("ApproximateReceiveCount", "1"))
    try:
        msg = parse_message(message["Body"])
    except JobError as exc:
        log.error("se descarta el mensaje %s: %s", message.get("MessageId"), exc)
        job_id = safe_job_id(message["Body"])
        if job_id is None or report_failure(sqs, job_id, f"mensaje inválido: {exc}"):
            delete_message(sqs, receipt)
        return

    job_id = msg["jobId"]
    log.info("[%s] trabajo recibido para %s (cámara %s), intento %d de %d",
             job_id, msg.get("originalFileName"), msg.get("cameraId"),
             attempt, MAX_ATTEMPTS)
    started = time.monotonic()
    report_started(sqs, job_id)
    try:
        with JobHeartbeat(sqs, receipt, job_id):
            process_job(sqs, s3, msg)
    except JobError as exc:
        log.error("[%s] se descarta el trabajo: %s", job_id, exc)
        if report_failure(sqs, job_id, str(exc)):
            delete_message(sqs, receipt)
        return
    except Exception as exc:
        if attempt < MAX_ATTEMPTS:
            log.exception("[%s] el trabajo falló; se reintentará", job_id)
            return
        log.exception("[%s] el trabajo falló %d veces; se descarta", job_id, attempt)
        reason = f"el procesamiento falló tras {attempt} intentos: {describe_error(exc)}"
        if report_failure(sqs, job_id, reason):
            discard_job_files(job_id)
            delete_message(sqs, receipt)
        return

    delete_message(sqs, receipt)
    log.info("[%s] terminado en %.0fs", job_id, time.monotonic() - started)


def poll_forever():
    check_pipeline_mode()
    import boto3

    if not QUEUE_URL:
        raise SystemExit("AWS_SQS_QUEUE_URL no está definida")
    if not RESULTS_QUEUE_URL:
        raise SystemExit("AWS_SQS_RESULTS_QUEUE_URL no está definida")
    sqs = boto3.client("sqs", region_name=AWS_REGION)
    s3 = boto3.client("s3", region_name=AWS_REGION)

    log.info("consultando %s (resultados a %s)", QUEUE_URL, RESULTS_QUEUE_URL)
    while not stopping.is_set():
        try:
            resp = sqs.receive_message(
                QueueUrl=QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=POLL_WAIT,
                VisibilityTimeout=VISIBILITY_TIMEOUT,
                MessageSystemAttributeNames=["ApproximateReceiveCount"],
            )
        except Exception:
            log.exception("receive_message falló; nuevo intento en %ds", POLL_ERROR_BACKOFF)
            stopping.wait(POLL_ERROR_BACKOFF)
            continue
        for message in resp.get("Messages", []):
            handle_message(sqs, s3, message)
    log.info("detenido")


def run_local(video_path, job_id):
    check_pipeline_mode()
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
    parser = argparse.ArgumentParser(description="Worker de la cola: procesa los videos del backend con la "
                                     "identidad por rostro (PIPELINE_MODE=faces, por defecto) "
                                     "o con el seguimiento de personas en dos etapas "
                                     "(PIPELINE_MODE=tracks)")
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
