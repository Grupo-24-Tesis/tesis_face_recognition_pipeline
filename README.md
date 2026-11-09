# Pipeline de reconocimiento facial (YOLO26 + RetinaFace)

Encuentra a las personas que aparecen en un video de CCTV y entrega el mejor rostro de
cada una. La etapa 1 sigue a las personas con **YOLO26m** y el tracker TrackTrack con
ReID (Ultralytics). La etapa 2 busca rostros con **RetinaFace** en los recortes de cada
persona y guarda el mejor. `worker.py` conecta el pipeline con el backend: recibe los
videos por SQS, sube los rostros a S3 y envía los resultados a la API.

Las dos etapas corren en **procesos y entornos virtuales separados**. No es una
decisión de estilo: PyTorch y TensorFlow en un mismo proceso terminan en segfault en
esta máquina.

```
                   venvs/track (solo PyTorch)         venvs/faces (solo TensorFlow)
+---------+   +-------------------------------+   +--------------------------------+
| video   |-->| tracking.py                   |-->| face_recognition.py            |
+---------+   | YOLO26m + TrackTrack + ReID   |   | RetinaFace en cada recorte     |
              | 1 fotograma cada 0.5 s        |   | descarta rostros de < 12 px    |
              | hasta 40 recortes por persona |   | guarda el mejor por persona    |
              +---------------+---------------+   +---------------+----------------+
                              v                                   v
              candidates/<id>/candidate_NNN.jpg   faces/<id>/face_of_<id>.jpg
              candidates/manifest.json            faces/results.json
```

Las etapas solo se comunican por archivos. `worker.py` lanza cada una con el intérprete
de su propio venv y nunca importa torch ni TensorFlow.

---

## Configuración vigente

Ajustada con las pruebas del repositorio `tesis-experiments` sobre tres videos (cámara 6
en dos días distintos y cámara 3):

| Qué | Valor | Dónde |
|---|---|---|
| Tracker | TrackTrack con ReID `yolo26m-reid.onnx`, `iou_weight` 0.3, `reid_weight` 0.7, `min_track_len` 3, `match_thresh` 0.7 | `custom_tracktrack.yaml` |
| Muestreo | 1 fotograma analizado cada 0.5 s | `DETECT_INTERVAL_S` en `tracking.py` |
| Recortes por persona | 40 | `MAX_CANDIDATES` en `tracking.py` |
| Borde antes de RetinaFace | 0.8 veces el lado mayor del recorte | `DETECT_PADDING` en `face_recognition.py` |
| Confianza mínima del rostro | 0.92 | `FACE_CONFIDENCE_THRESHOLD` en `face_recognition.py` |
| Tamaño mínimo del rostro | 12 px de lado menor | `MIN_FACE_PX` en `face_recognition.py` |

Cada persona seguida entrega como máximo un rostro, y la identidad es por video. No hay
reconocimiento de la misma persona entre videos: con estas cámaras (960x480, rostros de
unos 21 px de mediana) los embeddings no distinguen personas mejor que el azar. Se
puede retomar con video de mayor resolución (decisión D001 en `tesis-experiments`).

---

## Máquina objetivo

| Componente | Valor |
|---|---|
| Instancia | AWS EC2 **g4dn.xlarge** (4 vCPU, 16 GiB de RAM) |
| GPU | NVIDIA **Tesla T4**, 16 GiB de VRAM (capacidad de cómputo 7.5) |
| SO | Ubuntu 24.04 |
| Python | 3.12 |
| Driver NVIDIA | 580.178.04 (sirve cualquier driver compatible con CUDA 12.6 o superior) |
| Almacenamiento | El repo vive en el volumen montado en `/data` (118 GB). Los dos venvs ocupan unos 14 GB, más los pesos de los modelos y las salidas. |

Dos cosas que parecen erróneas pero son correctas:

- `nvidia-smi` dice `CUDA Version: 13.0`: es el **máximo del driver**, no lo que se
  usa. Los venvs traen su propio espacio de usuario CUDA 12: **12.6** en `venvs/track`
  (el de torch) y 12.9 en `venvs/faces` (el de TensorFlow).
- `nvcc` no está instalado y **no hace falta**. No se debe instalar
  `nvidia-cuda-toolkit`.

### Qué corre en la GPU

| Modelo | Dispositivo | Qué lo hace posible |
|---|---|---|
| YOLO26m (etapa 1) | GPU | torch 2.13.0+cu126 |
| Codificador ReID (etapa 1) | GPU | `onnxruntime-gpu==1.26.0`, compilado para CUDA 12 (ver Instalación) |
| RetinaFace (etapa 2) | GPU | `face_recognition.py` precarga `libcusolver` antes de importar TensorFlow |

TensorFlow 2.21 trae un error de empaquetado: su cargador de librerías no busca en
`nvidia/cusolver/lib`, aunque el paquete está instalado en el venv. Sin la precarga
imprime `Skipping registering GPU devices` y RetinaFace corre en CPU. Por eso la
precarga está en `face_recognition.py` y no en un script de arranque: así funciona con
el worker, con `run.sh` y con cualquier otro lanzador.

Con `Camera6.mp4` (136 s de video, 960x480 a 15 fps) la salida en GPU es idéntica a la
de CPU (mismos tracks, mismos recortes y mismos rostros), y el tiempo de punta a punta
baja de 137 s a 94 s. La etapa 2 pasa de 104 s a 67 s; buena parte de lo que queda es
la carga de TensorFlow y de los pesos.

---

## Estructura del repositorio

| Archivo | Contenido |
|---|---|
| `tracking.py` | Etapa 1: seguimiento de personas y recortes candidatos |
| `face_recognition.py` | Etapa 2: mejor rostro por persona |
| `worker.py` | Conexión con el backend (SQS, S3 y API) |
| `run_worker.sh` | Lanza `worker.py` con la configuración de `.env` |
| `run.sh` | Corre las dos etapas sobre un video y escribe un video anotado |
| `custom_tracktrack.yaml` | Configuración del tracker |
| `requirements-track.txt` | Dependencias congeladas de `venvs/track` (incluye el índice cu126 de torch) |
| `requirements-faces.txt` | Dependencias congeladas de `venvs/faces` |
| `requirements-worker.txt` | Dependencias de `venvs/worker` (boto3) |

No versionados: `venvs/`, `database/`, `.env`, `yolo26m.pt`, `yolo26m-reid.onnx` y los
videos de entrada.

---

## Archivos que no están en el repo

| Archivo | Cómo obtenerlo |
|---|---|
| Video de entrada (`*.mp4`, `*.mkv`) | Lo aporta el usuario. Con el worker llega desde S3. |
| `yolo26m.pt` (unos 44 MB) | Ultralytics lo descarga en la raíz del repo en la primera ejecución de `tracking.py`. |
| `yolo26m-reid.onnx` (unos 48 MB) | Ultralytics lo descarga la primera vez que el tracker corre con `with_reid: True`. También se puede bajar a mano desde [ultralytics/assets](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-reid.onnx). |
| Pesos de RetinaFace (unos 120 MB) | Se descargan en `~/.deepface/weights/` en la primera ejecución de `face_recognition.py` (o en `$DEEPFACE_HOME/.deepface/weights/` si esa variable está definida). |
| `.env` | Configuración del worker (ver más abajo). Nunca se sube. |
| `database/`, `venvs/` | Los crean la instalación y el propio pipeline. |

Las primeras ejecuciones tardan más por estas descargas y por el calentamiento de los
modelos. Esa pausa es normal.

---

## Instalación (una vez por máquina)

Desde la raíz del repo (`/data/TesisFaceRecognition/tesis_face_recognition_pipeline`).

### Opción A: desde los archivos de requisitos congelados (preferida)

```bash
python3 -m venv venvs/track
venvs/track/bin/pip install --upgrade pip
venvs/track/bin/pip install -r requirements-track.txt
venvs/track/bin/pip uninstall -y opencv-python opencv-python-headless
venvs/track/bin/pip install --no-deps opencv-python-headless==5.0.0.93

python3 -m venv venvs/faces
venvs/faces/bin/pip install --upgrade pip
venvs/faces/bin/pip install -r requirements-faces.txt
venvs/faces/bin/pip uninstall -y opencv-python opencv-python-headless
venvs/faces/bin/pip install --no-deps opencv-python-headless==5.0.0.93

python3 -m venv venvs/worker
venvs/worker/bin/pip install -r requirements-worker.txt
```

`requirements-track.txt` tiene `--extra-index-url https://download.pytorch.org/whl/cu126`
en su primera línea para que pip descargue **PyTorch para CUDA 12.6**. Esa línea no se
debe borrar: el torch de PyPI es para CUDA 13 y rompe el entorno.

`requirements-track.txt` fija `onnxruntime-gpu==1.26.0`, la última versión de PyPI
compilada para CUDA 12 que funciona con el CUDA 12.6 de torch. Desde la 1.27 la versión
de PyPI es para CUDA 13, y la 1.29.0 del índice CUDA 12 de onnxruntime necesita un CUDA
12 más nuevo (`undefined symbol: cudaLibraryGetKernel`). Con cualquiera de ellas el
codificador ReID pasa a CPU sin avisar más que en el log.

`venvs/worker` solo hace falta para consultar SQS: `worker.py --local` corre también con
el `python3` del sistema.

### Opción B: desde cero (si los archivos de requisitos están obsoletos)

`venvs/track` lleva solo la familia PyTorch y `venvs/faces` solo la familia TensorFlow
(regla 1). `onnxruntime-gpu` y `onnx` son para el codificador ReID `.onnx` en CUDA 12.

```bash
python3 -m venv venvs/track
venvs/track/bin/pip install --upgrade pip
venvs/track/bin/pip install torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu126
venvs/track/bin/pip install ultralytics lap
venvs/track/bin/pip install onnxruntime-gpu==1.26.0 onnx
venvs/track/bin/pip uninstall -y opencv-python opencv-python-headless
venvs/track/bin/pip install --no-deps opencv-python-headless

python3 -m venv venvs/faces
venvs/faces/bin/pip install --upgrade pip
venvs/faces/bin/pip install "tensorflow[and-cuda]==2.21.*"
venvs/faces/bin/pip install retina-face tf-keras deepface
venvs/faces/bin/pip uninstall -y opencv-python opencv-python-headless
venvs/faces/bin/pip install --no-deps opencv-python-headless
```

El par desinstalar/reinstalar de opencv al final de cada bloque es deliberado: varias
dependencias traen `opencv-python`, y cada venv debería tener solo el build headless
(es un servidor, no hay pantalla). Después de eso `pip check` avisa que ultralytics pide
`opencv-python`; es esperado, porque el build headless da el mismo módulo `cv2`.

**No se debe ejecutar `pip install -U ultralytics`.** La versión congelada ya soporta
TrackTrack y ReID, y una actualización puede traer torch desde PyPI (rompe la regla 2) y
reinstalar `opencv-python` (rompe la regla 3).

### Verificación

```bash
venvs/track/bin/python -c "import torch, cv2; print(torch.__version__, torch.cuda.is_available(), cv2.__version__)"
venvs/track/bin/python -c "import torch, onnxruntime as ort; s = ort.InferenceSession('yolo26m-reid.onnx', providers=['CUDAExecutionProvider']); print(ort.__version__, s.get_providers())"
venvs/faces/bin/python -c "import face_recognition, tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

Salida esperada, en el mismo orden:

```
2.13.0+cu126 True 5.0.0
1.26.0 ['CUDAExecutionProvider', 'CPUExecutionProvider']
2.21.0 [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

La segunda y la tercera se corren desde la raíz del repo (usan `yolo26m-reid.onnx` y
`face_recognition.py`). La tercera importa `face_recognition` y no solo `tensorflow`
porque es ese módulo el que precarga `libcusolver` (ver "Qué corre en la GPU").

La primera comprobación es la que importa: si deja de imprimir `2.13.0+cu126 True`, algo
cambió el stack CUDA. En ese caso no se depura el venv: se borra y se reconstruye desde
el archivo de requisitos.

---

## Uso

### Con el backend (producción)

```bash
./run_worker.sh
```

`run_worker.sh` carga `.env` si existe y lanza `venvs/worker/bin/python worker.py`, que
consulta SQS sin parar. Variables:

| Variable | Valor por defecto | Uso |
|---|---|---|
| `AWS_SQS_QUEUE_URL` | (obligatoria) | Cola donde el backend publica los trabajos |
| `AWS_REGION` | `us-east-2` | Región de SQS y S3 |
| `BACKEND_URL` | `http://localhost:8080` | API del backend |
| `FACES_S3_PREFIX` | `faces` | Carpeta del bucket donde se suben los rostros |
| `FACES_URL_BASE` | (vacía) | URL base de las imágenes para el frontend. Vacía, se usa la URL S3 del bucket, que solo funciona si esa carpeta es pública |
| `CAMERA_TZ` | `America/Bogota` | Zona horaria del reloj de las cámaras |
| `KEEP_JOB_FILES` | `0` | Con `1` se conservan los archivos de cada trabajo en `database/jobs/` |
| `STAGE_TIMEOUT` | `7200` | Segundos máximos por etapa |

Flujo por mensaje `VIDEO_UPLOAD_READY` (lo publica `SqsVideoJobPublisher` del backend):

1. Descarga `s3://<bucket>/<objectKey>` en `database/jobs/<jobId>/`.
2. Corre `tracking.py` (venvs/track) y `face_recognition.py` (venvs/faces).
3. Sube cada rostro a `s3://<bucket>/<FACES_S3_PREFIX>/<jobId>/<trackId>.jpg`.
4. Envía `POST /api/video-jobs/<jobId>/results` y borra el mensaje de la cola.

Payload que recibe el backend (`VideoResultRequest`):

```json
{
  "appearances": [
    {
      "faceId": "<jobId>_<trackId>",
      "timestamp": "00:01:23",
      "frameUrl": "https://<bucket>.s3.us-east-2.amazonaws.com/faces/<jobId>/<trackId>.jpg",
      "capturedAt": "2026-10-16T10:05:21-05:00"
    }
  ]
}
```

- `faceId`: identidad por video, una por persona seguida.
- `timestamp`: posición del rostro elegido dentro del video.
- `capturedAt`: ISO-8601 con offset, como exige `OffsetDateTime.parse`. Es el inicio de
  la grabación tomado del nombre del archivo del NVR (`Camera_chN_YYYYMMDDHHMMSS...`) más
  `timestamp`; si el nombre no tiene la hora o falta el tiempo del rostro, se usa la hora
  de subida. Siempre en
  `CAMERA_TZ`.

Manejo de fallos:

- Antes de tomar un mensaje consulta `GET /api/health`. Mientras el backend no responde,
  deja los trabajos en la cola.
- Errores transitorios (5xx, red, una etapa que falla): el mensaje queda en la cola y SQS
  lo vuelve a entregar al vencer el visibility timeout. Una dead-letter queue limita los
  reintentos.
- Errores permanentes (mensaje mal formado, video inexistente, 4xx del backend): el
  mensaje se descarta. `jobId` forma parte de rutas locales y de S3, así que solo se
  aceptan letras, números, `_` y `-`, hasta 64 caracteres.
- Si el pipeline terminó pero el POST falló, se conserva solo `payload.json` y la
  siguiente entrega lo reenvía sin repetir las etapas.
- Un heartbeat extiende la visibilidad del mensaje mientras corre el trabajo, y cada
  etapa se corta tras `STAGE_TIMEOUT`.
- Con SIGTERM termina el trabajo en curso antes de salir. Cada etapa corre en su propia
  sesión, así que la señal no la corta a mitad de camino. Con systemd se usa
  `KillMode=mixed` y un `TimeoutStopSec` mayor que la duración de un trabajo.

### Un video local, sin AWS ni backend

```bash
python3 worker.py --local /ruta/al/video.mp4 --job-id prueba
```

Corre las dos etapas igual que en producción e imprime el payload. `frameUrl` es la ruta
local del rostro. Los archivos quedan en `database/jobs/prueba/`.

### Etapas a mano

```bash
venvs/track/bin/python tracking.py --video /ruta/al/video.mp4 --candidates-dir database/candidates
venvs/faces/bin/python face_recognition.py --candidates-dir database/candidates --faces-dir database/faces
```

Conviene usar una carpeta de candidatos vacía: `tracking.py` no borra los recortes de una
ejecución anterior y la etapa 2 los mezclaría. Si hay recortes más nuevos que
`manifest.json`, la etapa 2 ignora el manifest y `frame_idx` y `time_s` quedan vacíos. `--video` es obligatorio; sin las otras
opciones se usan `database/candidates_cfc_reid` y `database/faces_cfc_reid`.

Nunca se usa `source venv/bin/activate` + `python3`: lanzar cada script con el intérprete
de su venv es lo que garantiza que corra sobre el stack correcto.

### Las dos etapas con video anotado (`run.sh`)

```bash
./run.sh /ruta/al/video.mp4 [carpeta_de_salida]
```

Corre `tracking.py` y `face_recognition.py` con la configuración vigente, igual que el
worker, y además escribe `tracked.mp4` con las cajas y los IDs del tracker para
revisarlos. La salida va a `database/run/<nombre del video>/` (`candidates/`, `faces/` y
`tracked.mp4`) si no se indica otra carpeta. Cada ejecución borra antes `candidates/` y
`faces/` de esa carpeta.

---

## Etapa 1: seguimiento (`tracking.py`)

El script lleva su propio bucle de lectura con `cv2.VideoCapture`: `cap.grab()` avanza
los fotogramas que se saltan sin decodificarlos y `cap.retrieve()` solo decodifica los
que se analizan. `persist=True` conserva el estado del tracker entre llamadas.

| Parámetro | Valor | Efecto |
|---|---|---|
| `--video` | (obligatorio) | Video de entrada |
| `CANDIDATES_DIR` / `--candidates-dir` | `database/candidates_cfc_reid` | Carpeta de salida |
| `--annotated-video` | (ninguno) | Si se indica, escribe ahí un video con las cajas y los IDs del tracker |
| `TRACKER` | `custom_tracktrack.yaml` | Configuración del tracker |
| `DETECT_INTERVAL_S` | `0.5` | Segundos de video entre fotogramas analizados, sin importar los fps |
| `MAX_CANDIDATES` | `40` | Recortes guardados por persona |
| `SCENE_CUT_CORRELATION` | `0.5` | Umbral de correlación de histogramas para detectar un corte de escena |
| `YOLO_WEIGHTS` | `yolo26m.pt` | Pesos del detector |
| `YOLO_CONF` / `YOLO_IOU` / `YOLO_IMGSZ` | `0.1` / `0.7` / `640` | Confianza, IoU de NMS y tamaño de entrada de YOLO (los valores por defecto de Ultralytics) |

Muestreo de recortes: se guarda un recorte de cada persona en cada fotograma analizado
hasta llegar a `MAX_CANDIDATES`. Entonces se descarta uno de cada dos y se duplica el
intervalo de esa persona, así los recortes siempre cubren todo el tiempo que está en
pantalla.

Cortes de escena: si dos fotogramas analizados seguidos tienen histogramas de color con
correlación menor a `SCENE_CUT_CORRELATION`, se reinicia el tracker y los IDs nuevos
siguen después de los anteriores. Es para videos editados; las cámaras fijas nunca lo
activan.

`manifest.json` guarda `video`, `fps`, `detect_every` y, por track, el fotograma de cada
recorte (`tracks`) y su caja en el fotograma (`boxes`).

### Configuración del tracker (`custom_tracktrack.yaml`)

```yaml
with_reid: True
model: yolo26m-reid.onnx
iou_weight: 0.3
reid_weight: 0.7
min_track_len: 3
match_thresh: 0.7
gmc_method: sparseOptFlow
```

- `model: yolo26m-reid.onnx` da embeddings de apariencia más discriminativos que
  `model: auto`.
- Con 0.5 s entre análisis las personas se mueven bastante, así que el peso pasa de la
  posición (HMIoU) a la apariencia (0.3 / 0.7).
- `min_track_len 3` y `match_thresh 0.7` son la configuración A. En los tres videos de
  prueba tuvo 8 cambios de identidad frente a 15 de la configuración B
  (`min_track_len 2`, `match_thresh 0.8`) (prueba TRK-04, decisión D003). Una persona
  tiene que aparecer en 3 análisis para que su track se confirme.
- El resto son los valores por defecto de TrackTrack en Ultralytics.

---

## Etapa 2: rostros (`face_recognition.py`)

Para cada persona, la etapa 2 revisa **todos** sus recortes:

1. Agrega un borde negro de `DETECT_PADDING` veces el lado mayor del recorte, porque
   RetinaFace no detecta rostros que tocan el borde (primeros planos).
2. Corre RetinaFace con umbral `FACE_CONFIDENCE_THRESHOLD`, toma el rostro de mayor
   confianza y lleva su caja al recorte original.
3. Descarta el rostro si su lado menor mide menos de `MIN_FACE_PX`.
4. Ordena los rostros válidos por área del rostro por la raíz cuadrada de la nitidez
   (varianza del laplaciano, medida sobre el rostro con margen `FACE_MARGIN`). La
   confianza de RetinaFace se satura cerca de 0.99 y casi no distingue entre recortes.
5. Guarda el ganador con cabeza y hombros en `<faces-dir>/<track_id>/face_of_<track_id>.jpg`,
   acotado al recorte de la persona.

| Parámetro | Valor | Efecto |
|---|---|---|
| `CANDIDATES_DIR` / `--candidates-dir` | `database/candidates_cfc_reid` | Recortes de la etapa 1 |
| `FACES_DIR` / `--faces-dir` | `database/faces_cfc_reid` | Carpeta de salida |
| `FACE_CONFIDENCE_THRESHOLD` | `0.92` | Confianza mínima de RetinaFace (escala de 0 a 1, **no** porcentaje) |
| `DETECT_PADDING` | `0.8` | Borde agregado antes de detectar, como fracción del lado mayor del recorte |
| `RETINA_ALLOW_UPSCALING` | `True` | Deja que RetinaFace amplíe imágenes pequeñas |
| `MIN_FACE_PX` | `12` | Lado menor mínimo del rostro, en píxeles. Los rostros más pequeños se descartan |
| `FACE_MARGIN` | `0.25` | Margen del recorte con el que se mide la nitidez |
| `OUTPUT_MARGIN_SIDE` / `TOP` / `BOTTOM` | `0.8` / `0.6` / `1.8` | Márgenes de la imagen guardada, en tamaños de rostro |
| `RANKING` | `"quality"` | `"quality"` ordena por tamaño y nitidez; `"score"` usa la confianza |

`results.json` tiene `tracks_checked` y, por rostro guardado, `track_id`, `face_file`,
`candidate`, `score`, `face_box`, `face_px`, `sharpness`, `candidates_checked`,
`frame_idx` y `time_s`. El worker usa `track_id`, `face_file` y `time_s`.

Cada línea del log muestra el recorte ganador, su confianza, el tamaño del rostro
guardado y la nitidez.

#### Tamaño mínimo de rostro: `MIN_FACE_PX = 12`

Este es el valor con el que se trabaja. Los rostros de menos de 12 px no permiten
reconocer a nadie y en las cámaras reales aparecen con frecuencia. En las pruebas
con los tres videos (cámara 6 día 1, cámara 6 otro día y cámara 3):

- No se pierde ninguna persona en ninguno de los tres videos.
- En el video del otro día de la cámara 6 desaparecen los rostros duplicados de una
  misma persona y un recorte que no era un rostro: la salida pasa de 8 a 5 rostros.
- En los otros dos videos la salida no cambia.
- Un mínimo de 16 px ya es demasiado: en la cámara 3 deja sin detectar más de la
  mitad de los rostros.

El detalle de estas pruebas está en el repositorio `tesis-experiments` (pruebas
DET-05 y E2E-03).

---

## Notas y peculiaridades

- **Sin interfaz gráfica.** Los venvs usan OpenCV headless, así que `show=True` en
  `model.track(...)` no abre ventanas. `--annotated-video` escribe el video con
  `cv2.VideoWriter` a `fps / detect_every` fps (2 fps con `DETECT_INTERVAL_S` 0.5), para
  que la reproducción vaya en tiempo real. Los IDs que dibuja son los del tracker; tras un
  corte de escena el manifest los desplaza para que no se repitan, el video no.
- **`mp4v` no se reproduce en navegadores ni en QuickTime** (VLC y mpv sí). Para verlo en
  cualquier lado se convierte con ffmpeg; `setpts` lo acelera 8 veces:
  ```bash
  ffmpeg -i tracked.mp4 -vf "setpts=PTS/8" -r 8 -c:v libx264 -pix_fmt yuv420p \
    -movflags +faststart tracked_h264.mp4
  ```
  `tracking.py` usa `mp4v` porque `avc1` suele fallar: las ruedas de OpenCV para Linux
  vienen sin x264.
- **`cv2.VideoWriter_fourcc` no existe en OpenCV 5.** Se usa `cv2.VideoWriter.fourcc(...)`.
- **Memoria de GPU:** `face_recognition.py` define `TF_FORCE_GPU_ALLOW_GROWTH=true` para
  que TensorFlow no tome toda la memoria de la T4. Esa línea se debe mantener.
- **Un segfault siempre significa** que un entorno se contaminó con el otro framework.
  No se depura: se reconstruye.

---

## Reconocimiento entre videos (no incluido)

`venvs/faces` tiene `deepface`, pero el pipeline no calcula embeddings. Se probó en
`tesis-experiments`: con la resolución de las cámaras actuales no sirve para reconocer
a la misma persona entre videos (decisión D001), y agrupar rostros dentro de un video
perdía personas (decisión D002). Si se retoma con mejor video, el script corre en
`venvs/faces`, lee los rostros de la etapa 2 y puede usar `detector_backend="skip"`,
porque los rostros ya vienen recortados.

---

## Las cuatro reglas de oro

1. **Un framework por venv y por proceso.** La familia torch (`torch`, `torchvision`,
   `ultralytics`, `triton`) nunca entra en `venvs/faces`; la familia TF (`tensorflow`,
   `tf-keras`, `retina-face`, `deepface`, `mtcnn`) nunca entra en `venvs/track`. Ojo con
   las dependencias transitivas: `pip install retina-face` **instala** tensorflow y
   `pip install ultralytics` **instala** torch.
2. **torch siempre desde el índice cu126**, nunca desde PyPI.
3. **Un solo cv2 por venv, el headless.** Hay que repetir el par desinstalar/reinstalar
   de opencv después de cualquier comando pip que toque ultralytics, retina-face o
   deepface.
4. **Lanzar con `venvs/<nombre>/bin/python`**, no con activate + `python3`.

Al volver a congelar con `pip freeze > requirements-track.txt` hay que agregar a mano la
línea `--extra-index-url` al principio: `pip freeze` no la conserva y sin ella la próxima
reconstrucción trae un torch para CUDA 13. Por la misma razón `onnxruntime-gpu` se queda
en 1.26.0 mientras torch siga en cu126.

---

## Uso responsable

Este pipeline procesa grabaciones de videovigilancia de personas identificables.
Se ejecutará únicamente sobre video que tengamos derecho legal a procesar y cumple
con las leyes de privacidad y de datos biométricos aplicables.
