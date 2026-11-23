# Pipeline de identidad por rostro (YOLO26m + RetinaFace + GhostFaceNet)

Encuentra los rostros que aparecen en un video de CCTV, los agrupa en identidades dentro
del video, entrega el mejor rostro de cada identidad y un embedding por identidad para
buscarla en otros videos. Hay un solo flujo:

1. `worker.py` toma un trabajo de la cola de SQS y descarga el video desde S3.
2. `presence_gate.py` (`venvs/track`) revisa el video con **YOLO26m** unas pocas veces
   por segundo y escribe `presence.json` con los tramos donde hay alguien en cámara. Se
   omite con `FACE_GATING=0`.
3. `face_identities.py` (`venvs/faces`) corre **RetinaFace** sobre el fotograma completo
   (solo en los tramos activos, si hay compuerta), alinea cada rostro con sus puntos
   faciales, calcula su embedding con **GhostFaceNet**, agrupa los rostros en
   identidades, guarda el mejor rostro de cada una y escribe `embeddings.json` con el
   promedio unitario de los embeddings de cada identidad.
4. `worker.py` sube los rostros a S3 y publica los resultados en otra cola de SQS, con
   un embedding por detección y en partes numeradas.

Las dos pasadas corren en **procesos y entornos virtuales separados**. No es una
decisión de estilo: PyTorch y TensorFlow en un mismo proceso terminan en segfault en
esta máquina.

```
                   venvs/track (solo PyTorch)         venvs/faces (solo TensorFlow)
+---------+   +-------------------------------+   +--------------------------------+
| video   |-->| presence_gate.py              |-->| face_identities.py             |
+---------+   | YOLO26m, solo clase persona   |   | RetinaFace en el fotograma     |
              | 2 revisiones por segundo      |   | 1 fotograma cada 0.1 s         |
              | tramos con alguien en cámara  |   | GhostFaceNet y agrupación      |
              +---------------+---------------+   +---------------+----------------+
                              v                                   v
              presence.json                       faces/<id>/face_of_<id>.jpg
                                                  faces/results.json
                                                  faces/embeddings.json
```

Las pasadas solo se comunican por archivos. `worker.py` lanza cada una con el intérprete
de su propio venv y nunca importa torch ni TensorFlow.

---

## Configuración vigente

Son los valores por defecto del código. Los de la identidad por rostro son valores
congelados (ver "Identidad por rostro") y los de la compuerta salen del barrido descrito
en "Resultados medidos de la compuerta".

| Qué | Valor | Dónde |
| --- | --- | --- |
| Compuerta de presencia | Activa (`FACE_GATING` 1) | `worker.py` y `run.sh` |
| Revisiones de la compuerta | 2 por segundo de video, YOLO26m a 640 px, confianza mínima 0.25, tramo desde 1 s antes hasta 2 s después de cada detección | `GATE_*` en `presence_intervals.py` |
| Muestreo de rostros | Intervalo fijo, 1 fotograma analizado cada 0.1 s (paso `max(1, round(fps * 0.1))`) | `FACE_INTERVAL_S` en `face_sampling.py` (`FACE_ADAPTIVE` 0) |
| Detector de rostros | RetinaFace sobre el fotograma completo, umbral 0.9, con escalado | `RETINA_THRESHOLD` y `RETINA_ALLOW_UPSCALING` en `face_identities.py` |
| Alineación | Cinco puntos faciales a la plantilla de ArcFace de 112x112 px | `face_identities.py` |
| Embedding | GhostFaceNet de DeepFace, 512 dimensiones, norma 1, en bloques de 64 rostros | `EMBEDDING_MODEL` en `face_identities.py`, `EMBED_CHUNK` en `face_embedding.py` |
| Confianza mínima para las identidades | 0.99 | `MIN_SCORE` en `identity_linking.py` |
| Unión en tracklets | IoU de al menos 0.3 y similitud de al menos 0.2, hasta 2 fotogramas analizados de salto | `LINK_IOU`, `LINK_SIMILARITY` y `MAX_SKIP` en `identity_linking.py` |
| Fusión de tracklets | Enlace promedio hasta similitud 0.35 | `MERGE_THRESHOLD` en `identity_linking.py` |
| Gemelo espejo | Similitud de al menos 0.6 en un mismo fotograma | `TWIN_THRESHOLD` en `identity_linking.py` |
| Rostros mínimos por identidad | 8 | `FACE_MIN_FACES` en `identity_linking.py` |
| Mejor rostro | Área de la caja del rostro por la raíz cuadrada de la nitidez (varianza del laplaciano), con la confianza de RetinaFace como desempate; se guarda con cabeza y hombros (márgenes 0.8 a los lados, 0.6 arriba y 1.8 abajo, en tamaños de rostro) | `rank_key`, `crop_face` y `OUTPUT_MARGIN_*`, que importa `face_identities.py` |
| Embedding por identidad | Promedio de los embeddings de sus rostros, llevado a norma 1, `float32` little-endian en base64 | `face_embedding.py` |
| Tamaño de cada parte del resultado | Hasta 245760 bytes (240 KiB) | `RESULT_PART_BYTES` en `worker.py` |

La identidad es por video: cada identidad entrega un rostro y un embedding, y la
búsqueda entre videos la hace el backend (ver "Reconocimiento entre videos").

---

## Máquina objetivo

| Componente | Valor |
| --- | --- |
| Instancia | AWS EC2 **g4dn.xlarge** (4 vCPU, 16 GiB de RAM) |
| GPU | NVIDIA **Tesla T4**, 16 GiB de VRAM (capacidad de cómputo 7.5) |
| SO | Ubuntu 24.04 o 26.04 |
| Python | 3.12.3. Ubuntu 26.04 trae 3.14 y no sirve: ver "Ubuntu 26.04" en Instalación |
| Driver NVIDIA | 580.178.04 (sirve cualquier driver compatible con CUDA 12.6 o superior) |
| Almacenamiento | El repo vive en `/data/TesisFaceRecognition`, en un volumen EBS. El disco NVMe de instancia de la g4dn no sirve: se borra al detener la instancia. Los venvs ocupan unos 14 GB, más los pesos de los modelos y las salidas. |

Dos cosas que parecen erróneas pero son correctas:

- `nvidia-smi` dice `CUDA Version: 13.0`: es el **máximo del driver**, no lo que se
  usa. Los venvs traen su propio espacio de usuario CUDA 12: **12.6** en `venvs/track`
  (el de torch) y 12.9 en `venvs/faces` (el de TensorFlow).
- `nvcc` no está instalado y **no hace falta**. No se debe instalar
  `nvidia-cuda-toolkit`.

### Qué corre en la GPU

| Modelo | Dispositivo | Qué lo hace posible |
| --- | --- | --- |
| YOLO26m (`presence_gate.py`) | GPU | torch 2.13.0+cu126 |
| RetinaFace y GhostFaceNet (`face_identities.py`) | GPU | `face_identities.py` precarga `libcusolver` antes de importar TensorFlow |

TensorFlow 2.21 trae un error de empaquetado: su cargador de librerías no busca en
`nvidia/cusolver/lib`, aunque el paquete está instalado en el venv. Sin la precarga
imprime `Skipping registering GPU devices` y RetinaFace corre en CPU. Por eso la
precarga está al principio de `face_identities.py` y no en un script de arranque: así
funciona con el worker, con `run.sh` y con cualquier otro lanzador.

---

## Estructura del repositorio

| Archivo | Contenido |
| --- | --- |
| `worker.py` | Conexión con el backend (SQS y S3): corre la compuerta y la identidad por rostro sobre cada video, sube los rostros y publica los resultados |
| `run_worker.sh` | Lanza `worker.py` con la configuración de `.env` |
| `run.sh` | Corre el flujo sobre un video local y escribe un video anotado |
| `presence_gate.py` | Compuerta de presencia: revisa el video con YOLO26m y escribe `presence.json` con los tramos donde hay alguien en cámara |
| `presence_intervals.py` | Cálculo de los tramos activos y del registro de cambios, lectura de `presence.json`, las decisiones de la compuerta y la validación de `GATE_*` (solo biblioteca estándar, lo importan todos los venvs, el worker incluido) |
| `face_identities.py` | Detecta, alinea y agrupa los rostros del video, guarda el mejor de cada identidad y escribe `embeddings.json` |
| `face_embedding.py` | Preprocesamiento de GhostFaceNet, embeddings en bloque y embedding medio por identidad (numpy y OpenCV) |
| `face_sampling.py` | Elección del siguiente fotograma a analizar (intervalo fijo, o adaptativo con `FACE_ADAPTIVE=1`, y tramos de la compuerta; sin dependencias externas) |
| `identity_linking.py` | Agrupación de rostros en identidades (solo numpy) |
| `env_settings.py` | Lectura y validación de las variables de entorno numéricas y de 0 o 1 (solo biblioteca estándar) |
| `face_recognition.py` | No forma parte del flujo como script. `face_identities.py` importa de él `OUTPUT_MARGIN_SIDE`, `OUTPUT_MARGIN_TOP`, `OUTPUT_MARGIN_BOTTOM`, `RESULTS_NAME`, `crop_face`, `rank_key` y `sharpness` |
| `tracking.py` y `custom_tracktrack.yaml` | No los usan el worker ni `run.sh`. El banco de pruebas (`tesis-experiments`) los llama para reproducir experimentos anteriores |
| `requirements-track.txt` | Dependencias congeladas de `venvs/track` (incluye el índice cu126 de torch) |
| `requirements-faces.txt` | Dependencias congeladas de `venvs/faces` |
| `requirements-worker.txt` | Dependencias de `venvs/worker` (boto3) |

No versionados: `venvs/`, `database/`, `.env`, `yolo26m.pt`, `yolo26m-reid.onnx` y los
videos de entrada.

---

## Archivos que no están en el repo

| Archivo | Cómo obtenerlo |
| --- | --- |
| Video de entrada (`*.mp4`, `*.mkv`) | Lo aporta el usuario. Con el worker llega desde S3. |
| `yolo26m.pt` (unos 44 MB) | Ultralytics lo descarga en la raíz del repo en la primera ejecución de `presence_gate.py`, que lo busca junto al script. |
| `yolo26m-reid.onnx` (unos 48 MB) | No lo usa el flujo: es el codificador ReID que usa el banco de pruebas. Ultralytics lo descarga la primera vez que el banco corre el tracker con ReID; también se puede bajar a mano desde [ultralytics/assets](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-reid.onnx). |
| Pesos de RetinaFace (unos 120 MB) | Se descargan en `~/.deepface/weights/` en la primera ejecución de `face_identities.py` (o en `$DEEPFACE_HOME/.deepface/weights/` si esa variable está definida). |
| Pesos de GhostFaceNet (unos 17 MB) | DeepFace los descarga en la misma carpeta (`ghostfacenet_v1.h5`) en la primera ejecución de `face_identities.py`. |
| `.env` | Configuración del worker (ver más abajo). Nunca se sube. |
| `database/`, `venvs/` | Los crean la instalación y el propio pipeline. |

Las primeras ejecuciones tardan más por estas descargas y por el calentamiento de los
modelos. Esa pausa es normal.

---

## Instalación (una vez por máquina)

Desde la raíz del repo (`/data/TesisFaceRecognition/tesis_face_recognition_pipeline`).

### Ubuntu 26.04: driver y Python 3.12

Ubuntu 26.04 no trae el driver de NVIDIA ni Python 3.12. El driver usa los módulos ya
compilados para el kernel de AWS, y Python 3.12.3 se instala con uv en `/opt/python`:

```bash
sudo apt-get install -y linux-modules-nvidia-580-server-aws nvidia-headless-no-dkms-580-server nvidia-utils-580-server
sudo reboot

curl -LsSf https://astral.sh/uv/install.sh | sh
sudo mkdir -p /opt/python && sudo chown ubuntu:ubuntu /opt/python
UV_PYTHON_INSTALL_DIR=/opt/python ~/.local/bin/uv python install 3.12.3
printf '#!/bin/sh\nexec /opt/python/cpython-3.12.3-linux-x86_64-gnu/bin/python3.12 "$@"\n' | sudo tee /usr/local/bin/python3.12
sudo chmod 755 /usr/local/bin/python3.12
```

`/usr/local/bin/python3.12` tiene que ser ese script y no un enlace simbólico: con un
enlace, los venvs que se crean desde él no encuentran la biblioteca estándar. Después
`nvidia-smi` debe mostrar la T4 y el resto de la instalación es igual que en 24.04.

Si los venvs se copian desde otra máquina en lugar de crearse, en cada uno `bin/python3`
debe apuntar a `/opt/python/cpython-3.12.3-linux-x86_64-gnu/bin/python3.12`, y `home` de
`pyvenv.cfg` a la carpeta `bin` de esa instalación.

### Opción A: desde los archivos de requisitos congelados (preferida)

```bash
python3.12 -m venv venvs/track
venvs/track/bin/pip install --upgrade pip
venvs/track/bin/pip install -r requirements-track.txt
venvs/track/bin/pip uninstall -y opencv-python opencv-python-headless
venvs/track/bin/pip install --no-deps opencv-python-headless==5.0.0.93

python3.12 -m venv venvs/faces
venvs/faces/bin/pip install --upgrade pip
venvs/faces/bin/pip install -r requirements-faces.txt
venvs/faces/bin/pip uninstall -y opencv-python opencv-python-headless
venvs/faces/bin/pip install --no-deps opencv-python-headless==5.0.0.93

python3.12 -m venv venvs/worker
venvs/worker/bin/pip install -r requirements-worker.txt
```

`venvs/track` es el venv de `presence_gate.py` y `venvs/faces` el de
`face_identities.py`.

`requirements-track.txt` tiene `--extra-index-url https://download.pytorch.org/whl/cu126`
en su primera línea para que pip descargue **PyTorch para CUDA 12.6**. Esa línea no se
debe borrar: el torch de PyPI es para CUDA 13 y rompe el entorno.

`requirements-track.txt` fija `onnxruntime-gpu==1.26.0`, la última versión de PyPI
compilada para CUDA 12 que funciona con el CUDA 12.6 de torch. La compuerta no lo usa: es
para el codificador ReID del banco de pruebas. Desde la 1.27 la versión de PyPI es para
CUDA 13, y la 1.29.0 del índice CUDA 12 de onnxruntime necesita un CUDA 12 más nuevo
(`undefined symbol: cudaLibraryGetKernel`). Con cualquiera de ellas el codificador ReID
pasa a CPU sin avisar más que en el log.

`venvs/worker` solo hace falta para consultar SQS: `worker.py --local` corre también con
el `python3` del sistema.

### Opción B: desde cero (si los archivos de requisitos están obsoletos)

`venvs/track` lleva solo la familia PyTorch y `venvs/faces` solo la familia TensorFlow
(regla 1). `onnxruntime-gpu` y `onnx` son para el codificador ReID `.onnx` del banco de
pruebas, en CUDA 12.

```bash
python3.12 -m venv venvs/track
venvs/track/bin/pip install --upgrade pip
venvs/track/bin/pip install torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu126
venvs/track/bin/pip install ultralytics lap
venvs/track/bin/pip install onnxruntime-gpu==1.26.0 onnx
venvs/track/bin/pip uninstall -y opencv-python opencv-python-headless
venvs/track/bin/pip install --no-deps opencv-python-headless

python3.12 -m venv venvs/faces
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

**No se debe ejecutar `pip install -U ultralytics`.** La versión congelada ya funciona con
YOLO26m, y una actualización puede traer torch desde PyPI (rompe la regla 2) y reinstalar
`opencv-python` (rompe la regla 3).

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

La segunda y la tercera se corren desde la raíz del repo (usan `yolo26m-reid.onnx` y el
módulo `face_recognition`). La segunda solo comprueba el codificador ReID del banco de
pruebas. La tercera importa `face_recognition` y no solo `tensorflow` porque ese módulo
hace la misma precarga de `libcusolver` que `face_identities.py` (ver "Qué corre en la
GPU").

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
| --- | --- | --- |
| `FACE_GATING` | `1` | `1` corre antes la compuerta de presencia y analiza solo los tramos con alguien en cámara; `0` analiza todo el video. Con otro valor el worker no arranca. Ver "Compuerta de presencia" |
| `GATE_CHECKS_PER_S`, `GATE_REWIND_S`, `GATE_EXIT_S`, `GATE_CONF`, `GATE_IMGSZ` | `2`, `1.0`, `2.0`, `0.25`, `640` | Ajustes de la compuerta (ver "Compuerta de presencia"); con `FACE_GATING=1` el worker los valida al arrancar y se los pasa tal cual a `presence_gate.py`. Con un valor no válido el worker no arranca |
| `FACE_ADAPTIVE` | `0` | `0` intervalo fijo `FACE_INTERVAL_S`, `1` muestreo adaptativo. Con `FACE_ADAPTIVE=1` y `FACE_GATING=1` el worker no arranca |
| `FACE_MIN_FACES` | `8` | Con `FACE_ADAPTIVE=0`: rostros mínimos para conservar una identidad. Entero positivo |
| `FACE_INTERVAL_S` | `0.1` | Con `FACE_ADAPTIVE=0`: segundos de video entre fotogramas analizados. Número positivo |
| Otras `FACE_*` | | `FACE_EMBED_BATCH`, `FACE_DETECT_UPSCALE`, los intervalos y umbrales del muestreo adaptativo y la presencia mínima: ver "Identidad por rostro" |
| `AWS_SQS_QUEUE_URL` | (obligatoria) | Cola donde el backend publica los trabajos |
| `AWS_SQS_RESULTS_QUEUE_URL` | (obligatoria) | Cola donde el worker publica el avance, los resultados y los fallos |
| `AWS_REGION` | `us-east-2` | Región de SQS y S3 |
| `FACES_S3_PREFIX` | `faces` | Carpeta del bucket donde se suben los rostros |
| `FACES_URL_BASE` | (vacía) | URL base de las imágenes. Debe quedar vacía: el backend solo acepta la URL S3 del bucket y entrega al frontend una URL prefirmada |
| `CAMERA_TZ` | `America/Bogota` | Zona horaria del reloj de las cámaras |
| `KEEP_JOB_FILES` | `0` | Con `1` se conservan los archivos de cada trabajo en `database/jobs/` |
| `STAGE_TIMEOUT` | `7200` | Segundos máximos por pasada (`presence_gate.py` o `face_identities.py`) |
| `RESULT_PART_BYTES` | `245760` | Tamaño máximo en bytes de cada parte del resultado: un entero entre 1 y 245760 (240 KiB). El worker lo valida al arrancar y no arranca con otro valor |
| `MAX_ATTEMPTS` | `3` | Entregas de un mismo trabajo antes de darlo por fallido |

El worker y el backend no se conectan entre sí: solo hablan con SQS y S3. Por eso pueden
estar en máquinas distintas, sin IP fija ni puertos abiertos, y el worker puede arrancar
antes o después que el backend.

Permisos de AWS que necesita el worker, mejor con un rol de IAM asociado a la instancia
que con llaves en disco:

- `sqs:ReceiveMessage`, `sqs:DeleteMessage` y `sqs:ChangeMessageVisibility` sobre la cola
  de trabajos.
- `sqs:SendMessage` sobre la cola de resultados.
- `s3:GetObject` sobre `videos/*` y `s3:PutObject` sobre `faces/*` en el bucket.

Flujo por mensaje `VIDEO_UPLOAD_READY` (lo publica `SqsVideoJobPublisher` del backend):

1. Publica `VIDEO_PROCESSING_STARTED` en la cola de resultados.
2. Descarga `s3://<bucket>/<objectKey>` en `database/jobs/<jobId>/`.
3. Con `FACE_GATING=1` (por defecto) corre `presence_gate.py` (`venvs/track`) y después
   `face_identities.py --presence` (`venvs/faces`); con `FACE_GATING=0` corre solo
   `face_identities.py`.
4. Sube cada rostro a `s3://<bucket>/<FACES_S3_PREFIX>/<jobId>/<trackId>.jpg`, donde
   `<trackId>` es el número de la identidad.
5. Publica `VIDEO_PROCESSING_COMPLETED` en partes, con los rostros y sus embeddings, y
   borra el mensaje de la cola de trabajos.

Eventos que publica el worker en la cola de resultados (los consume
`SqsVideoResultListener` del backend):

| `eventType` | Cuándo | Campos propios |
| --- | --- | --- |
| `VIDEO_PROCESSING_STARTED` | Al tomar el trabajo y cada 120 s mientras corre | ninguno |
| `VIDEO_PROCESSING_COMPLETED` | Al terminar, una vez por parte | `part`, `parts`, `attemptId`, `embeddingModel`, `embeddingDimensions` y `appearances` |
| `VIDEO_PROCESSING_FAILED` | Cuando el trabajo se descarta | `reason` |

Todos llevan `eventVersion`, `jobId` y `sentAt` (UTC, ISO-8601). Ejemplo de un resultado
de una sola parte:

```json
{
  "eventType": "VIDEO_PROCESSING_COMPLETED",
  "eventVersion": "1",
  "jobId": "<jobId>",
  "sentAt": "2026-10-16T15:12:40.512345Z",
  "part": 1,
  "parts": 1,
  "attemptId": "<uuid>",
  "embeddingModel": "GhostFaceNet",
  "embeddingDimensions": 512,
  "appearances": [
    {
      "faceId": "<jobId>_<trackId>",
      "timestamp": "00:01:23",
      "frameUrl": "https://<bucket>.s3.us-east-2.amazonaws.com/faces/<jobId>/<trackId>.jpg",
      "capturedAt": "2026-10-16T10:05:21-05:00",
      "embedding": "<base64 de 512 float32>"
    }
  ]
}
```

- `faceId`: identidad por video, una por cada identidad de `face_identities.py`.
- `timestamp`: posición del rostro elegido dentro del video.
- `capturedAt`: ISO-8601 con offset, como exige `OffsetDateTime.parse`. Es el inicio de
  la grabación tomado del nombre del archivo del NVR (`Camera_chN_YYYYMMDDHHMMSS...`) más
  `timestamp`; si el nombre no tiene la hora o falta el tiempo del rostro, se usa la hora
  de subida. Siempre en `CAMERA_TZ`.
- `embedding`: el embedding medio de la identidad (ver `embeddings.json` en "Opciones y
  salidas").
- `embeddingDimensions` es la longitud de los embeddings: 512 con GhostFaceNet, y 0 si el
  video no dejó ninguna identidad.
- Un mensaje no puede pasar de 256 KiB, el tamaño que admite cualquier cola de SQS. Los
  resultados van en partes, así que no hay tope de rostros por video; solo una detección
  que por sí sola no quepa en una parte hace fallar el trabajo.

Manejo de fallos:

- Errores transitorios (red, una pasada que falla): el mensaje queda en la cola y SQS lo
  vuelve a entregar al vencer el visibility timeout. En la entrega número `MAX_ATTEMPTS`
  el worker publica `VIDEO_PROCESSING_FAILED` con el motivo y borra el mensaje.
- Errores permanentes (mensaje mal formado, video inexistente, una detección que no cabe
  en una parte del resultado): el worker publica `VIDEO_PROCESSING_FAILED` y borra el
  mensaje. `jobId` forma parte de rutas locales y de S3, así que solo se aceptan letras,
  números, `_` y `-`, hasta 64 caracteres; un mensaje sin un `jobId` así se borra sin
  avisar.
- Si no se puede publicar `VIDEO_PROCESSING_FAILED`, el mensaje queda en la cola para
  avisar en la siguiente entrega.
- Si el pipeline terminó pero no se pudo publicar el resultado, se conserva solo
  `payload.json` y la siguiente entrega lo reenvía sin repetir las pasadas. El archivo
  guarda `attemptId`, `embeddingModel`, `embeddingDimensions` y `appearances`; los
  formatos anteriores (una lista, o un objeto sin `attemptId`) se siguen leyendo y reciben
  un `attemptId` nuevo.
- `VIDEO_PROCESSING_COMPLETED` va en partes numeradas. El worker agrupa las detecciones en
  orden y cada mensaje queda por debajo de `RESULT_PART_BYTES` (240 KiB por defecto, unas
  80 detecciones con embedding); un resultado de una sola parte lleva `part` 1 y `parts`
  1. Cada parte lleva `attemptId` (un UUID nuevo por cada corrida del pipeline), `part`,
  `parts`, `embeddingModel` (`GhostFaceNet`) y `embeddingDimensions`. Cada detección lleva
  `embedding`: el vector unitario float32 little-endian en base64. Si falla un envío, la
  siguiente entrega reenvía todas las partes con el mismo `attemptId`, y el backend toma
  una parte repetida como una sobrescritura. El heartbeat sigue hasta enviar la última
  parte.
- Un heartbeat extiende la visibilidad del mensaje y publica `VIDEO_PROCESSING_STARTED`
  cada 120 s mientras corre el trabajo. El backend da por fallido un trabajo que deja de
  reportar avance. Cada pasada se corta tras `STAGE_TIMEOUT`.
- Con SIGTERM termina el trabajo en curso antes de salir. Cada pasada corre en su propia
  sesión, así que la señal no la corta a mitad de camino. Con systemd se usa
  `KillMode=mixed` y un `TimeoutStopSec` mayor que la duración de un trabajo.

### Un video local, sin AWS ni backend

```bash
python3 worker.py --local /ruta/al/video.mp4 --job-id prueba
```

Corre el mismo flujo que en producción (incluida la compuerta de presencia según
`FACE_GATING`) e imprime el payload, con el embedding de cada detección. `frameUrl` es la
ruta local del rostro. Los archivos quedan en `database/jobs/prueba/`.

### La identidad por rostro a mano

```bash
venvs/track/bin/python presence_gate.py --video /ruta/al/video.mp4 --out database/presence.json
venvs/faces/bin/python face_identities.py --video /ruta/al/video.mp4 --faces-dir database/faces_identidades --presence database/presence.json
```

La primera línea es la compuerta de presencia; `--video` y `--out` son obligatorios. En
`face_identities.py`, `--video` es obligatorio, sin `--faces-dir` se usa
`database/faces_identidades` y `--annotated-video` es opcional. Sin `--presence` (o con
`--no-gating`) `face_identities.py` analiza todo el video.

Nunca se usa `source venv/bin/activate` + `python3`: lanzar cada script con el intérprete
de su venv es lo que garantiza que corra sobre el stack correcto.

### El pipeline con video anotado (`run.sh`)

```bash
./run.sh [--no-gating] /ruta/al/video.mp4 [carpeta_de_salida]
```

`run.sh` usa `FACE_GATING` igual que el worker (`1` por defecto) y se detiene con un
error si tiene otro valor. La salida va a `database/run/<nombre del video>/` si no se
indica otra carpeta.

Con `FACE_GATING=1` corre `presence_gate.py` (escribe `presence.json`) y después
`face_identities.py` con `--presence`; con `FACE_GATING=0` o `--no-gating` (que gana sobre
`FACE_GATING=1`) omite la compuerta y analiza todo el video. Con `FACE_ADAPTIVE=1` y la
compuerta activa se detiene con un error antes de correr la compuerta. Escribe `faces/` e
`identities.mp4`, el video de los fotogramas analizados con cada rostro conservado en una
caja con su `ID <n>`; con la compuerta solo trae los fotogramas analizados de los tramos
activos, cada uno con su tiempo de origen (`HH:MM:SS.s`). Cada ejecución borra antes
`faces/` de esa carpeta.

---

## Identidad por rostro (`face_identities.py`)

Corre en `venvs/faces`, en un solo proceso. Busca todos los rostros del video y los agrupa
en identidades con sus embeddings:

1. Lee el video con `cv2.VideoCapture`: `grab()` avanza los fotogramas que se saltan sin
   decodificarlos y `retrieve()` solo decodifica los que se analizan.
   `face_sampling.py` elige qué fotogramas analizar. Con `FACE_ADAPTIVE=0` (por defecto)
   analiza un fotograma cada `FACE_INTERVAL_S` segundos. Con `FACE_ADAPTIVE=1` elige tras
   cada fotograma analizado cuánto avanzar hasta el siguiente (ver "Muestreo adaptativo").
   Si recibe un `presence.json` (ver "Compuerta de presencia"), analiza solo los
   fotogramas de ese intervalo fijo que caen dentro de los tramos activos.
2. Corre RetinaFace sobre el fotograma completo, con escalado (`allow_upscaling`) si
   `FACE_DETECT_UPSCALE=1`.
3. Alinea cada rostro con sus cinco puntos a la plantilla de ArcFace de 112x112 px.
4. Calcula el embedding con GhostFaceNet y lo normaliza a norma 1. Con
   `FACE_EMBED_BATCH=1` (por defecto) `face_embedding.py` junta los rostros de varios
   fotogramas en bloques de `EMBED_CHUNK` y hace una sola llamada al modelo por bloque; el
   último bloque se rellena con ceros para que el modelo vea siempre el mismo tamaño y no
   se recompile. El preprocesamiento es el mismo de `DeepFace.represent` con
   `detector_backend="skip"` y `align=False` en la versión instalada (0.0.100): canales en
   BGR, tamaño 112x112 (escala y relleno centrado si hiciera falta), `float32` y división
   por 255 solo si el máximo pasa de 1. Con `FACE_EMBED_BATCH=0` se llama a
   `DeepFace.represent` una vez por rostro, como antes. Sobre 199 rostros de P2E_S5_C1
   las dos formas difieren como mucho en 5e-7 por componente, ya normalizadas. Mide la
   nitidez como varianza del laplaciano dentro de la caja del rostro.
5. `identity_linking.py` agrupa los rostros en identidades (ver abajo).
6. Elige el mejor rostro de cada identidad por el área de la caja del rostro por la raíz
   cuadrada de la nitidez, con la confianza de RetinaFace como desempate, y lo guarda con
   cabeza y hombros (márgenes `OUTPUT_MARGIN_SIDE` / `TOP` / `BOTTOM`: 0.8 / 0.6 / 1.8
   veces el tamaño del rostro).
7. Escribe `embeddings.json` con el promedio unitario de los embeddings de los rostros de
   cada identidad que tiene rostro guardado (ver "Opciones y salidas").

Agrupación (`identity_linking.py`, solo numpy):

1. Solo entran los rostros con confianza de al menos `MIN_SCORE`.
2. Tracklets: un rostro se une al tracklet cuyo último rostro tiene IoU de al menos
   `LINK_IOU` y similitud coseno de al menos `LINK_SIMILARITY` con él. La asignación es
   voraz por IoU más similitud, de mayor a menor, con un rostro por tracklet y por
   fotograma. Un tracklet puede saltarse hasta `MAX_SKIP` fotogramas analizados; se
   cuentan fotogramas analizados y no fotogramas del video, porque con el muestreo
   adaptativo el paso cambia. Con el intervalo fijo, el número de muestra de cada
   fotograma analizado es su índice en la rejilla (`fotograma // paso`), con compuerta o
   sin ella, así que ambas corridas numeran igual los mismos fotogramas. Un tramo omitido
   por la compuerta de hasta `MAX_SKIP` fotogramas de la rejilla se comporta igual que
   esos fotogramas analizados sin rostros en la corrida sin compuerta; uno más largo corta
   el tracklet.
3. Fusión: agrupamiento jerárquico con enlace promedio (ponderado por el tamaño de cada
   tracklet) de los embeddings medios de los tracklets, hasta `MERGE_THRESHOLD`. Dos
   grupos que comparten un fotograma analizado nunca se fusionan: una persona no aparece
   dos veces en el mismo fotograma.
4. Presencia mínima. Con `FACE_ADAPTIVE=0` se descartan las identidades con menos de
   `MIN_FACES` rostros. Con `FACE_ADAPTIVE=1` cada rostro vale el intervalo (en segundos
   de video) que el muestreo elige después de su fotograma, es decir, el tiempo hasta el
   siguiente fotograma analizado. Tras un fotograma con rostros ese intervalo es denso o
   disperso, nunca inactivo, así que un rostro que llega tras un rato sin gente no cuenta
   el segundo vacío anterior. Se conserva la identidad si la suma llega a
   `MIN_PRESENCE_S` y tiene al menos `MIN_FACES_ABS` rostros. 0.8 s equivale a los 8
   rostros del intervalo fijo de 0.1 s.
5. Gemelo espejo: si dos identidades comparten un fotograma y sus embeddings medios
   tienen similitud de al menos `TWIN_THRESHOLD`, se descarta la de menor calidad. La
   calidad es la mediana de la nitidez por la mediana del tamaño del rostro (lado menor,
   en px). Pasa con reflejos y con detecciones repetidas de la misma persona.
6. Las identidades se ordenan por su primera aparición y se numeran desde 1.

| Parámetro | Valor | Dónde |
| --- | --- | --- |
| `INTERVAL_S` | `0.1` (con `FACE_ADAPTIVE=0`, un fotograma analizado cada 0.1 s, paso `max(1, round(fps * FACE_INTERVAL_S))`), o el valor de la variable `FACE_INTERVAL_S` | `face_sampling.py` |
| `RETINA_THRESHOLD` | `0.9` | `face_identities.py` |
| `RETINA_ALLOW_UPSCALING` | `True`, o el valor de la variable `FACE_DETECT_UPSCALE` | `face_identities.py` |
| `EMBEDDING_MODEL` | `GhostFaceNet` | `face_identities.py` |
| `EMBED_CHUNK` | `64` rostros por llamada al modelo | `face_embedding.py` |
| `MIN_SCORE` | `0.99` | `identity_linking.py` |
| `LINK_IOU` | `0.3` | `identity_linking.py` |
| `LINK_SIMILARITY` | `0.2` | `identity_linking.py` |
| `MAX_SKIP` | `2` | `identity_linking.py` |
| `MERGE_THRESHOLD` | `0.35` | `identity_linking.py` |
| `TWIN_THRESHOLD` | `0.6` | `identity_linking.py` |
| `MIN_FACES` | `8`, o el valor de la variable `FACE_MIN_FACES` (solo con `FACE_ADAPTIVE=0`) | `identity_linking.py` |
| `MIN_PRESENCE_S` | `0.8` s, o el valor de la variable `FACE_MIN_PRESENCE_S` (solo con `FACE_ADAPTIVE=1`) | `identity_linking.py` |
| `MIN_FACES_ABS` | `3`, o el valor de la variable `FACE_MIN_FACES_ABS` (solo con `FACE_ADAPTIVE=1`) | `identity_linking.py` |

Son valores congelados: no se cambian sin repetir la validación. Las excepciones se
eligen con variables de entorno. Las leen `face_identities.py`, `face_sampling.py` e
`identity_linking.py`, y `run.sh` y el worker se las pasan tal cual. Vacías usan el valor
por defecto; con un valor no válido el script se detiene con un mensaje:

| Variable | Valor por defecto | Valores válidos | Para qué |
| --- | --- | --- | --- |
| `FACE_EMBED_BATCH` | `1` | `0` o `1` | `1` calcula los embeddings en bloques; `0` vuelve a una llamada de DeepFace por rostro |
| `FACE_DETECT_UPSCALE` | `1` | `0` o `1` | `allow_upscaling` de RetinaFace. `0` es más rápido y puede perder rostros pequeños |
| `FACE_ADAPTIVE` | `0` | `0` o `1` | `0` (por defecto) intervalo fijo `FACE_INTERVAL_S` y regla `FACE_MIN_FACES`; `1` muestreo adaptativo, que en la evaluación perdió 7 de 78 personas. No se puede combinar con la compuerta de presencia |
| `FACE_INTERVAL_DENSE_S` | `0.1` | Número positivo, en segundos | Intervalo denso del muestreo adaptativo |
| `FACE_INTERVAL_SPARSE_S` | `0.3` | Número positivo, en segundos | Intervalo disperso del muestreo adaptativo |
| `FACE_INTERVAL_IDLE_S` | `1.0` | Número positivo, en segundos | Intervalo sin rostros del muestreo adaptativo |
| `FACE_CROWD_FACES` | `3` | Entero positivo | Rostros en un fotograma a partir de los cuales se pasa a denso |
| `FACE_IDLE_FRAMES` | `3` | Entero positivo | Fotogramas analizados seguidos sin rostros antes de pasar a inactivo |
| `FACE_DENSE_HOLD_S` | `1.0` | Número positivo, en segundos | Tiempo de video mínimo en denso desde el último motivo para entrar |
| `FACE_MIN_PRESENCE_S` | `0.8` | Número positivo, en segundos | Presencia mínima de una identidad con `FACE_ADAPTIVE=1` |
| `FACE_MIN_FACES_ABS` | `3` | Entero positivo | Rostros mínimos de una identidad con `FACE_ADAPTIVE=1` |
| `FACE_MIN_FACES` | `8` | Entero positivo | Con `FACE_ADAPTIVE=0`. En cámaras de baja resolución 5 encuentra algunas personas más |
| `FACE_INTERVAL_S` | `0.1` | Número positivo, en segundos | Con `FACE_ADAPTIVE=0`. Acelerar videos largos analizando menos fotogramas |

### Compuerta de presencia (`presence_gate.py`)

La mayoría de los videos de vigilancia pasan largos ratos sin nadie en cámara, y
RetinaFace con escalado sobre cada fotograma analizado es la parte cara del flujo.
La compuerta decide qué tramos del video merecen esa pasada: una revisión barata con
YOLO26m busca personas unas pocas veces por segundo, y `face_identities.py` solo analiza
los fotogramas de los tramos donde alguien apareció. Los fotogramas analizados son un
subconjunto exacto de los que se analizarían sin ella, así que las detecciones son un
subconjunto de las de la corrida sin compuerta. Las detecciones y las identidades son
iguales a las de esa corrida, salvo diferencias numéricas de la GPU del orden de 1e-7 en
los puntajes, cuando los tramos omitidos no tienen rostros (los números de
muestra coinciden y un hueco corto se trata igual en ambas, ver "Agrupación"), y fueron
iguales en todas las mediciones (ver "Resultados medidos de la compuerta").

Son dos pasadas, dos procesos, porque PyTorch (YOLO) y TensorFlow (RetinaFace) no pueden
convivir en un proceso en esta máquina. Se comunican por un archivo, `presence.json`:

1. `presence_gate.py` (`venvs/track`) lee el video en orden con `grab()`, recupera
   (`retrieve()`) un fotograma cada `max(1, round(fps / GATE_CHECKS_PER_S))` empezando por
   el 0, y corre `YOLO(yolo26m.pt)` (los pesos junto al script, así que no importa desde
   qué carpeta se lance) solo con la clase persona. Hay presencia si detecta al menos una
   caja. Si un fotograma no se puede decodificar, imprime un aviso, lo cuenta como
   presencia (así la pasada de rostros queda activa) y sigue leyendo.
2. Cada fotograma con persona abre o amplía un tramo activo que cubre desde
   `GATE_REWIND_S` segundos antes hasta `GATE_EXIT_S` segundos después. Los tramos que se
   solapan o se tocan se unen, y se recortan a los límites del video.
3. `face_identities.py --presence presence.json` mantiene la misma rejilla de
   fotogramas, anclada en el fotograma 0 (un fotograma cada `FACE_INTERVAL_S`), y analiza
   solo los que caen dentro de un tramo. Los fotogramas fuera de los tramos solo se
   avanzan con `grab()`: no se decodifican a imagen, no se guardan en memoria ni se
   vuelve atrás en el video.

Variables de la compuerta (las lee `presence_gate.py`; vacías usan el valor por defecto
y un valor no válido detiene el script con un mensaje y también el arranque del
worker). Los valores por defecto salen del barrido de umbrales descrito en "Resultados
medidos de la compuerta":

| Variable | Valor por defecto | Valores válidos | Para qué |
| --- | --- | --- | --- |
| `FACE_GATING` | `1` | `0` o `1` | La leen el worker y `run.sh`. `1` corre la compuerta antes de `face_identities.py`; `0` analiza todo el video |
| `GATE_CHECKS_PER_S` | `2` | Número positivo | Revisiones de YOLO por segundo de video |
| `GATE_REWIND_S` | `1.0` | Número no negativo, en segundos (0 desactiva el margen) | Cuánto antes de la detección empieza el tramo, para no perder a quien YOLO ve tarde |
| `GATE_EXIT_S` | `2.0` | Número no negativo, en segundos (0 desactiva el margen) | Cuánto después de la última detección sigue activo el tramo |
| `GATE_CONF` | `0.25` | Número mayor que 0 y hasta 1 | Confianza mínima de YOLO para contar una persona |
| `GATE_IMGSZ` | `640` | Entero positivo | Tamaño de imagen de YOLO |

Con `FACE_ADAPTIVE=1` la compuerta no se admite: `face_identities.py --presence` se
detiene con un error y el worker no arranca si tiene `FACE_ADAPTIVE=1` y `FACE_GATING=1`.
Para ignorar el archivo de presencia en `face_identities.py` está
`--no-gating`. En el worker y en `run.sh` se desactiva con `FACE_GATING=0`, y en `run.sh`
también con `--no-gating`; `run.sh` rechaza `FACE_ADAPTIVE=1` con la compuerta antes de
correrla.

El worker valida `GATE_CHECKS_PER_S`, `GATE_REWIND_S`, `GATE_EXIT_S`, `GATE_CONF` y
`GATE_IMGSZ` al arrancar con `FACE_GATING=1`, con las mismas reglas y
mensajes que `presence_gate.py` (el código es el mismo, `read_gate_settings` de
`presence_intervals.py`, que no importa torch, cv2 ni ultralytics), y no arranca si alguna
no es válida. Así un valor erróneo en `.env` no hace fallar cada trabajo `MAX_ATTEMPTS`
veces.

Si la pasada de la compuerta falla, el trabajo falla como con la pasada de rostros: se
reintenta hasta `MAX_ATTEMPTS` veces y después se publica `VIDEO_PROCESSING_FAILED`. No
hay respaldo que corra la pasada de rostros sin compuerta, a propósito: ese respaldo
ocultaría un `venvs/track` roto (sin CUDA, sin `yolo26m.pt`, con dependencias dañadas) y
duplicaría en silencio el costo de cada trabajo, porque todos pasarían a analizar el
video completo sin que nadie lo notara.

Modelo de costo, por hora de video:

- Pasada de la compuerta: la lectura secuencial de todos los fotogramas más
  `3600 * GATE_CHECKS_PER_S` inferencias de YOLO. Es un costo fijo que no depende de
  cuánta gente aparece.
- Pasada de rostros: solo sobre el tiempo activo. Con fracción activa `f`, analiza
  `f * 3600 / FACE_INTERVAL_S` fotogramas con RetinaFace y GhostFaceNet, en lugar de
  `3600 / FACE_INTERVAL_S`. La lectura de los fotogramas omitidos sigue costando un
  `grab()` cada uno.
- Costo fijo de la pasada de rostros, que no depende de `f`: cargar los modelos y
  decodificar el video en orden, una vez con `grab()` hasta el último tramo y otra al
  escribir los rostros, hasta el último mejor rostro.
- Total: costo de la compuerta, más el costo fijo de la pasada de rostros, más `f` veces
  el costo por fotograma de la pasada de rostros sin compuerta. Ahorra mientras `f` sea
  pequeña y la compuerta más ese costo fijo cuesten menos que la parte omitida de la
  pasada de rostros. En el video largo, sin el costo fijo la cuenta daría
  171 + 0.043 x 3645 = 328 s contra 466 s medidos; la diferencia, unos 140 s, se
  explica por ese costo fijo.

Formato de `presence.json`:

| Campo | Contenido |
| --- | --- |
| `video` | Nombre del archivo de video |
| `fps`, `total_frames` | Fotogramas por segundo y fotogramas que la compuerta leyó de verdad (no el conteo que declara el contenedor, que puede ser menor) |
| `check_step` | Fotogramas entre dos revisiones |
| `settings` | Los valores de `GATE_*` con que se corrió |
| `checks` | Lista de `[fotograma, hay_persona]` en orden |
| `confidences` | Lista de `[fotograma, confianza]` alineada con `checks`: la mayor confianza de persona que devolvió YOLO en esa revisión (0 si no hubo ninguna) |
| `intervals` | Lista de `[inicio, fin]` de fotogramas, inclusivos, ordenados y sin solaparse |
| `switches` | Registro de cambios (abajo) |
| `active_frames`, `active_fraction` | Fotogramas en tramos activos y su fracción del video |
| `wall_s` | Segundos que tardó la compuerta |

Cada elemento de `switches` es `{"evento": "activo" o "inactivo", "frame", "tiempo",
"disparo_frame", "disparo_tiempo"}` con los tiempos en `HH:MM:SS.s`. Un `activo` marca el
inicio de un tramo y su disparo es la primera persona que lo abrió; un `inactivo` marca el
fotograma siguiente al fin del tramo y su disparo es la última persona vista en él (no
aparece si el tramo llega al último fotograma). `presence_gate.py` imprime cada cambio,
por ejemplo `activo desde 00:01:02.3 (persona a las 00:01:03.3)`, y una línea de
resumen con las revisiones, la fracción activa y el tiempo.

`face_identities.py` escribe `"gating": true|false` en `detections.json` y en
`identities.json`; con la compuerta agrega `intervals` y `active_fraction`, y en
`detections.json` además `presence_total_frames` y `frames_read` (los fotogramas que la
pasada de rostros avanzó de verdad). Si el video entrega más de un segundo de fotogramas
menos de los que `presence.json` implica, imprime una advertencia: puede ser el archivo
de otro video. En su resumen imprime la fracción de fotogramas analizados.

### Resultados medidos de la compuerta

Medido en una T4 contra la misma configuración sin compuerta (`--no-gating`). Una
detección perdida es una cara que la corrida sin compuerta encontró y la corrida con
compuerta no.

Las cifras de esta sección vienen de las corridas más amplias del 2026-11-17. La medición
se repitió el 2026-11-22 con el código entregado sobre un conjunto reducido de 8 videos
(sección "Pruebas"): 0 detecciones perdidas y las mismas 107 personas encontradas, con un
factor de 1.74 con compuerta contra 1.68 sin ella.

| Conjunto | Sin compuerta | Con compuerta | Detecciones perdidas | Personas encontradas |
| --- | --- | --- | --- | --- |
| Video de 45 min, 4.3 % con personas (ChokePoint unido con tramos vacíos: unos 7.4 s de metraje vacío real repetidos en ida y vuelta, más un fotograma congelado) | 3645 s (factor 1.35) | 466 s (factor 0.17: compuerta 171 s, rostros 295 s) | 0 | 15 de 15 en ambas |
| 78 videos de ChokePoint (portales 1 y 2, entrando y saliendo, más los de mucha gente) | factor 1.66 en 69 de ellos | factor 1.80 en esos 69, medido con 4 revisiones por segundo | 0 con 4 revisiones por segundo (corrida real, los 78) y 0 con 2 (barrido sobre los 78, solo detecciones) | iguales con 4 revisiones por segundo (corrida real: 1115 en ambas en los 72 videos con etiquetas, 1329 identidades en ambas); con 2 no se midieron en estos videos |
| Camera3 y Camera_ch6 (residenciales, 960x480) | factor 1.75 y 1.60 | factor 1.82 y 1.68 | 0 | iguales (7 y 2) |
| Casos borde: persona desde el fotograma 0, hasta el último, menos de 1 s, varias seguidas, quieta 60 s | | | 0 | iguales |

Las corridas reales con el valor por defecto, 2 revisiones por segundo, son el video
largo, los casos borde y las dos cámaras. Los 78 videos de ChokePoint se corrieron de
verdad con 4 revisiones por segundo; con 2 solo se simularon sus detecciones en el
barrido. La exhaustividad con compuerta se mide sobre los fotogramas analizados de la
corrida sin compuerta, así que una persona etiquetada en un fotograma omitido cuenta como
no encontrada.

- En todas las mediciones no se perdió ninguna detección y las identidades fueron las
  mismas que sin compuerta. Eso no es una garantía general: solo vale mientras los tramos
  omitidos no tengan rostros.
- La fracción activa y el factor del video largo son el mejor caso: sus tramos vacíos son
  unos 7.4 s de metraje vacío real repetidos en ida y vuelta durante unos 40 minutos, más
  un fotograma congelado. Una grabación residencial real, con cambios de luz, paso a
  infrarrojo, sombras, autos o animales, puede activar la compuerta más seguido.
- Ahorra en videos mayormente vacíos: el video largo es 7.8 veces más rápido. En clips
  casi siempre ocupados (86 a 100 % activos) cuesta entre 4 y 9 % más, porque se suma la
  pasada de la compuerta y no hay tramos que saltar.
- Con 4.3 % activo, el costo que queda es la compuerta (0.06 veces la duración) más la
  pasada de rostros, que decodifica con `grab()` todo el video y carga los modelos.

Barrido de umbrales: las revisiones por segundo y el tamaño de imagen se corrieron de
verdad (`GATE_CHECKS_PER_S` 2 y 4, `GATE_IMGSZ` 320 y 640, con confianza mínima 0.05 y la
confianza de cada revisión guardada). `GATE_CONF` (0.1, 0.25, 0.4) y `GATE_EXIT_S` (1 y
2) se aplicaron después sobre esos archivos, simulando la pasada de rostros a partir de
la corrida sin compuerta, lo que es exacto para las detecciones porque los fotogramas
analizados con compuerta son un subconjunto de los de la corrida sin ella. Los tiempos del
barrido son estimaciones optimistas, porque no modelan la decodificación de los
fotogramas omitidos: con 4 revisiones por segundo estimó 432 s para el video largo contra
552 s medidos. Ese costo es el mismo para todos los ajustes, así que el orden entre ellos
casi no cambia. El barrido cubre 81 videos: los 78 de ChokePoint, el video largo y las
dos cámaras. Los umbrales se eligieron con el portal 1 y se confirmaron con el portal 2 y
las dos cámaras residenciales:

- Con `GATE_IMGSZ` 640 ninguna combinación pierde una detección.
- Con `GATE_IMGSZ` 320, YOLO no ve a las personas pequeñas y lejanas de Camera_ch6: con
  2 revisiones por segundo pierde hasta 36 detecciones (5 puntos de exhaustividad en ese
  video), y con 4 revisiones por segundo y confianza 0.4 pierde 13.
- Se eligió 2 revisiones por segundo a 640: es lo más barato sin pérdidas, la compuerta
  cuesta un 32 % menos que con 4, y toda la gama de confianza de 0.1 a 0.4 queda sin
  pérdidas, así que 0.25 tiene margen hacia ambos lados.
- Con 2 revisiones por segundo, una persona visible menos de 0.5 s puede no activar la
  compuerta. Tal aparición no forma identidad de todos modos: hacen falta
  `FACE_MIN_FACES` (8) rostros, unos 0.8 s con `FACE_INTERVAL_S` 0.1.

### Muestreo adaptativo

Es la alternativa al intervalo fijo dentro del mismo flujo, no la predeterminada: se
activa con `FACE_ADAPTIVE=1` y no se puede combinar con la compuerta de presencia.

Tras cada fotograma analizado, `face_sampling.py` elige el modo según lo que encontró, y
el modo da el intervalo hasta el siguiente fotograma (paso en fotogramas
`max(1, round(fps * intervalo))`):

| Modo | Intervalo | Cuándo |
| --- | --- | --- |
| Denso | `FACE_INTERVAL_DENSE_S` (0.1 s) | El fotograma tiene al menos `FACE_CROWD_FACES` rostros, o aparece un rostro que no continúa ninguno del fotograma analizado anterior (una llegada). Se mantiene al menos `FACE_DENSE_HOLD_S` de video desde el último motivo |
| Disperso | `FACE_INTERVAL_SPARSE_S` (0.3 s) | 1 o 2 rostros y todos continúan |
| Inactivo | `FACE_INTERVAL_IDLE_S` (1.0 s) | `FACE_IDLE_FRAMES` fotogramas analizados seguidos sin rostros |

Un rostro continúa si su caja tiene IoU de al menos `LINK_IOU` (0.3) con la de algún
rostro del fotograma analizado anterior. Cuentan todos los rostros que devuelve
RetinaFace con su umbral (0.9), no solo los que pasan `MIN_SCORE`. El video empieza en
denso; un fotograma sin rostros antes de llegar a `FACE_IDLE_FRAMES` conserva el modo que
había. Los umbrales están en segundos y no en fotogramas, así que no dependen de los fps
del video.

### Tiempo de proceso

Costo medido en una T4 con P2E_S5_C1 (800x600, 30 fps, mucha gente):

| Parte | Costo |
| --- | --- |
| RetinaFace sobre el fotograma completo | unos 139 ms por fotograma con escalado, 87 ms sin él |
| GhostFaceNet, una llamada a `DeepFace.represent` por rostro | unos 190 ms por rostro |
| GhostFaceNet en bloques de 64 | unos 3 ms por rostro |
| Arranque (TensorFlow, modelos, compilación del bloque) | unos 18 s fijos |

Antes el costo lo dominaba GhostFaceNet (rostros por fotograma por 190 ms); con los
bloques lo domina RetinaFace, que se paga por fotograma analizado. Por eso el muestreo
adaptativo y `FACE_DETECT_UPSCALE` son las palancas que quedan: el tiempo es
aproximadamente arranque + fotogramas analizados por el costo de RetinaFace + rostros por
3 ms.

P2E_S5_C1 (26.9 s de video), `face_identities.py` corrido a mano con
`venvs/faces/bin/python` (tiempo total del proceso):

| Configuración | Fotogramas analizados | Identidades | Tiempo |
| --- | --- | --- | --- |
| `FACE_EMBED_BATCH=0 FACE_ADAPTIVE=0` (una llamada por rostro) | 269 | 24 | 232 s |
| `FACE_EMBED_BATCH=1 FACE_ADAPTIVE=0` | 269 | 24 | 60 s |
| `FACE_EMBED_BATCH=1 FACE_ADAPTIVE=0 FACE_DETECT_UPSCALE=0` | 269 | 24 | 36 s |
| `FACE_EMBED_BATCH=1 FACE_ADAPTIVE=1` | 221 (215 densos, 2 dispersos, 4 inactivos) | 24 | 53 s |

Las configuraciones 1 y 2 dan las mismas 24 identidades con los mismos 889 rostros
(fotograma y caja) y el mismo `results.json`. Sin escalado salen 24 identidades con 868
rostros y con el muestreo adaptativo 24 con 881; en los dos casos cada una comparte
rostros con una identidad distinta de la configuración 1. `FACE_DETECT_UPSCALE` sigue en
`1` hasta validar la exactitud sin escalado. En este video casi siempre hay 3 rostros o
más, así que el muestreo adaptativo ahorra poco; ahorra más en grabaciones con ratos sin
gente o con pocas personas. En un video corto pesa el arranque fijo de unos 18 s
(importar TensorFlow, cargar los modelos, la primera llamada de RetinaFace y la
compilación del bloque de 64).

El worker corta cada pasada tras `STAGE_TIMEOUT` (7200 s por defecto).

Con `FACE_ADAPTIVE=0`, `FACE_INTERVAL_S` sigue siendo la palanca para las grabaciones
largas: con 0.2 s se analiza la mitad de los fotogramas. Los parámetros de la agrupación
se validaron con 0.1 s y con otro intervalo hay que revisar el resultado: cada persona
deja menos rostros y `FACE_MIN_FACES` descarta más identidades, y un tracklet se corta
tras `MAX_SKIP` fotogramas analizados, que pasan a ser más segundos. En P2E_S5_C1 (24
identidades con 0.1 s), con 0.2 s siguen saliendo 24 identidades. Con 0.5 s y
`FACE_MIN_FACES` 8 quedan solo 4; con 5 quedan 17 y con 3 salen 26, ya con algún
duplicado posible. En este video 0.2 s no cambia el resultado; con intervalos mayores hay
que bajar también `FACE_MIN_FACES` y revisar el resultado.

### Opciones y salidas

| Opción | Valor por defecto | Efecto |
| --- | --- | --- |
| `--video` | (obligatorio) | Video de entrada |
| `--faces-dir` | `database/faces_identidades` | Carpeta de salida |
| `--annotated-video` | (ninguno) | Si se indica, escribe ahí un video `mp4v` de los fotogramas analizados, a `fps / paso` fps (con el paso denso en el muestreo adaptativo, repitiendo cada fotograma hasta el siguiente analizado; las repeticiones se calculan sobre el tiempo acumulado para que el video no se desfase), con cada rostro conservado en una caja con su `ID <n>`. Con la compuerta escribe cada fotograma analizado una sola vez, con su tiempo de origen (`HH:MM:SS.s`) |
| `--presence` | (ninguno) | `presence.json` de `presence_gate.py`: solo se analizan los fotogramas de la rejilla que caen en los tramos activos |
| `--no-gating` | (apagada) | Ignora `--presence` y analiza todo el video |

Salidas en `--faces-dir`:

| Archivo | Contenido |
| --- | --- |
| `<id>/face_of_<id>.jpg` | Mejor rostro de la identidad, con cabeza y hombros |
| `results.json` | `tracks_checked` (número de identidades) y `faces`, con una entrada por rostro guardado: `track_id` (el número de la identidad, como texto), `face_file` (ruta relativa a `--faces-dir`), `candidate` (siempre `null`), `score` (confianza de RetinaFace), `face_box`, `face_px` (área de la caja del rostro, en px cuadrados), `sharpness`, `candidates_checked` (rostros de la identidad), `frame_idx` y `time_s`. El worker usa `track_id`, `face_file` y `time_s` |
| `detections.json` | `fps`, `step`, `frames_analysed` (índices de los fotogramas analizados) y `detections`, con `frame`, `time_s`, `box`, `score` y `px` de cada rostro que devolvió RetinaFace en cada fotograma analizado, antes de los filtros de identidad |
| `embeddings.json` | `model` (`GhostFaceNet`), `dimensions` (512, o 0 si no quedó ninguna identidad) y `embeddings`: por cada `track_id` de `results.json`, el vector unitario del promedio de los embeddings de sus rostros (`float32` little-endian en base64). El worker lo envía con cada detección |
| `identities.json` | `video`, `fps`, `step` (el paso fijo, o el paso denso con el muestreo adaptativo), `adaptive`, `gating` (si se usó la compuerta; con ella también `intervals`, los tramos activos, y `active_fraction`, la fracción del video que cubren), `steps` (paso de cada modo), `analysed_frames`, `frames_per_mode`, `wall_time_s` y, por identidad, `id` y todos sus rostros con `frame`, `time_s`, `box`, `score`, `px` e `interval_s` (intervalo que se eligió después de su fotograma y que el rostro aporta a la presencia). No guarda embeddings (están en `embeddings.json`) |

Al terminar imprime cuántos fotogramas analizó (por modo, con el muestreo adaptativo),
cuántos rostros detectó, cuántas identidades conservó y el tiempo total del proceso.

### Por qué

Una versión anterior del pipeline daba la identidad por el seguimiento del cuerpo y
elegía después un rostro por persona seguida. En ChokePoint P2E_S5 (tres cámaras de
800x600 con unas 24 personas cada una) encontraba entre 9 y 11 personas, la mitad de las
pistas mezclaban a varias personas y el rostro de una podía quedar en la pista de otra;
la identidad por rostro da 24, 24 y 27 identidades (C1, C2 y C3), cada una coherente en
sus rostros. En Camera3 (CCTV etiquetado del proyecto, 14 personas) el seguimiento del
cuerpo daba 7 personas, 8 duplicados y 4 identidades mezcladas, y la identidad por rostro
da 7 personas, 2 duplicados y 0 identidades mezcladas. Una de las tres cámaras de
ChokePoint se dejó aparte y no se usó para ajustar los parámetros.

---

## Pruebas

Las pruebas del flujo completo se corren desde el banco de pruebas `tesis-experiments`
(`/data/TesisFaceRecognition/tesis_face_recognition_experiments`), que llama al pipeline
como subprocesos y nunca edita sus archivos. La última corrida fue el 2026-11-22, entre
las 21:08 y las 22:02 UTC, en la g4dn.xlarge del proyecto (Tesla T4), contra el commit
`a9c95c7` de este repo (rama `test-suite`, sin cambios pendientes) y con los parámetros de
producción: compuerta de presencia activa y los valores por defecto de "Configuración
vigente".

### Cómo se corren

```bash
cd /data/TesisFaceRecognition/tesis_face_recognition_experiments
bench/flujo_actual.sh
venvs/eval/bin/python -m harness.export_excel
```

`bench/flujo_actual.sh` escribe sus resultados en `runs/flujo_actual/`, y
`harness.export_excel` regenera `tuning_results.xlsx` en la raíz de `tesis-experiments`,
con las hojas Resumen, Flujo - totales, Flujo - videos, Entre cámaras, Entre cámaras -
umbrales y Entorno. El detalle por video y por umbral está en ese Excel; aquí solo van
las cifras principales.

### Videos

El conjunto se redujo para que la corrida completa quepa en una hora. Son 8 videos:

- 2 cámaras residenciales con etiquetas del proyecto, Camera_ch6 y Camera3 (960x480,
  rostros pequeños).
- 4 videos de ChokePoint con su ground truth oficial: P1E_S1_C1, P1E_S1_C2, P1E_S2_C1 y
  P2E_S1_C2.1.
- 2 clips de casos borde para la compuerta: borde_inicio y borde_seguidos.

El video de 45 minutos no se repitió: solo su corrida sin compuerta tarda cerca de una
hora.

### Qué se mide, para qué y resultados

- **PRU-01, pruebas automáticas.** La suite pytest de `tesis-experiments` cubre las
  métricas del banco y `worker.py`, para poder confiar en las herramientas de medición y
  en el worker. Resultado: 459 pruebas aprobadas y 0 fallidas.
- **ENV-01, entorno.** Revisa los venvs `track`, `faces`, `eval` y `worker`: bibliotecas,
  GPU y lectura de video. Resultado: 14 de 14 revisiones aprobadas.
- **FLU-01, flujo de producción.** Corre la compuerta y `face_identities.py` para medir
  cuántos rostros se detectan, con qué precisión y si cada persona queda en una sola
  identidad por video. En ChokePoint la precisión de detección fue 97.0 % y la
  exhaustividad 99.9 %; 95 de las 99 personas etiquetadas tuvieron su propia identidad,
  sin duplicados, con 6 identidades que mezclan personas y una pureza de 95.9 %. En las
  cámaras residenciales, con rostros pequeños, la precisión fue 85.8 % y la exhaustividad
  80.1 %; se encontraron 9 de 17 personas, con 2 duplicados y 1 identidad mezclada.
- **FLU-02, compuerta.** Compara el mismo flujo con y sin compuerta (`--no-gating`) para
  confirmar que la compuerta no pierde nada y ver su costo. Ninguna detección de la
  corrida sin compuerta faltó con ella, la exhaustividad no cambió (95.0 % en los 8
  videos), se encontraron las mismas 107 personas y no hubo regresiones. El factor de
  tiempo (segundos de proceso por segundo de video) fue 1.68 sin compuerta y 1.74 con
  ella: estos videos tienen personas casi todo el tiempo, así que la compuerta cuesta un
  poco. El ahorro aparece en video mayormente vacío, como el de 45 minutos medido el
  2026-11-17, que fue 7.8 veces más rápido.
- **FLU-03, búsqueda entre cámaras.** Usa el embedding medio unitario de GhostFaceNet por
  identidad que envía el worker, leído del `embeddings.json` de la corrida con compuerta,
  convertido con `bench/entre_camaras_corrida.py` y puntuado con `bench/entre_camaras.py`
  contra el ground truth de ChokePoint, para ver si la búsqueda del backend propone a la
  misma persona en otros videos. Con 4 videos y 99 identidades (89 etiquetadas), el primer
  candidato fue la misma persona en el 100 % de las búsquedas, también en otra sesión y en
  el otro portal (azar: 3.8 %). Con el umbral 0.55 del backend, el 98.0 % de los
  candidatos listados eran correctos y se encontró el 100 % de los pares reales. Con solo
  4 videos la galería es pequeña; la medición de 72 videos del 2026-11-20 (98.4 %) sigue
  siendo la referencia más amplia.

### Pruebas anteriores

Las suites anteriores del banco (TRK, DET, SEL, EMB, CLU, E2E y XV) midieron el flujo
previo basado en seguimiento (`tracking.py` y `face_recognition.py`), que producción ya
no corre. Ya no están en el Excel; sus corridas siguen en `tesis-experiments`.

---

## Notas y peculiaridades

- **Sin interfaz gráfica.** Los venvs usan OpenCV headless, así que no se abren ventanas.
  `--annotated-video` escribe el video con `cv2.VideoWriter` a `fps / paso` fps (10 fps
  en un video de 30 fps con `FACE_INTERVAL_S` 0.1), para que la reproducción vaya en
  tiempo real; con la compuerta solo trae los fotogramas analizados de los tramos activos.
- **`mp4v` no se reproduce en navegadores ni en QuickTime** (VLC y mpv sí). Para verlo en
  cualquier lado se convierte con ffmpeg; `setpts` lo acelera 8 veces:
  ```bash
  ffmpeg -i identities.mp4 -vf "setpts=PTS/8" -r 8 -c:v libx264 -pix_fmt yuv420p \
    -movflags +faststart identities_h264.mp4
  ```
  `face_identities.py` usa `mp4v` porque `avc1` suele fallar: las ruedas de OpenCV para
  Linux vienen sin x264.
- **`cv2.VideoWriter_fourcc` no existe en OpenCV 5.** Se usa `cv2.VideoWriter.fourcc(...)`.
- **Memoria de GPU:** `face_identities.py` define `TF_FORCE_GPU_ALLOW_GROWTH=true` para
  que TensorFlow no tome toda la memoria de la T4. Esa línea se debe mantener.
- **Un segfault siempre significa** que un entorno se contaminó con el otro framework.
  No se depura: se reconstruye.

---

## Reconocimiento entre videos

La identidad sigue siendo por video: `face_identities.py` compara embeddings solo dentro
de un mismo video. Para la búsqueda entre videos escribe en `embeddings.json` un vector
por detección, el promedio unitario de los embeddings de GhostFaceNet de sus rostros, y el
worker lo envía con cada detección en `VIDEO_PROCESSING_COMPLETED`. El pipeline no compara
videos entre sí: el backend guarda los vectores, propone detecciones de otros videos
ordenadas por similitud (producto punto, umbral 0.55 por defecto) y un vigilante confirma
o descarta cada par.

La decisión D001 había descartado el reconocimiento entre videos con las cámaras
residenciales (960x480, rostros de unos 21 px de mediana), donde los embeddings no
distinguían personas mejor que el azar. La decisión D007 lo retomó con los videos de
ChokePoint, de rostros más grandes. En los 72 videos de ChokePoint con etiquetas, con el
embedding medio por identidad, el primer candidato fue la misma persona en el 98.4 % de
las búsquedas sobre los demás videos (azar: 3.2 %), y con el umbral 0.55 el 93 % de los
candidatos listados eran correctos y se encontró el 94 % de los pares reales
(`bench/entre_camaras.py` en `tesis-experiments`). Esas cifras vienen de la medición
amplia del 2026-11-20; se repitió el 2026-11-22 con el código entregado sobre un conjunto
reducido de 4 videos de ChokePoint (sección "Pruebas"), donde el primer candidato fue la
misma persona en el 100 % de las búsquedas y, con el umbral 0.55, el 98.0 % de los
candidatos listados eran correctos. Con video como el de las cámaras
residenciales la búsqueda no es confiable. La decisión D002 (agrupar con embeddings los
rostros de las personas seguidas en un video perdía personas y mezclaba a personas
distintas) es anterior a la identidad por rostro actual.

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
