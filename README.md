# Pipeline de reconocimiento facial (YOLO26 + RetinaFace)

Rastrea personas en grabaciones de CCTV con **YOLO26m** (Ultralytics) y después usa
**RetinaFace** para localizar y guardar el mejor rostro de cada persona rastreada.
La salida alimenta una etapa de reconocimiento/embeddings con **DeepFace**.

El pipeline está dividido en **dos procesos con dos entornos virtuales separados**.
Esto no es una decisión de estilo: PyTorch y TensorFlow no pueden compartir de forma
segura un mismo proceso Linux en esta máquina (provocan segfaults durante la
inicialización de las bibliotecas nativas).

```
                     venvs/track (solo PyTorch)             venvs/faces (solo TensorFlow)
┌───────────┐   ┌──────────────────────────────┐   ┌───────────────────────────────────┐
│ input.mkv │──▶│ tracking.py                  │──▶│ face_recognition.py               │
└───────────┘   │ Seguimiento de personas      │   │ Detección de rostros              │
                │ con YOLO26m + ReID           │   │ con RetinaFace                    │
                │ 1 de cada 30 fotogramas      │   │ evalúa TODOS los candidatos       │
                │ guarda recortes candidatos   │   │ guarda el MEJOR rostro por persona│
                └──────────────┬───────────────┘   └───────────────┬───────────────────┘
                               ▼                                   ▼
                 database/candidates/<id>/*.jpg      database/faces/<id>/face_of_<id>.jpg
                                                     (entrada para DeepFace, mismo venv)
```

El sistema de ficheros (`database/`) es el único puente entre las dos etapas.

---

## Máquina objetivo

Desarrollado y probado en esta configuración exacta. Otros entornos Linux + NVIDIA
deberían funcionar, pero esta es la referencia:

| Componente | Valor |
|---|---|
| Instancia | AWS EC2 **g4dn.xlarge** (4 vCPU, 16 GiB de RAM) |
| GPU | NVIDIA **Tesla T4**, 16 GiB de VRAM (capacidad de cómputo 7.5) |
| SO | Ubuntu 22.04 |
| Python | 3.12 |
| Driver NVIDIA | 580.173.02 (sirve cualquier driver compatible con CUDA ≥ 12.6 — comprueba `nvidia-smi`) |
| Almacenamiento | El repo vive en el volumen EBS secundario montado en `/data` (115 GB). Mantén ≥ 40 GB libres: los dos venvs suman ≈ 14 GB, más los pesos de los modelos y las salidas. |

Dos cosas que parecen erróneas pero son correctas en esta máquina:

- `nvidia-smi` dice `CUDA Version: 13.0` — ese es el **máximo del driver**, no lo
  que usamos. Los venvs incluyen su propio espacio de usuario CUDA **12.6**.
- `nvcc` no está instalado y **no hace falta**. Nunca ejecutes
  `sudo apt install nvidia-cuda-toolkit`.

---

## Estructura del repositorio

```
FaceRecognition/
├── tracking.py               # Etapa 1: seguimiento con YOLO26m → recortes candidatos
├── face_recognition.py       # Etapa 2: RetinaFace sobre los candidatos → rostros guardados
├── run.sh                    # Ejecuta ambas etapas en orden
├── custom_tracktrack.yaml    # Configuración del tracker de Ultralytics
├── requirements-track.txt    # Deps congeladas de venvs/track (incrusta el índice cu126 de torch)
├── requirements-faces.txt    # Deps congeladas de venvs/faces
├── README.md                 # Este fichero
├── .gitignore
│
│  # Presentes en local, NO versionados (ver la siguiente sección):
├── venvs/                    # los dos entornos virtuales (generados)
├── database/                 # salidas del pipeline (generadas)
├── yolo26m.pt                # pesos de YOLO (descargados)
├── yolo26m-reid.onnx         # codificador ReID (descargado)
├── tracked.mp4               # vídeo anotado opcional (generado)
└── *.mkv / *.mp4             # vídeo(s) de entrada (aportados por el usuario)
```

`face_recognition_pipeline.py` (si aún conservas una copia) es la versión
monoproceso **obsoleta**. Importa ambos frameworks en un mismo intérprete y provoca
segfaults en esta clase de máquina. Consérvalo solo como referencia; nunca lo
ejecutes.

---

## Ficheros que NO están en el repo (y de dónde salen)

| Fichero | Por qué falta | Cómo obtenerlo |
|---|---|---|
| Vídeo de entrada (`*.mkv`, `*.mp4`) | Grande; el metraje es específico de cada usuario | Coloca cualquier vídeo en la raíz del repo y ajusta `VIDEO_PATH` al principio de `tracking.py`. El clip de demostración se descargó de YouTube con `yt-dlp <url>`. |
| `yolo26m.pt` | Pesos del modelo, ~50 MB | **Se descarga automáticamente** por Ultralytics en la raíz del repo en la primera ejecución de `tracking.py`. También puedes usar tu propio checkpoint de YOLO26. |
| `yolo26m-reid.onnx` | Codificador ReID, ~50 MB | **Se descarga automáticamente** por Ultralytics la primera vez que el tracker corre con `with_reid: True`. También puede bajarse a mano desde [ultralytics/assets](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-reid.onnx) si la máquina no tiene salida a GitHub en tiempo de ejecución. |
| Pesos de RetinaFace | ~120 MB | **Se descargan automáticamente** en `~/.deepface/weights/` en la primera ejecución de `face_recognition.py`. Sobreviven a las reconstrucciones de los venvs. |
| Pesos de los modelos de DeepFace (ArcFace, etc.) | Grandes | Se descargan automáticamente en `~/.deepface/weights/` la primera vez que uses ese modelo. |
| `database/`, `venvs/` | Salidas generadas / entornos | Creados por el setup y por el propio pipeline. |

Por eso las primeras ejecuciones son más lentas que el régimen estable: espera unos
minutos de descargas más el calentamiento de los modelos antes de que se procese
cualquier fotograma. Esa pausa es normal, no es un cuelgue.

---

## Instalación (una vez por máquina)

Desde la raíz del repo (`/data/FaceRecognition`):

### Opción A — desde los ficheros de requisitos congelados (preferida)

```bash
python3 -m venv venvs/track
venvs/track/bin/pip install --upgrade pip
venvs/track/bin/pip install -r requirements-track.txt

python3 -m venv venvs/faces
venvs/faces/bin/pip install --upgrade pip
venvs/faces/bin/pip install -r requirements-faces.txt
```

`requirements-track.txt` incrusta `--extra-index-url https://download.pytorch.org/whl/cu126`
en su primera línea para que pip descargue la **compilación de PyTorch para CUDA
12.6**. Esa línea es estructural — no la elimines (PyPI a secas sirve un torch para
CUDA 13 que rompe el entorno).

### Opción B — desde cero (si los ficheros de requisitos están obsoletos)

```bash
# Entorno de tracking — SOLO familia PyTorch. TensorFlow nunca debe entrar en este venv.
python3 -m venv venvs/track
venvs/track/bin/pip install --upgrade pip
venvs/track/bin/pip install torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu126
venvs/track/bin/pip install ultralytics lap
venvs/track/bin/pip install onnxruntime          # ejecuta el codificador ReID .onnx
venvs/track/bin/pip uninstall -y opencv-python opencv-python-headless
venvs/track/bin/pip install opencv-python-headless

# Entorno de rostros — SOLO familia TensorFlow. torch nunca debe entrar en este venv.
python3 -m venv venvs/faces
venvs/faces/bin/pip install --upgrade pip
venvs/faces/bin/pip install "tensorflow[and-cuda]==2.21.*"
venvs/faces/bin/pip install retina-face tf-keras deepface
venvs/faces/bin/pip uninstall -y opencv-python opencv-python-headless
venvs/faces/bin/pip install opencv-python-headless
```

El par desinstalar/reinstalar de opencv al final de cada bloque es deliberado: varias
dependencias arrastran `opencv-python`, y cada venv debe contener **exactamente un**
build de cv2 — el headless (esto es un servidor; no hay pantalla).

**Sobre `onnxruntime`:** el codificador ReID solo corre en la etapa 1, así que
`onnxruntime` va **únicamente** en `venvs/track`. Usamos la variante CPU a propósito:
el tracker se invoca 1 vez cada 30 fotogramas sobre un puñado de recortes, el coste
es despreciable frente a YOLO26m, y `onnxruntime-gpu` arrastra sus propias ruedas
`nvidia-*` que compiten con el espacio de usuario CUDA 12.6 de torch. Si algún día
hace falta la variante GPU, instálala con el fichero congelado como restricción:

```bash
venvs/track/bin/pip install onnxruntime-gpu -c requirements-track.txt
```

**No ejecutes `pip install -U ultralytics`.** La versión congelada ya soporta
TrackTrack y ReID; una actualización puede re-resolver `torch` desde PyPI a secas
(rompe la regla 2) y reinstalar `opencv-python` no headless (rompe la regla 3).

### Verificación

```bash
venvs/track/bin/python -c "import torch, cv2; print(torch.__version__, torch.cuda.is_available(), cv2.__version__)"
# esperado: 2.13.0+cu126 True 5.0.x

venvs/track/bin/python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"

venvs/faces/bin/python -c "import os; os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'; import tensorflow as tf; from retinaface import RetinaFace; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
# esperado: 2.21.0 [PhysicalDevice(... GPU:0 ...)]
```

La primera comprobación es el canario: si deja de imprimir `2.13.0+cu126 True`,
algo movió el stack CUDA. No depures el venv: bórralo y reconstrúyelo desde el
fichero de requisitos (existen precisamente para que los venvs sean desechables).

---

## Ejecutar el pipeline

Coloca tu vídeo en la raíz del repo, ajusta `VIDEO_PATH` en `tracking.py` y luego:

```bash
./run.sh
```

que equivale a:

```bash
venvs/track/bin/python tracking.py          # etapa 1: vídeo → database/candidates/
venvs/faces/bin/python face_recognition.py  # etapa 2: candidatos → database/faces/
```

Nunca uses `source venv/bin/activate` + `python3` a secas — lanzar los scripts a
través de la ruta del intérprete es lo que garantiza que cada uno se ejecute sobre el
stack correcto.

### Etapa 1 — seguimiento (`tracking.py`)

El script **es dueño de su propio bucle de lectura** con `cv2.VideoCapture`; no le
pasa el vídeo entero a Ultralytics. Solo 1 de cada `DETECT_EVERY` fotogramas llega al
modelo: `cap.grab()` avanza los fotogramas descartados sin descodificarlos y
`cap.retrieve()` solo paga la descodificación de los que se conservan. `persist=True`
mantiene el estado del tracker entre llamadas sucesivas.

| Parámetro | Valor por defecto | Efecto |
|---|---|---|
| `VIDEO_PATH` | nombre del vídeo de demo | Vídeo de entrada |
| `TRACKER` | `custom_tracktrack.yaml` | Configuración del tracker de Ultralytics |
| `DETECT_EVERY` | `30` | Fotogramas del vídeo entre inferencias. `30` ≈ 1 inferencia por segundo a 30 fps |
| `MAX_CANDIDATES` | `10` | Recortes de persona guardados por ID de seguimiento |
| `SAMPLE_EVERY_STEPS` | `1` | Pasos **procesados** (no fotogramas del vídeo) entre recortes guardados de la misma persona. Con `1`, se guarda un recorte en cada inferencia |
| `SAVE_VIDEO` | `True` | Escribe un vídeo anotado para inspeccionar los IDs. Ponlo en `False` en producción |
| `OUTPUT_VIDEO` | `tracked.mp4` | Ruta del vídeo anotado |
| `FOURCC` | `mp4v` | Códec del vídeo anotado (ver notas más abajo) |

`SAMPLE_EVERY_STEPS` se cuenta en pasos procesados, **no** en fotogramas del vídeo:
como el modelo solo ve 1 de cada 30, contar fotogramas ya no tendría sentido. El
tiempo real entre recortes es `DETECT_EVERY × SAMPLE_EVERY_STEPS / fps` segundos.

### Etapa 2 — rostros (`face_recognition.py`)

Para cada persona, la etapa 2 pasa **todos** sus candidatos por RetinaFace, descarta
los que no superan el umbral de confianza y se queda con el mejor. Guarda el recorte
ajustado al rostro (no el cuerpo completo) en
`database/faces/<track_id>/face_of_<track_id>.jpg`.

| Parámetro | Valor por defecto | Efecto |
|---|---|---|
| `FACE_CONFIDENCE_THRESHOLD` | `0.92` | Confianza mínima de RetinaFace (escala 0–1, **no** porcentaje) |
| `FACE_MARGIN` | `0.25` | Margen alrededor de la caja del rostro, como fracción de su tamaño |
| `RANKING` | `"quality"` | `"quality"` ordena por tamaño del rostro ponderado por nitidez; `"score"` usa la confianza cruda |

El criterio por defecto es `área_del_rostro × √nitidez` (nitidez = varianza del
Laplaciano). La confianza de RetinaFace se satura cerca de 0.99 y apenas discrimina
entre recortes usables, mientras que la resolución del rostro es lo que realmente
determina si el embedding de DeepFace sirve.

Esto cuesta ~10× más inferencia de RetinaFace que la versión anterior, que paraba en
el primer rostro encontrado. Es un intercambio deliberado: la etapa 2 es offline y la
calidad del rostro condiciona todo lo que viene después.

Cada línea de log imprime `score`, dimensiones del rostro y nitidez del ganador. Si
los rostros ganadores bajan de forma consistente de ~60×60 px, ningún ajuste de
umbral lo va a arreglar: el problema es la resolución de la fuente o la distancia a
la cámara.

Los candidatos persisten entre ejecuciones; descomenta la línea
`rm -rf database/candidates` en `run.sh` para partir de cero cada vez.

### Configuración del tracker (`custom_tracktrack.yaml`)

```yaml
with_reid: True
model: yolo26m-reid.onnx
iou_weight: 0.3
reid_weight: 0.7
gmc_method: none
```

Los pesos por defecto de TrackTrack (`0.5` / `0.5`) asumen fotogramas consecutivos.
Con un salto de 30 fotogramas las personas se desplazan demasiado entre
observaciones y el término HMIoU aporta poco, así que el peso se desplaza hacia la
apariencia. `gmc_method: none` porque la cámara es fija y el flujo óptico disperso
entre fotogramas tan separados no estima nada útil (además ahorra unos ms por
llamada).

`model: auto` usa las características nativas del detector sin coste extra, pero es
notablemente menos discriminativo con este intervalo de muestreo. Tamaños
disponibles del codificador: `n`, `s`, `m`, `l`, `x`.

### Comprobaciones rápidas de la salida

```bash
ls database/faces/                    # una carpeta por persona detectada
find database/faces -name '*.jpg' | wc -l
```

### Notas y peculiaridades

- **Sin GUI en esta máquina.** Ambos venvs usan OpenCV headless, así que `show=True`
  en `model.track(...)` no puede abrir ninguna ventana. `tracking.py` escribe su
  propio vídeo anotado con `cv2.VideoWriter` (`SAVE_VIDEO = True`) en lugar de usar
  `save=True` de Ultralytics, que con una fuente de un solo fotograma genera un
  fichero por llamada en vez de un vídeo.
- **El vídeo anotado va a `src_fps / DETECT_EVERY` fps** (1 fps con una fuente de 30
  fps), para que la reproducción coincida con el tiempo real. Parece una
  presentación de diapositivas porque eso es literalmente todo lo que ve el tracker.
- **`mp4v` no se reproduce en navegadores ni en QuickTime.** El fichero es válido
  (VLC y mpv lo abren), pero es MPEG-4 Part 2. Para verlo en cualquier sitio,
  transcodifica con el ffmpeg del sistema:
  ```bash
  ffmpeg -i tracked.mp4 -vf "setpts=PTS/8" -r 8 -c:v libx264 -pix_fmt yuv420p \
    -movflags +faststart tracked_h264.mp4
  ```
  El `setpts` acelera ×8 para poder revisarlo sin aburrirse. `FOURCC = "avc1"` suele
  fallar en las ruedas de OpenCV para Linux (vienen sin x264 por licencia).
- **`cv2.VideoWriter_fourcc` no existe en OpenCV 5.** Usa
  `cv2.VideoWriter.fourcc(...)`, que funciona en 4.x y 5.x.
- **Uso compartido de la GPU:** `face_recognition.py` establece internamente
  `TF_FORCE_GPU_ALLOW_GROWTH=true` para que TensorFlow no acapare los 16 GB de la T4
  desde el principio. Mantén esa línea si editas el fichero.
- **Cualquier segfault, siempre** = un entorno se ha contaminado con el otro
  framework. No lo depures, reconstruye.

---

## Siguiente etapa: reconocimiento / embeddings (DeepFace)

`venvs/faces` ya contiene `deepface`. Cualquier script de reconocimiento se ejecuta
en ese mismo entorno y lee de `database/faces/`:

```bash
venvs/faces/bin/python your_embedding_script.py
```

Dentro de él, `DeepFace.represent(...)` / `DeepFace.verify(...)` con
`detector_backend="retinaface"` son seguros — todo el proceso es solo TF.

Como la etapa 2 ya entrega el rostro recortado con margen, puedes pasar
`detector_backend="skip"` si quieres evitar una segunda detección redundante.

---

## Las cuatro reglas de oro

1. **Un framework por venv y por proceso.** Los paquetes de la familia torch
   (`torch`, `torchvision`, `ultralytics`, `triton`) nunca entran en `venvs/faces`;
   los de la familia TF (`tensorflow`, `tf-keras`, `retina-face`, `deepface`,
   `mtcnn`) nunca entran en `venvs/track`. Cuidado con las instalaciones
   transitivas: `pip install retina-face` **es** `pip install tensorflow`;
   `pip install ultralytics` **es** `pip install torch`.
2. **Torch siempre desde el índice cu126**, nunca desde PyPI a secas.
3. **Exactamente un cv2 por venv, y es el headless.** Vuelve a ejecutar el par
   desinstalar/reinstalar de opencv después de cualquier comando pip que toque
   ultralytics, retina-face o deepface.
4. **Lanza mediante `venvs/<nombre>/bin/python`**, no con activate + `python3`.

Al re-congelar con `pip freeze > requirements-track.txt`, recuerda volver a añadir a
mano la línea `--extra-index-url` al principio: `pip freeze` no la conserva y es lo
único que impide que la próxima reconstrucción traiga un torch para CUDA 13.

---

## Uso responsable

Este pipeline procesa grabaciones de videovigilancia de personas identificables.
Se ejecutará únicamente sobre vídeo que tengamos derecho legal a procesar y cumple
con las leyes de privacidad y de datos biométricos aplicables.