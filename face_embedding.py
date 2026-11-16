import cv2
import numpy as np

EMBED_CHUNK = 64


def unit_embedding(values):
    emb = np.array(values, dtype=np.float32)
    return emb / (np.linalg.norm(emb) + 1e-9)


def model_input(face, input_shape):
    target_size = (input_shape[1], input_shape[0])
    factor = min(target_size[0] / face.shape[0], target_size[1] / face.shape[1])
    img = cv2.resize(face, (int(face.shape[1] * factor), int(face.shape[0] * factor)))
    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]
    img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2),
                       (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), "constant")
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)
    img = np.asarray(img, dtype=np.float32)
    if img.max() > 1:
        img = (img.astype(np.float32) / 255.0).astype(np.float32)
    return img


def embed_faces(forward, faces, input_shape, chunk=EMBED_CHUNK):
    embeddings = []
    for start in range(0, len(faces), chunk):
        part = [model_input(face, input_shape) for face in faces[start:start + chunk]]
        batch = np.zeros((chunk,) + part[0].shape, dtype=np.float32)
        batch[:len(part)] = part
        rows = np.asarray(forward(batch), dtype=np.float32).reshape(chunk, -1)
        embeddings.extend(unit_embedding(row) for row in rows[:len(part)])
    return embeddings
