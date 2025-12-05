# utils.py
import os
import json
from typing import List
from PIL import Image
import numpy as np
import torch
import open_clip
import time

# -----------------------
# Config / constants
# -----------------------
GALLERY_DIR = "gallery"
EMBEDDINGS_FILE = "embeddings.json"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

# -----------------------
# Model load (cached at module import)
# -----------------------
_model = None
_preprocess = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_USE_AMP = _device.type == "cuda"

def load_model_and_preprocess():
    global _model, _preprocess
    if _model is None or _preprocess is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            model_name=MODEL_NAME,
            pretrained=PRETRAINED
        )
        _model.to(_device)
        _model.eval()
    return _model, _preprocess

# eager load
_model, _preprocess = load_model_and_preprocess()

# -----------------------
# I/O helpers
# -----------------------
def safe_load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def safe_save_json(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)

def list_images(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    if not os.path.isdir(folder):
        return []
    return [f for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]

def get_embedding_dim() -> int:
    # Try to infer embedding dim from model attributes
    try:
        if hasattr(_model, "visual") and hasattr(_model.visual, "embed_dim"):
            return _model.visual.embed_dim
    except Exception:
        pass
    # fallback common CLIP dims
    return 512

# -----------------------
# Embedding computation
# -----------------------
def compute_batch_image_embeddings(pil_images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
    """
    Compute embeddings (normalized) for a list of PIL images.
    Returns: (N, D) float32 numpy array
    """
    model = _model
    preprocess = _preprocess
    device = _device

    embeddings = []
    n = len(pil_images)
    i = 0
    while i < n:
        batch_imgs = pil_images[i: i + batch_size]
        tensors = torch.stack([preprocess(img).to(device) for img in batch_imgs], dim=0)
        with torch.no_grad():
            if _USE_AMP:
                with torch.cuda.amp.autocast():
                    feats = model.encode_image(tensors)
            else:
                feats = model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.detach().cpu().numpy()
        embeddings.append(feats)
        i += batch_size
    if embeddings:
        return np.vstack(embeddings).astype(np.float32)
    else:
        return np.zeros((0, get_embedding_dim()), dtype=np.float32)

def compute_single_image_embedding(pil_image: Image.Image) -> np.ndarray:
    return compute_batch_image_embeddings([pil_image], batch_size=1).squeeze(0)

def compute_text_embedding(texts: List[str]) -> np.ndarray:
    device = _device
    tokens = open_clip.tokenize(texts).to(device)
    with torch.no_grad():
        if _USE_AMP:
            with torch.cuda.amp.autocast():
                feats = _model.encode_text(tokens)
        else:
            feats = _model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy().astype(np.float32)

def cosine_sim_matrix(query_emb: np.ndarray, gallery_embs: np.ndarray) -> np.ndarray:
    # numeric stable normalized dot product
    q = query_emb / (np.linalg.norm(query_emb, axis=-1, keepdims=True) + 1e-10)
    g = gallery_embs / (np.linalg.norm(gallery_embs, axis=-1, keepdims=True) + 1e-10)
    return np.dot(q, g.T)

# -----------------------
# File save utilities
# -----------------------
def save_uploaded_file(uploaded_file, dest_folder: str) -> str:
    """
    Save streamlit uploaded file-like object to dest_folder and return saved filename.
    Ensures unique filename by numeric suffix if needed.
    """
    os.makedirs(dest_folder, exist_ok=True)
    raw_name = uploaded_file.name
    safe_name = raw_name.replace("/", "_").replace("\\", "_")
    base, ext = os.path.splitext(safe_name)
    dest_path = os.path.join(dest_folder, safe_name)
    i = 1
    while os.path.exists(dest_path):
        dest_path = os.path.join(dest_folder, f"{base}_{i}{ext}")
        i += 1
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return os.path.basename(dest_path)

def append_embeddings_for_filenames(filenames: List[str], batch_size: int = 32) -> int:
    """
    Compute embeddings for new filenames (in GALLERY_DIR) and append to EMBEDDINGS_FILE.
    Returns the number of new embeddings added.
    """
    data = safe_load_json(EMBEDDINGS_FILE)
    before = set(data.keys())

    to_compute = []
    names = []
    for fname in filenames:
        if fname in data:
            continue
        path = os.path.join(GALLERY_DIR, fname)
        try:
            with Image.open(path) as img:
                to_compute.append(img.convert("RGB").copy())
                names.append(fname)
        except Exception as e:
            # skip unreadable files
            continue

    if not to_compute:
        return 0

    embs = compute_batch_image_embeddings(to_compute, batch_size=batch_size)
    for n, emb in zip(names, embs):
        data[n] = emb.tolist()

    safe_save_json(EMBEDDINGS_FILE, data)
    after = set(data.keys())
    return len(after - before)

def recompute_all_embeddings(batch_size: int = 32) -> int:
    files = list_images(GALLERY_DIR)
    imgs = []
    names = []
    for fname in files:
        path = os.path.join(GALLERY_DIR, fname)
        try:
            with Image.open(path) as img:
                imgs.append(img.convert("RGB").copy())
                names.append(fname)
        except Exception:
            continue

    if not imgs:
        return 0

    embs = compute_batch_image_embeddings(imgs, batch_size=batch_size)
    data = {n: e.tolist() for n, e in zip(names, embs)}
    safe_save_json(EMBEDDINGS_FILE, data)
    return len(data)

# -----------------------
# Gallery limit enforcement (auto-evict oldest)
# -----------------------
def enforce_gallery_limit(folder: str, embeddings: dict, max_size: int = 50) -> int:
    """
    Ensure the folder has at most max_size images.
    If > max_size, delete oldest files (by modification time) until within limit.
    This function mutates the embeddings dict (removes keys for deleted files).
    Returns the number of removed files.
    """
    files = list_images(folder)
    if len(files) <= max_size:
        return 0

    files_with_mtime = [(f, os.path.getmtime(os.path.join(folder, f))) for f in files]
    files_with_mtime.sort(key=lambda x: x[1])  # oldest first

    removed = 0
    while len(files_with_mtime) > max_size:
        fname, _ = files_with_mtime.pop(0)
        try:
            os.remove(os.path.join(folder, fname))
            if fname in embeddings:
                del embeddings[fname]
            removed += 1
        except Exception:
            pass

    return removed
