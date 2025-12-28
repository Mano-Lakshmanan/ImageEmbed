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
MODEL_NAME = "ViT-B-32"  # Fast, good accuracy model (change to "ViT-L-14" for better accuracy, slower download)
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
        try:
            _model, _, _preprocess = open_clip.create_model_and_transforms(
                model_name=MODEL_NAME,
                pretrained=PRETRAINED
            )
            _model.to(_device)
            _model.eval()
        except Exception as e:
            print(f"Error loading model {MODEL_NAME}: {str(e)}")
            print("This usually happens when the model cache is corrupted.")
            print("Clearing cache and retrying with ViT-B-32...")
            import shutil
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                except:
                    pass
            # Fallback to smaller model
            _model, _, _preprocess = open_clip.create_model_and_transforms(
                model_name="ViT-B-32",
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
    Returns: (N, D) float32 numpy array with optimized accuracy
    """
    model = _model
    preprocess = _preprocess
    device = _device

    if not pil_images:
        return np.zeros((0, get_embedding_dim()), dtype=np.float32)

    embeddings = []
    n = len(pil_images)
    i = 0
    
    while i < n:
        batch_imgs = pil_images[i: i + batch_size]
        try:
            tensors = torch.stack([preprocess(img).to(device) for img in batch_imgs], dim=0)
            with torch.no_grad():
                if _USE_AMP:
                    with torch.cuda.amp.autocast():
                        feats = model.encode_image(tensors)
                else:
                    feats = model.encode_image(tensors)
                
                # Improved normalization for accuracy
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-10)
                feats = feats.detach().cpu().numpy().astype(np.float32)
                
                # Ensure normalized
                feats = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-10)
                embeddings.append(feats)
        except Exception as e:
            # Skip problematic images
            continue
        
        i += batch_size
    
    if embeddings:
        result = np.vstack(embeddings).astype(np.float32)
        # Final normalization pass
        result = result / (np.linalg.norm(result, axis=-1, keepdims=True) + 1e-10)
        return result
    else:
        return np.zeros((0, get_embedding_dim()), dtype=np.float32)

def compute_single_image_embedding(pil_image: Image.Image) -> np.ndarray:
    return compute_batch_image_embeddings([pil_image], batch_size=1).squeeze(0)

def compute_text_embedding(texts: List[str]) -> np.ndarray:
    """
    Compute text embeddings with high accuracy and proper normalization.
    """
    if not texts or not texts[0].strip():
        return np.zeros((1, get_embedding_dim()), dtype=np.float32)
    
    device = _device
    try:
        tokens = open_clip.tokenize(texts).to(device)
        with torch.no_grad():
            if _USE_AMP:
                with torch.cuda.amp.autocast():
                    feats = _model.encode_text(tokens)
            else:
                feats = _model.encode_text(tokens)
            
            # Improved normalization
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-10)
            result = feats.detach().cpu().numpy().astype(np.float32)
            
            # Ensure normalized
            result = result / (np.linalg.norm(result, axis=-1, keepdims=True) + 1e-10)
            return result
    except Exception as e:
        # Fallback: return zero vector if tokenization fails
        return np.zeros((len(texts), get_embedding_dim()), dtype=np.float32)

def cosine_sim_matrix(query_emb: np.ndarray, gallery_embs: np.ndarray) -> np.ndarray:
    """
    Compute normalized cosine similarity matrix with high precision.
    Handles edge cases and ensures stable, accurate results.
    """
    # Normalize query embedding
    q_norm = np.linalg.norm(query_emb, axis=-1, keepdims=True)
    q_norm = np.where(q_norm < 1e-10, 1.0, q_norm)  # Avoid division by zero
    q = query_emb / q_norm
    
    # Normalize gallery embeddings
    g_norm = np.linalg.norm(gallery_embs, axis=-1, keepdims=True)
    g_norm = np.where(g_norm < 1e-10, 1.0, g_norm)  # Avoid division by zero
    g = gallery_embs / g_norm
    
    # Compute similarity with high precision
    sims = np.dot(q, g.T)
    
    # Clamp to [-1, 1] range to handle numerical errors
    sims = np.clip(sims, -1.0, 1.0)
    
    return sims

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
    Includes robust error handling and validation.
    """
    data = safe_load_json(EMBEDDINGS_FILE)
    before = set(data.keys())

    to_compute = []
    names = []
    
    for fname in filenames:
        if fname in data:
            continue
        
        path = os.path.join(GALLERY_DIR, fname)
        
        # Validate file exists
        if not os.path.isfile(path):
            continue
        
        try:
            with Image.open(path) as img:
                # Validate image
                if img.size[0] < 32 or img.size[1] < 32:
                    continue  # Skip very small images
                
                img_rgb = img.convert("RGB").copy()
                to_compute.append(img_rgb)
                names.append(fname)
        except Exception as e:
            # Skip unreadable files silently
            continue

    if not to_compute:
        return 0

    try:
        embs = compute_batch_image_embeddings(to_compute, batch_size=batch_size)
        
        # Validate embeddings
        for n, emb in zip(names, embs):
            # Check for NaN or invalid embeddings
            if not np.all(np.isfinite(emb)):
                continue  # Skip invalid embeddings
            
            data[n] = emb.tolist()
        
        safe_save_json(EMBEDDINGS_FILE, data)
    except Exception as e:
        # If batch fails, try individual processing
        for img, fname in zip(to_compute, names):
            try:
                emb = compute_batch_image_embeddings([img], batch_size=1)[0]
                if np.all(np.isfinite(emb)):
                    data[fname] = emb.tolist()
            except Exception:
                continue
        
        safe_save_json(EMBEDDINGS_FILE, data)
    
    after = set(data.keys())
    return len(after - before)

def recompute_all_embeddings(batch_size: int = 32) -> int:
    """
    Recompute embeddings for all images in gallery.
    Includes validation and robust error handling.
    """
    files = list_images(GALLERY_DIR)
    
    if not files:
        return 0
    
    imgs = []
    names = []
    
    for fname in files:
        path = os.path.join(GALLERY_DIR, fname)
        
        try:
            with Image.open(path) as img:
                # Validate image
                if img.size[0] < 32 or img.size[1] < 32:
                    continue  # Skip very small images
                
                imgs.append(img.convert("RGB").copy())
                names.append(fname)
        except Exception:
            continue

    if not imgs:
        return 0

    try:
        embs = compute_batch_image_embeddings(imgs, batch_size=batch_size)
        
        # Build embeddings dict with validation
        data = {}
        for n, e in zip(names, embs):
            if np.all(np.isfinite(e)):  # Only keep valid embeddings
                data[n] = e.tolist()
        
        safe_save_json(EMBEDDINGS_FILE, data)
        return len(data)
    except Exception:
        return 0

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
