# config.py — Centralized configuration for ImageEmbed Pro
"""
Configuration file for ImageEmbed Pro.
Modify these settings to customize behavior.
"""

# ========================
# FILE & STORAGE
# ========================
GALLERY_DIR = "gallery"                    # Directory to store uploaded images
EMBEDDINGS_FILE = "embeddings.json"        # File to store computed embeddings
MAX_GALLERY_SIZE = 100                     # Max images before auto-eviction (oldest deleted)

# ========================
# CLIP MODEL SETTINGS
# ========================
MODEL_NAME = "ViT-B-32"                    # Fast, good accuracy (quick download)
# Options: "ViT-B-32" (fast), "ViT-L-14" (slow download but better), "ViT-bigG-14" (very slow)
PRETRAINED = "openai"                      # Pretrained weights source

# ========================
# PROCESSING SETTINGS
# ========================
EMBED_BATCH_SIZE = 16                      # Batch size for embedding computation
# Lower = more memory efficient, slower
# Higher = faster, more memory usage
# Recommended: 8-32 depending on GPU/CPU

MIN_IMAGE_SIZE = 32                        # Minimum image dimensions (pixels)
# Images smaller than this will be skipped

# ========================
# SEARCH SETTINGS
# ========================
DEFAULT_SIMILARITY_THRESHOLD = 0.15        # Default minimum similarity score (0.0-1.0)
DEFAULT_TOP_K = 12                         # Default number of results to show
DEFAULT_SORT_ORDER = "Highest Match"       # Default result sorting

# ========================
# UI/UX SETTINGS
# ========================
PAGE_TITLE = "ImageEmbed Pro — AI Image Search"
PAGE_ICON = "🎯"
LAYOUT = "wide"
SHOW_DEBUG_INFO = True                     # Show embeddings count & debug metrics

# ========================
# SUPPORTED IMAGE FORMATS
# ========================
SUPPORTED_FORMATS = ("jpg", "jpeg", "png", "webp", "bmp")

# ========================
# PERFORMANCE
# ========================
USE_GPU = True                             # Use CUDA if available (set False to force CPU)
ENABLE_AMP = True                          # Enable automatic mixed precision (GPU only)

# ========================
# ADVANCED
# ========================
EMBEDDING_DIM_DEFAULT = 768                # Default embedding dimension (ViT-L-14 uses 768)
SIMILARITY_CLAMP_MIN = -1.0                # Min cosine similarity (for numerical stability)
SIMILARITY_CLAMP_MAX = 1.0                 # Max cosine similarity (for numerical stability)
EPSILON = 1e-10                            # Small value to avoid division by zero
