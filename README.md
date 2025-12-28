## ImageEmbed Pro — Advanced AI-Powered Image Search

### 🎯 Overview
A professional-grade **Streamlit application** for semantic image search using **OpenAI CLIP (ViT-L-14)** embeddings. Users can:
- 📤 Upload multiple images to a managed gallery
- 🔍 Search by natural language descriptions (e.g., "red apple on wooden table")
- 🖼️ Find similar images using image-to-image search
- 📊 View similarity scores and confidence percentages
- ⚙️ Manage gallery with auto-cleanup and embedding validation

### ✨ Key Features
- **High-Accuracy Model**: OpenAI CLIP ViT-L-14 (upgraded from ViT-B-32)
- **Professional UI**: Modern interface with custom styling, tabs, and metrics
- **Robust Search**: Normalized cosine similarity with clamping for stability
- **Error Handling**: Validation of images, embeddings, and file operations
- **Scalable Gallery**: Auto-evicts oldest images when limit is reached
- **Search History**: Tracks recent queries for quick access
- **Performance**: Batch processing with optimized CUDA/CPU support

### 📋 Files
| File | Purpose |
|------|---------|
| `app.py` | Streamlit web UI with professional styling and tabs |
| `utils.py` | Embedding computation, validation, and I/O helpers |
| `detect.py` | CLI utilities for batch operations |
| `gallery/` | Local storage for uploaded images |
| `embeddings.json` | Pre-computed embeddings for fast search |

### 🚀 Installation

#### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

#### Setup
1. Clone and navigate to project:
   ```bash
   cd ImageEmbed-main
   ```

2. Create virtual environment:
   ```bash
   python -m venv myenv
   myenv\Scripts\activate  # Windows
   source myenv/bin/activate  # Mac/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   > **For GPU support**, install PyTorch with CUDA:
   > ```bash
   > pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   > ```

### 💻 Usage

#### Web Interface (Recommended)
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

**Workflow:**
1. **Upload & Manage Tab**: Upload images → Preview → Save & Compute Embeddings
2. **Search Tab**: Enter text query OR upload image → Adjust threshold → View results
3. **Gallery Tab**: Browse all images, search by filename, delete if needed

#### Command Line
```bash
# Compute embeddings for all images in gallery
python detect.py embed_all --batch 16

# Add a single image and compute embedding
python detect.py add_file --file path/to/image.jpg --batch 16

# Search by text
python detect.py search_text --query "golden retriever dog" --topk 6 --threshold 0.25

# Search by image
python detect.py search_image --file path/to/query.jpg --topk 6 --threshold 0.25
```

### ⚙️ Configuration

Edit variables in `utils.py`:
```python
GALLERY_DIR = "gallery"           # Where images are stored
EMBEDDINGS_FILE = "embeddings.json"  # Where embeddings are saved
MODEL_NAME = "ViT-L-14"           # CLIP model (options: ViT-B-32, ViT-L-14, ViT-bigG-14)
PRETRAINED = "openai"            # Pretrained weights source
```

Edit in `app.py`:
```python
MAX_GALLERY_SIZE = 100            # Max images before auto-eviction
EMBED_BATCH = 16                  # Batch size for embedding (lower = more memory efficient)
```

### 🔍 How It Works

1. **Image Upload**: User uploads images → Saved to `gallery/`
2. **Embedding Computation**: Each image processed by CLIP encoder → Vector embedding
3. **Storage**: Embeddings saved to `embeddings.json` (mapping filename → vector)
4. **Search Query**: User enters text/image → Converted to embedding
5. **Similarity Matching**: Cosine similarity computed between query and all images
6. **Results**: Sorted by similarity score → Displayed with confidence %

### 📊 Performance

| Model | Accuracy | Speed | Memory |
|-------|----------|-------|--------|
| ViT-B-32 | Good | Fast | Low |
| **ViT-L-14** ⭐ | **Excellent** | **Medium** | **Medium** |
| ViT-bigG-14 | Best | Slow | High |

**Recommended**: ViT-L-14 for best balance of accuracy and performance.

### 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| "No images in gallery" | Upload images first via Upload & Manage tab |
| "Missing embeddings" | Click "🔄 Recompute All" button in Upload tab |
| "Corrupted embeddings.json" | Delete the file; app will regenerate it |
| CUDA out of memory | Reduce `EMBED_BATCH` or use CPU-only |
| Slow startup | Model downloads on first run (~500MB); patience! |

### 📈 Accuracy Tips

1. **Use descriptive queries**: "a golden retriever running in a field" > "dog"
2. **Lower threshold**: More results but some may be less relevant
3. **Higher threshold**: Fewer but more accurate results (default 0.25)
4. **Recompute embeddings**: After model upgrade, recompute for consistency
5. **Quality images**: Better image quality → Better embeddings → More accurate search

### 🔐 Privacy & Security
- All data stored locally (gallery/ and embeddings.json)
- No cloud uploads or external API calls
- No user tracking or analytics

### 📝 License
MIT License - See LICENSE file

### 🙏 Credits
- **CLIP**: OpenAI Research
- **Streamlit**: Streamlit Inc.
- **PyTorch**: Meta AI
- **open-clip-torch**: OpenCLIP Contributors

