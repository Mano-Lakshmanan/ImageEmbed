## AI Image Search (Streamlit + CLIP)

## Overview
A Streamlit app to upload images, store them in a local `gallery/`, compute CLIP embeddings, and search by text or by image. The gallery enforces a size limit (default 50) and auto-evicts oldest images when needed.

## Files
- `app.py` - Streamlit frontend.
- `utils.py` - shared helpers for IO and embedding computation.
- `detect.py` - CLI utilities (embed, add, search).
- `gallery/` - saved user images (created at runtime).
- `embeddings.json` - mapping filename -> embedding vector (auto-generated).

## Install
1. Create a venv and activate it.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
