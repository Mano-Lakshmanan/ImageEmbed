# app.py — FIXED: immediate embeddings + predictable search + debug info
import os
import io
import streamlit as st
from PIL import Image
import numpy as np

import utils

# Config
GALLERY_DIR = utils.GALLERY_DIR
EMBEDDINGS_FILE = utils.EMBEDDINGS_FILE
MAX_GALLERY_SIZE = 50
EMBED_BATCH = 32

os.makedirs(GALLERY_DIR, exist_ok=True)

st.set_page_config(page_title="AI Image Search — Fixed", layout="wide")
st.title("AI Image Search — Fixed (Immediate embeddings & search)")

# session state for pending uploads
if "pending_uploads" not in st.session_state:
    st.session_state.pending_uploads = []  # list of {"name","bytes"}

# --- Sidebar: uploader ---
st.sidebar.header("Upload Images")
uploaded = st.sidebar.file_uploader(
    "Select images and click 'Add to pending'.",
    type=["jpg","jpeg","png","webp","bmp"],
    accept_multiple_files=True
)

if uploaded:
    if st.sidebar.button("Add to pending"):
        cnt = 0
        for f in uploaded:
            try:
                b = f.getbuffer().tobytes()
                st.session_state.pending_uploads.append({"name": f.name, "bytes": b})
                cnt += 1
            except Exception as e:
                st.sidebar.error(f"Failed reading {f.name}: {e}")
        st.sidebar.success(f"Added {cnt} file(s) to pending.")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.write(f"Pending: {len(st.session_state.pending_uploads)}")
if st.sidebar.button("Clear pending"):
    st.session_state.pending_uploads = []
    st.sidebar.info("Cleared pending")
    st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("Recompute all embeddings (overwrite)"):
    if not utils.list_images(GALLERY_DIR):
        st.sidebar.warning("Gallery empty.")
    else:
        with st.spinner("Recomputing all embeddings..."):
            total = utils.recompute_all_embeddings(batch_size=EMBED_BATCH)
        st.sidebar.success(f"Recomputed {total} embeddings.")
        st.rerun()

# --- Pending previews ---
st.header("Pending uploads (preview)")
if not st.session_state.pending_uploads:
    st.info("No pending uploads.")
else:
    cols = st.columns(3)
    for i, it in enumerate(st.session_state.pending_uploads):
        name = it.get("name", "unnamed")
        buf = it.get("bytes", b"")
        try:
            img = Image.open(io.BytesIO(buf)).convert("RGB")
            with cols[i % 3]:
                st.image(img, caption=name, use_column_width=True)
        except Exception as e:
            with cols[i % 3]:
                st.write(f"Preview failed: {e}")

# --- Save pending to gallery & immediate embedding ---
st.markdown("---")
c1, c2 = st.columns([1, 2])
with c1:
    if st.button("Save pending to gallery & embed immediately"):
        if not st.session_state.pending_uploads:
            st.warning("No pending uploads.")
        else:
            saved_names = []
            embeddings = utils.safe_load_json(EMBEDDINGS_FILE)
            for it in st.session_state.pending_uploads:
                name = it.get("name", "unnamed")
                buf = it.get("bytes", b"")
                safe_name = name.replace("/", "_").replace("\\", "_")
                base, ext = os.path.splitext(safe_name)
                dest = os.path.join(GALLERY_DIR, safe_name)
                i = 1
                while os.path.exists(dest):
                    dest = os.path.join(GALLERY_DIR, f"{base}_{i}{ext}")
                    i += 1
                try:
                    with open(dest, "wb") as wf:
                        wf.write(buf)
                    saved_names.append(os.path.basename(dest))
                except Exception as e:
                    st.error(f"Failed to save {name}: {e}")

            # enforce gallery size (auto-evict oldest)
            removed = utils.enforce_gallery_limit(GALLERY_DIR, embeddings, max_size=MAX_GALLERY_SIZE)
            if removed > 0:
                utils.safe_save_json(EMBEDDINGS_FILE, embeddings)
                st.info(f"Removed {removed} oldest images due to limit.")

            # compute embeddings immediately for saved files (force compute even if name exists)
            if saved_names:
                with st.spinner("Computing embeddings for saved images..."):
                    added = utils.append_embeddings_for_filenames(saved_names, batch_size=EMBED_BATCH)
                st.success(f"Saved {len(saved_names)} images and added {added} new embeddings.")
            else:
                st.warning("No files saved.")

            # clear pending and refresh
            st.session_state.pending_uploads = []
            st.rerun()

with c2:
    st.write(f"Gallery size: {len(utils.list_images(GALLERY_DIR))} images (limit {MAX_GALLERY_SIZE}).")
    st.write("After save the app computes embeddings immediately and appends to embeddings.json.")

# --- Search area ---
st.markdown("---")
st.header("Search / Find similar")

# load embeddings
emb_data = utils.safe_load_json(EMBEDDINGS_FILE)
if emb_data:
    image_names = list(emb_data.keys())
    image_embs = np.array(list(emb_data.values()), dtype=np.float32)
else:
    image_names = []
    image_embs = np.zeros((0, utils.get_embedding_dim()), dtype=np.float32)

# show debug info
st.subheader("Debug: Embeddings & gallery")
col_a, col_b, col_c = st.columns(3)
col_a.write(f"Files in gallery: **{len(utils.list_images(GALLERY_DIR))}**")
col_b.write(f"Embeddings stored: **{image_embs.shape[0]}**")
col_c.write("Tip: If a recently saved file is missing from embeddings, recompute or check logs.")

left, right = st.columns([2, 1])

with left:
    st.subheader("Text search")
    query_text = st.text_input("Enter text command (e.g., 'a photo of a dog running')")

    st.subheader("OR upload a query image for image-to-image search")
    query_img = st.file_uploader("Query image (optional)", type=["jpg","jpeg","png","webp","bmp"], key="query_search")

    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.2, 0.01)
    top_k = st.slider("Max results to show", 1, 20, 6)

    if st.button("Search now"):
        if image_embs.shape[0] == 0:
            st.warning("No embeddings present. Upload & save images first.")
        else:
            sims = None
            with st.spinner("Computing query embedding..."):
                if query_img is not None:
                    try:
                        qpil = Image.open(query_img).convert("RGB")
                        qemb = utils.compute_single_image_embedding(qpil)
                        sims = utils.cosine_sim_matrix(qemb[None, :], image_embs)[0]
                    except Exception as e:
                        st.error(f"Failed processing query image: {e}")
                elif query_text.strip():
                    qemb = utils.compute_text_embedding([query_text])[0]
                    sims = utils.cosine_sim_matrix(qemb[None, :], image_embs)[0]
                else:
                    st.info("Provide text or query image.")
                    sims = np.zeros((image_embs.shape[0],), dtype=np.float32)

            # show results
            idx_sorted = np.argsort(-sims)
            shown = 0
            st.subheader("Results")
            cols = st.columns(3)
            ci = 0
            for idx in idx_sorted:
                if shown >= top_k:
                    break
                score = float(sims[idx])
                if score < threshold:
                    continue
                fname = image_names[idx]
                with cols[ci]:
                    st.image(os.path.join(GALLERY_DIR, fname), caption=f"{fname}\nscore={score:.4f}", use_column_width=True)
                shown += 1
                ci = (ci + 1) % 3
            if shown == 0:
                st.info("No matches above threshold. Try lowering it or changing the query.")

with right:
    st.subheader("Find similar to a saved image (debug)")
    if not image_names:
        st.info("No saved images yet.")
    else:
        pick = st.selectbox("Choose a saved image to find similar images (includes itself)", options=image_names)
        if st.button("Find similar to selected"):
            # ensure embedding exists
            emb_dict = utils.safe_load_json(EMBEDDINGS_FILE)
            if pick not in emb_dict:
                st.error("Selected image has no embedding. Try 'Recompute all embeddings'.")
            else:
                qemb = np.array(emb_dict[pick], dtype=np.float32)
                sims = utils.cosine_sim_matrix(qemb[None, :], image_embs)[0]
                idx_sorted = np.argsort(-sims)
                st.subheader(f"Similar to {pick}")
                cols = st.columns(3)
                shown = 0
                ci = 0
                for idx in idx_sorted:
                    if shown >= 6:
                        break
                    score = float(sims[idx])
                    fname = image_names[idx]
                    with cols[ci]:
                        st.image(os.path.join(GALLERY_DIR, fname), caption=f"{fname}\nscore={score:.4f}", use_column_width=True)
                    shown += 1
                    ci = (ci + 1) % 3

st.markdown("---")
st.caption("If a saved image is not in embeddings, run 'Recompute all embeddings' in the sidebar. If embeddings.json is corrupted, delete it and recompute.")
