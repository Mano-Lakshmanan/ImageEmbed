import os
import streamlit as st
import numpy as np
from PIL import Image
import utils

# ---------------- CONFIG ----------------
GALLERY_DIR = utils.GALLERY_DIR
EMBEDDINGS_FILE = utils.EMBEDDINGS_FILE
MAX_GALLERY_SIZE = 100
EMBED_BATCH = 16

os.makedirs(GALLERY_DIR, exist_ok=True)

st.set_page_config(
    page_title="ImageEmbed",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============ PROFESSIONAL ENTERPRISE DESIGN ============
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    margin: 0;
    padding: 0;
}

header, footer {visibility: hidden;}

.block-container {
    padding: 2.5rem 3.5rem !important;
    max-width: 1600px !important;
    margin: 0 auto !important;
}

.stApp {
    background: #fafbfc;
    color: #1a1a1a;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

h1 {
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.25rem !important;
    color: #000000 !important;
    letter-spacing: -0.03em !important;
}

h2 {
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 1rem !important;
    color: #000000 !important;
}

h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    margin-bottom: 0.75rem !important;
    color: #1a1a1a !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-size: 0.9rem !important;
}

p, .stMarkdown {
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    color: #5a5a5a !important;
}

.stTabs {
    margin-top: -10px !important;
    border-bottom: 2px solid #e5e5e5 !important;
    padding-bottom: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    color: #8a8a8a !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    padding: 0.85rem 1.5rem !important;
    border-bottom: 3px solid transparent !important;
    transition: all 0.25s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    font-size: 0.85rem !important;
}

.stTabs [aria-selected="true"] {
    color: #000000 !important;
    border-bottom: 3px solid #000000 !important;
}

.stTextInput input, .stNumberInput input, .stSelectbox select {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    border: 1px solid #d0d0d0 !important;
    border-radius: 6px !important;
    padding: 10px 12px !important;
    font-size: 0.95rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
}

.stTextInput input:focus, .stNumberInput input:focus {
    border: 1.5px solid #000000 !important;
    box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.06) !important;
}

.stButton button {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 6px !important;
    padding: 10px 18px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    border: 1px solid #d0d0d0 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
    transition: all 0.25s ease !important;
    white-space: nowrap !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

.stButton button:hover {
    background-color: #f0f0f0 !important;
    border-color: #a0a0a0 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12) !important;
    transform: translateY(-1px) !important;
}

.stButton button:active {
    transform: translateY(0) !important;
}

.btn-secondary {
    background-color: #f0f0f0 !important;
    color: #1a1a1a !important;
}

.btn-secondary:hover {
    background-color: #e5e5e5 !important;
}

.stFileUploader {
    background: #ffffff !important;
    border: 1px solid #d0d0d0 !important;
    border-radius: 8px !important;
    padding: 2rem !important;
    transition: all 0.25s ease !important;
}

.stFileUploader:hover {
    border-color: #a0a0a0 !important;
    background-color: #fcfcfc !important;
}

.stFileUploader section {
    background: transparent !important;
    color: #000000 !important;
}

.stFileUploader section span {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    color: #000000 !important;
}

.stFileUploader section p {
    color: #000000 !important;
}

.stFileUploader section div {
    color: #000000 !important;
}

.stFileUploader button {
    background-color: #f5f5f5 !important;
    color: #000000 !important;
    border-radius: 6px !important;
    padding: 8px 14px !important;
    border: 1px solid #d0d0d0 !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
    transition: all 0.2s ease !important;
}

.stFileUploader button:hover {
    background-color: #1a1a1a !important;
}

.card {
    background: #ffffff !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 8px !important;
    padding: 1.5rem !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    transition: all 0.25s ease !important;
}

.card:hover {
    border-color: #d0d0d0 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
}

.gallery-item {
    background: #ffffff !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    transition: all 0.25s ease !important;
}

.gallery-item:hover {
    border-color: #d0d0d0 !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1) !important;
    transform: translateY(-2px) !important;
}

.gallery-item img {
    width: 100% !important;
    height: 200px !important;
    object-fit: cover !important;
}

.stMetric {
    background: #ffffff !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 8px !important;
    padding: 1.25rem !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    color: #000000 !important;
}

.stMetric label {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #000000 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

.stMetric span {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #000000 !important;
}

.stMetric p {
    color: #000000 !important;
}

.stAlert {
    border-radius: 6px !important;
    border-left: 4px solid !important;
    padding: 1rem !important;
    margin: 1rem 0 !important;
}

.stSuccess {
    background-color: #f0fdf4 !important;
    border-color: #22c55e !important;
    color: #166534 !important;
}

.stWarning {
    background-color: #fffbeb !important;
    border-color: #f59e0b !important;
    color: #92400e !important;
}

.stInfo {
    background-color: #f0f9ff !important;
    border-color: #0ea5e9 !important;
    color: #0c4a6e !important;
}

.stError {
    background-color: #fef2f2 !important;
    border-color: #ef4444 !important;
    color: #7f1d1d !important;
}

.big-image {
    border-radius: 8px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
}

.big-image img {
    max-width: 100% !important;
    height: auto !important;
    display: block !important;
}

.header-section {
    padding-bottom: 2rem !important;
    border-bottom: 1px solid #e5e5e5 !important;
    margin-bottom: 2rem !important;
}

.subtitle {
    font-size: 0.95rem !important;
    color: #8a8a8a !important;
    font-weight: 400 !important;
    margin-top: 0.5rem !important;
    letter-spacing: 0.02em !important;
}

.stat-section {
    display: flex !important;
    gap: 1rem !important;
    margin: 1.5rem 0 !important;
}

.stat-card {
    flex: 1 !important;
    background: #ffffff !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 8px !important;
    padding: 1.25rem !important;
}

.stat-label {
    font-size: 0.8rem !important;
    color: #8a8a8a !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    font-weight: 600 !important;
    margin-bottom: 0.5rem !important;
}

.stat-value {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #000000 !important;
}

.confidence-bar {
    width: 100% !important;
    height: 4px !important;
    background: #e5e5e5 !important;
    border-radius: 2px !important;
    overflow: hidden !important;
    margin-top: 0.5rem !important;
}

.confidence-fill {
    height: 100% !important;
    background: #000000 !important;
    transition: width 0.4s ease !important;
}
/* Underline for metric value (0/100) */
[data-testid="stMetricValue"] {
    color: #000000 !important;
    text-decoration: underline !important;
}
/* Uploaded file name text color */
.stFileUploader span,
.stFileUploader p,
.stFileUploader div {
    color: #000000 !important;
}

/* Uploaded file list background */
.stFileUploader section {
    background-color: #ffffff !important;
}



</style>
""", unsafe_allow_html=True)

# ============ SESSION STATE ============
if "pending_uploads" not in st.session_state:
    st.session_state.pending_uploads = []

# ============ HEADER ============
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.markdown("""
    <div class='header-section'>
        <h1>ImageEmbed</h1>
        <p class='subtitle'>AI-powered semantic vector search for images</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    count = len(utils.list_images(GALLERY_DIR))
    st.metric("Images", f"{count}/{MAX_GALLERY_SIZE}")

# ============ TABS ============
tab_upload, tab_search, tab_gallery = st.tabs(["Upload", "Search", "Gallery"])

# ============ UPLOAD TAB ============
with tab_upload:
    st.markdown("### Upload Images")
    st.markdown("Add images to your collection. Each image is automatically indexed with embeddings for similarity search.")
    st.markdown("")

    col1, col2 = st.columns([2, 1])
    with col1:
        files = st.file_uploader(
            "Select images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if files:
            for f in files:
                st.session_state.pending_uploads.append(
                    {"name": f.name, "bytes": f.getbuffer().tobytes()}
                )
            st.success(f"Ready to save: {len(files)} image(s)")

    st.markdown("")
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_b:
        if st.button("Save to Gallery", use_container_width=True):
            if not st.session_state.pending_uploads:
                st.warning("No images selected")
            else:
                with st.spinner("Processing embeddings..."):
                    embeddings = utils.safe_load_json(EMBEDDINGS_FILE)
                    saved = []

                    for item in st.session_state.pending_uploads:
                        path = os.path.join(GALLERY_DIR, item["name"])
                        with open(path, "wb") as f:
                            f.write(item["bytes"])
                        saved.append(item["name"])

                    utils.enforce_gallery_limit(GALLERY_DIR, embeddings, MAX_GALLERY_SIZE)
                    utils.append_embeddings_for_filenames(saved, EMBED_BATCH)

                    st.session_state.pending_uploads.clear()
                    st.cache_data.clear()
                
                st.success(f"Success: {len(saved)} image(s) saved")
                st.rerun()

# ============ SEARCH TAB ============
with tab_search:
    st.markdown("### Vector Search")
    st.markdown("Search your gallery using natural language. Results are ranked by semantic similarity to your query.")
    st.markdown("")

    emb_data = utils.safe_load_json(EMBEDDINGS_FILE)

    if not emb_data:
        st.info("No images in gallery. Start by uploading some images.")
    else:
        names = list(emb_data.keys())
        embs = np.array(list(emb_data.values()), dtype=np.float32)

        # Search interface
        col_q, col_s = st.columns([3, 1])
        
        with col_q:
            query = st.text_input(
                "Enter search query",
                placeholder="Example: dog, sunset, person...",
                label_visibility="collapsed"
            )

        with col_s:
            search_clicked = st.button("Search", use_container_width=True)

        if search_clicked and query.strip():
            with st.spinner("Searching vector space..."):
                q = utils.compute_text_embedding([query])[0]
                sims = utils.cosine_sim_matrix(q[None, :], embs)[0]

                # Get top 3 results
                top_indices = np.argsort(sims)[::-1][:3]
                top_scores = sims[top_indices]
                top_names = [names[i] for i in top_indices]

                st.markdown("---")
                st.markdown("### Search Results")
                
                for rank, (idx, score, name) in enumerate(zip(top_indices, top_scores, top_names), 1):
                    img_path = os.path.join(GALLERY_DIR, name)
                    
                    if os.path.exists(img_path):
                        col_rank, col_img, col_info = st.columns([0.5, 2, 1.5])
                        
                        with col_rank:
                            st.markdown(f"""
                            <div style='text-align: center; padding-top: 1rem;'>
                                <p style='font-size: 2rem; font-weight: 700; color: #000000;'>{rank}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_img:
                            st.image(img_path, use_container_width=True)

                        
                        with col_info:
                            confidence = score * 100
                            st.markdown(f"""
                            <div class='card'>
                                <p class='stat-label'>Similarity Score</p>
                                <p class='stat-value'>{confidence:.1f}%</p>
                                <div class='confidence-bar'>
                                    <div class='confidence-fill' style='width: {confidence}%;'></div>
                                </div>
                                <p style='margin-top: 1rem; font-size: 0.85rem; color: #8a8a8a;'>{name}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("")

# ============ GALLERY TAB ============
with tab_gallery:
    st.markdown("### Gallery Management")
    st.markdown("View and manage all images in your collection.")
    st.markdown("")

    files = utils.list_images(GALLERY_DIR)
    emb = utils.safe_load_json(EMBEDDINGS_FILE)

    if not files:
        st.info("Gallery is empty. Start by uploading some images.")
    else:
        # Gallery statistics
        st.markdown("#### Collection Statistics")
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Total Images", len(files))
        with col_stat2:
            total_size = sum(os.path.getsize(os.path.join(GALLERY_DIR, f)) for f in files if os.path.exists(os.path.join(GALLERY_DIR, f)))
            size_mb = total_size / (1024 * 1024)
            st.metric("Collection Size", f"{size_mb:.1f} MB")
        with col_stat3:
            remaining = MAX_GALLERY_SIZE - len(files)
            st.metric("Remaining Capacity", remaining)

        st.markdown("")
        st.markdown("#### Images")

        # Gallery grid
        cols = st.columns(4)
        for i, f in enumerate(files):
            img_path = os.path.join(GALLERY_DIR, f)
            if not os.path.exists(img_path):
                continue

            with cols[i % 4]:
                st.markdown(f'<div class="gallery-item">', unsafe_allow_html=True)
                st.image(img_path, use_column_width=True)
                
                # File info
                file_size = os.path.getsize(img_path) / 1024
                st.markdown(f"<small style='color: #8a8a8a;'>{f}</small>", unsafe_allow_html=True)
                st.markdown(f"<small style='color: #b0b0b0;'>{file_size:.1f} KB</small>", unsafe_allow_html=True)
                
                if st.button("Remove", key=f, use_container_width=True):
                    os.remove(img_path)
                    emb.pop(f, None)
                    utils.safe_save_json(EMBEDDINGS_FILE, emb)
                    st.cache_data.clear()
                    st.info("Image removed")
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        # Bulk operations
        st.markdown("---")
        st.markdown("#### Bulk Operations")
        col_del_a, col_del_b, col_del_c = st.columns([2, 1, 2])
        with col_del_b:
            if st.button("Clear Collection", use_container_width=True):
                for f in utils.list_images(GALLERY_DIR):
                    path = os.path.join(GALLERY_DIR, f)
                    if os.path.exists(path):
                        os.remove(path)

                utils.safe_save_json(EMBEDDINGS_FILE, {})
                st.cache_data.clear()
                st.success("Collection cleared")
                st.rerun()