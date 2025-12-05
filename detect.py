#!/usr/bin/env python3
# detect.py - CLI utilities for embeddings and search
import os
import argparse
from PIL import Image
import numpy as np
import json

import utils

DEFAULT_GALLERY = utils.GALLERY_DIR
DEFAULT_EMBED = utils.EMBEDDINGS_FILE

def cmd_embed_all(args):
    total = utils.recompute_all_embeddings(batch_size=args.batch)
    print(f"Recomputed embeddings for {total} images. Saved to {DEFAULT_EMBED}")

def cmd_add_file(args):
    if not os.path.isfile(args.file):
        print("File not found:", args.file)
        return
    # copy into gallery
    import shutil
    base = os.path.basename(args.file)
    dest = os.path.join(DEFAULT_GALLERY, base)
    # ensure unique
    name, ext = os.path.splitext(base)
    i = 1
    while os.path.exists(dest):
        dest = os.path.join(DEFAULT_GALLERY, f"{name}_{i}{ext}")
        i += 1
    os.makedirs(DEFAULT_GALLERY, exist_ok=True)
    shutil.copy(args.file, dest)
    added = utils.append_embeddings_for_filenames([os.path.basename(dest)], batch_size=args.batch)
    print(f"Added {os.path.basename(dest)} to gallery, embeddings added: {added}")

def cmd_search_text(args):
    data = utils.safe_load_json(DEFAULT_EMBED)
    if not data:
        print("No embeddings found. Run embed_all or add_file first.")
        return
    names = list(data.keys())
    embs = np.array(list(data.values()), dtype=np.float32)
    qemb = utils.compute_text_embedding([args.query])[0]
    sims = utils.cosine_sim_matrix(qemb[None, :], embs)[0]
    idx_sorted = np.argsort(-sims)
    printed = 0
    for i in idx_sorted:
        if printed >= args.topk:
            break
        score = float(sims[i])
        if score < args.threshold:
            continue
        print(f"{names[i]}  score={score:.4f}")
        printed += 1
    if printed == 0:
        print("No results above threshold.")

def cmd_search_image(args):
    if not os.path.isfile(args.file):
        print("Query image not found:", args.file)
        return
    data = utils.safe_load_json(DEFAULT_EMBED)
    if not data:
        print("No embeddings found. Run embed_all or add_file first.")
        return
    names = list(data.keys())
    embs = np.array(list(data.values()), dtype=np.float32)
    with Image.open(args.file) as img:
        qemb = utils.compute_single_image_embedding(img.convert("RGB"))
    sims = utils.cosine_sim_matrix(qemb[None, :], embs)[0]
    idx_sorted = np.argsort(-sims)
    printed = 0
    for i in idx_sorted:
        if printed >= args.topk:
            break
        score = float(sims[i])
        if score < args.threshold:
            continue
        print(f"{names[i]}  score={score:.4f}")
        printed += 1
    if printed == 0:
        print("No results above threshold.")

def build_parser():
    p = argparse.ArgumentParser(description="detect.py utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_e = sub.add_parser("embed_all")
    p_e.add_argument("--batch", type=int, default=32)

    p_a = sub.add_parser("add_file")
    p_a.add_argument("--file", required=True)
    p_a.add_argument("--batch", type=int, default=32)

    p_st = sub.add_parser("search_text")
    p_st.add_argument("--query", required=True)
    p_st.add_argument("--topk", type=int, default=6)
    p_st.add_argument("--threshold", type=float, default=0.25)

    p_si = sub.add_parser("search_image")
    p_si.add_argument("--file", required=True)
    p_si.add_argument("--topk", type=int, default=6)
    p_si.add_argument("--threshold", type=float, default=0.25)

    return p

def main():
    p = build_parser()
    args = p.parse_args()
    if args.cmd == "embed_all":
        cmd_embed_all(args)
    elif args.cmd == "add_file":
        cmd_add_file(args)
    elif args.cmd == "search_text":
        cmd_search_text(args)
    elif args.cmd == "search_image":
        cmd_search_image(args)

if __name__ == "__main__":
    main()
