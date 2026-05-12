#!/usr/bin/env python3
# Simple FAISS clustering across multiple similarity thresholds with category assignment and timing

import argparse
import glob
import os
import time
from collections import defaultdict

import faiss
import numpy as np
import torch


def load_data(path):
    """
    Load embeddings and protein IDs from .npz or .pt (with optional glob patterns).
    """
    start = time.time()
    if "*" in path:
        files = sorted(glob.glob(path))
        if not files:
            raise FileNotFoundError(f"No files match pattern: {path}")
        arrays, ids = [], []
        for f in files:
            data = np.load(f)
            arrays.append(data["embeddings"])
            ids.extend(data["protein_ids"])
        emb = np.vstack(arrays)
    else:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npz":
            data = np.load(path)
            emb = data["embeddings"]
            ids = list(data["protein_ids"])
        elif ext == ".pt":
            data = torch.load(path, map_location="cpu")
            ids = list(data.keys())
            emb = np.stack([t.cpu().numpy() for t in data.values()])
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    print(
        f"Loaded embeddings in {time.time() - start:.2f}s: N={emb.shape[0]}, D={emb.shape[1]}"
    )
    return emb, ids


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        ru, rv = self.find(u), self.find(v)
        if ru == rv:
            return
        if self.rank[ru] < self.rank[rv]:
            self.parent[ru] = rv
        else:
            self.parent[rv] = ru
            if self.rank[ru] == self.rank[rv]:
                self.rank[ru] += 1


def cluster_at_thresholds(
    emb: np.ndarray, thresholds: np.ndarray, cpu: bool, k: int, batch_size: int
):
    """
    Perform FAISS-based clustering for each similarity threshold.
    Returns dict: {threshold: [cluster_labels]}
    """
    N, D = emb.shape
    if cpu:
        emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(D)
        index.add(emb_norm)
    else:
        emb_norm = emb.copy()
        faiss.normalize_L2(emb_norm)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, D)
        index.add(emb_norm)
    # print(f"Built FAISS index in {time.time() - t0:.2f}s")

    results = {}
    for tau in thresholds:
        uf = UnionFind(N)
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            sims, inds = index.search(emb_norm[start_idx:end_idx], k + 1)
            for i in range(end_idx - start_idx):
                idx = start_idx + i
                for sim, j in zip(sims[i, 1:], inds[i, 1:], strict=False):
                    if sim >= tau:
                        uf.union(idx, j)
        labels = [uf.find(i) for i in range(N)]
        results[tau] = labels
        # print(f"Clustered at τ={tau:.4f} in {time.time() - t1:.2f}s")
    return results


def assign_categories(
    cluster_labels: list, protein_ids: list, core_thresh: float, shell_thresh: float
):
    """
    Assign core/shell/cloud per protein based on cluster membership.
    """
    strains = [pid.split("|")[1] if "|" in pid else "" for pid in protein_ids]
    unique_strains = set(strains)
    total = len(unique_strains)
    cluster_to_inds = defaultdict(list)
    for idx, cl in enumerate(cluster_labels):
        cluster_to_inds[cl].append(idx)
    cluster_cat = {}
    for cl, inds in cluster_to_inds.items():
        frac = len({strains[i] for i in inds}) / total if total > 0 else 0.0
        if frac >= core_thresh:
            cat = "core"
        elif frac >= shell_thresh:
            cat = "shell"
        else:
            cat = "cloud"
        cluster_cat[cl] = cat
    return [cluster_cat[cl] for cl in cluster_labels]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster embeddings at multiple similarity thresholds and assign categories with timing"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to embeddings (.npz, .pt, or glob)",
    )
    parser.add_argument("--start", type=float, default=0.5)
    parser.add_argument("--stop", type=float, default=1.0)
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=100000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--core_threshold", type=float, default=0.95)
    parser.add_argument("--shell_threshold", type=float, default=0.15)
    parser.add_argument("--output_clusters", required=True, help="Path to output clusters CSV file")
    parser.add_argument("--output_categories", required=True, help="Path to output categories CSV file")
    args = parser.parse_args()

    thresholds = np.linspace(args.start, args.stop, args.num)

    # Load embeddings
    emb, ids = load_data(args.input)

    # Perform clustering
    cluster_dict = cluster_at_thresholds(
        emb, thresholds, cpu=args.cpu, k=args.k, batch_size=args.batch
    )

    # Write clusters CSV
    t2 = time.time()
    with open(args.output_clusters, "w") as f:
        header = ["protein_id"] + [f"ST_{round(t,4)}" for t in thresholds]
        f.write(",".join(header) + "\n")
        for i, pid in enumerate(ids):
            row = [pid] + [str(cluster_dict[t][i]) for t in thresholds]
            f.write(",".join(row) + "\n")
    print(f"Wrote clusters to {args.output_clusters} in {time.time() - t2:.2f}s")

    # Compute and write categories CSV
    categories_dict = {}
    for t in thresholds:
        categories_dict[t] = assign_categories(
            cluster_dict[t], ids, args.core_threshold, args.shell_threshold
        )
    t3 = time.time()
    with open(args.output_categories, "w") as f:
        header = ["protein_id"] + [f"ST_{round(t,4)}_category" for t in thresholds]
        f.write(",".join(header) + "\n")
        for i, pid in enumerate(ids):
            row = [pid] + [categories_dict[t][i] for t in thresholds]
            f.write(",".join(row) + "\n")
    print(f"Wrote categories to {args.output_categories} in {time.time() - t3:.2f}s")
