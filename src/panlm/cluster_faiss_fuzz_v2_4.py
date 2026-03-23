import argparse
import csv
import gc
import glob
import json
import math
import os
import time
from collections import OrderedDict, defaultdict

import faiss
import numba
import numpy as np
import pandas as pd
import torch

# --- Imports added for HDBSCAN integration ---
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

try:
    from sklearn.cluster import HDBSCAN

    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
# ---------------------------------------------


@numba.njit
def numba_find(parents, u):
    """Numba-compiled find operation with path compression."""
    if parents[u] != u:
        parents[u] = numba_find(parents, parents[u])
    return parents[u]


@numba.njit
def numba_union(parents, ranks, u, v):
    """Numba-compiled union operation by rank."""
    u_root = numba_find(parents, u)
    v_root = numba_find(parents, v)
    if u_root == v_root:
        return
    if ranks[u_root] < ranks[v_root]:
        parents[u_root] = v_root
    else:
        parents[v_root] = u_root
        if ranks[u_root] == ranks[v_root]:
            ranks[u_root] += 1


@numba.njit
def process_batch_numba(
    parents_2d, ranks_2d, similarities, indices, similarity_thresholds, start_idx
):
    """
    This is the JIT-compiled version of the main processing loop.
    """
    num_points_in_batch = similarities.shape[0]

    for i in range(num_points_in_batch):
        query_idx = start_idx + i

        # Loop through neighbors for the current point
        for k in range(1, similarities.shape[1]):  # Start from 1 to skip self
            neighbor_idx = indices[i, k]
            if neighbor_idx == -1:
                break  # Neighbors are sorted by distance; -1 means no more valid neighbors

            sim = similarities[i, k]
            num_thresholds_passed = np.searchsorted(
                similarity_thresholds, sim, side="right"
            )

            # Apply union operation for each threshold passed
            for j in range(num_thresholds_passed):
                numba_union(parents_2d[j], ranks_2d[j], query_idx, neighbor_idx)


def get_optimal_nlist(num_vectors):
    """
    Calculates nlist based on the formula: 4 * sqrt(N).
    Enforces constraints:
    1. nlist must be an integer.
    2. We need sufficient training data per cluster (Faiss warns if N < 39 * nlist).
    3. Minimum nlist is 1 (fallback to Flat behavior).
    """
    if num_vectors < 1000:
        return 2

    # Heuristic: 4 * sqrt(N)
    target_nlist = int(4 * math.sqrt(num_vectors))

    # Constraint: Ensure we have enough vectors to train the clusters.
    max_nlist = num_vectors // 39

    return max(1, min(target_nlist, max_nlist))


def load_data(path):
    """Load data from either .npz files or one or more .pt files."""
    if "*" in path:
        pt_files = glob.glob(path)
        if not pt_files:
            raise ValueError(f"No files found matching pattern: {path}")

        # Check file extension consistency
        if all(file.lower().endswith(".pt") for file in pt_files):
            print(f"Found {len(pt_files)} .pt files to combine")
            all_embeddings = OrderedDict()
            for pt_file in pt_files:
                data = torch.load(pt_file)
                all_embeddings.update(data)
            protein_ids = list(all_embeddings.keys())
            embeddings = np.stack(
                [tensor.cpu().numpy() for tensor in all_embeddings.values()]
            ).astype("float32")
            return embeddings, np.array(protein_ids)

        elif all(file.lower().endswith(".npz") for file in pt_files):
            all_embeddings_list = []
            all_protein_ids_list = []
            for npz_file in pt_files:
                data = np.load(npz_file)
                all_embeddings_list.append(data["embeddings"].astype("float32"))
                all_protein_ids_list.extend(data["protein_ids"])
            embeddings = np.vstack(all_embeddings_list)
            protein_ids = np.array(all_protein_ids_list)
            return embeddings, protein_ids
        else:
            raise ValueError("All files must be of the same type (.pt or .npz)")
    else:
        file_extension = os.path.splitext(path)[1].lower()
        if file_extension == ".npz":
            data = np.load(path)
            return data["embeddings"].astype("float32"), data["protein_ids"]
        elif file_extension == ".pt":
            data = torch.load(path)
            protein_ids = list(data.keys())
            embeddings = np.stack(
                [tensor.cpu().numpy() for tensor in data.values()]
            ).astype("float32")
            return embeddings, np.array(protein_ids)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}.")


def normal_pdf(x, mu, sigma):
    """Calculates the PDF of a normal distribution."""
    if sigma == 0:
        return 1.0 if x == mu else 0.0
    var = float(sigma) ** 2
    denom = (2 * math.pi * var) ** 0.5
    num = math.exp(-((float(x) - float(mu)) ** 2) / (2 * var))
    return num / denom


def fast_calculate_probabilities_optimized(columns, sim_thresholds, mean=None, sd=None):
    """
    Efficiently calculate probabilities (core/shell/cloud and cluster).
    """
    t1 = time.time()
    use_weights = mean is not None and sd is not None
    st_weights = {}
    if use_weights:
        print("Using normal distribution weights for thresholds")
        for st in sim_thresholds:
            st_weights[st] = normal_pdf(st, mean, sd)
        total_weight_sum = sum(st_weights.values())
        if total_weight_sum == 0:
            use_weights = False
    if not use_weights:
        total_weight_sum = float(len(sim_thresholds))
        for st in sim_thresholds:
            st_weights[st] = 1.0

    print(f"weight mean: {mean}, weight sd: {sd}\nall weights: {st_weights}")

    protein_ids_list = columns["protein_id"]
    unique_protein_ids, protein_indices = np.unique(
        protein_ids_list, return_inverse=True
    )
    n_proteins = len(unique_protein_ids)
    protein_to_idx = {pid: i for i, pid in enumerate(unique_protein_ids)}

    category_types = ["core", "shell", "cloud"]
    category_to_idx = {cat: i for i, cat in enumerate(category_types)}

    weighted_category_scores = np.zeros(
        (n_proteins, len(category_types)), dtype=np.float64
    )
    for st in sim_thresholds:
        weight = st_weights[st]
        cat_array = columns[f"ST_{st:.4f}_category"]
        for i, cat in enumerate(cat_array):
            weighted_category_scores[protein_indices[i], category_to_idx[cat]] += weight

    weighted_cluster_scores = [defaultdict(float) for _ in range(n_proteins)]
    for st in sim_thresholds:
        weight = st_weights[st]
        cluster_array = columns[f"ST_{st:.4f}"]
        for i, cluster_label in enumerate(cluster_array):
            weighted_cluster_scores[protein_indices[i]][cluster_label] += weight

    category_probs = weighted_category_scores / total_weight_sum
    category_probabilities_d = {
        pid: {cat: category_probs[idx, i] for i, cat in enumerate(category_types)}
        for pid, idx in protein_to_idx.items()
    }

    cluster_probabilities = {}
    for pid, idx in protein_to_idx.items():
        total_cluster_score = sum(weighted_cluster_scores[idx].values())
        if total_cluster_score > 0:
            cluster_probabilities[pid] = {
                k: v / total_cluster_score
                for k, v in weighted_cluster_scores[idx].items()
            }
        else:
            cluster_probabilities[pid] = {}

    final_rows = []
    for pid in protein_ids_list:
        c_probs = cluster_probabilities[pid]
        best_cluster, best_cluster_prob = (
            max(c_probs.items(), key=lambda item: item[1]) if c_probs else ("", 0.0)
        )
        cat_probs_dict = category_probabilities_d[pid]
        best_cat, best_cat_prob = max(cat_probs_dict.items(), key=lambda item: item[1])
        final_rows.append((best_cluster, best_cluster_prob, best_cat, best_cat_prob))

    (
        columns["cluster_id"],
        columns["cluster_prob"],
        columns["category"],
        columns["category_prob"],
    ) = zip(*final_rows)

    print(f"--- Fast probability calculation took: {time.time() - t1:.2f} seconds ---")
    results = {
        "weight_info": {"weighting_enabled": use_weights, "mean": mean, "std_dev": sd}
    }
    return results, columns


def get_available_memory_bytes():
    """Attempts to read available memory on Linux. Returns None if fails."""
    try:
        if os.path.isfile("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemAvailable" in line:
                        parts = line.split()
                        return int(parts[1]) * 1024
    except:
        pass
    return None


def cluster_faiss_parallel(
    path: str,
    similarity_thresholds,
    core_threshold: float,
    shell_threshold: float,
    cpu: bool,
    k: int,
    batch_size: int,
    mean: float,
    sd: float,
    pca_dim: int = None,
    algorithm: str = "fuzzy",
    min_cluster_size: int = 5,
    eps: float = 0.1,
):
    t_start = time.time()
    embeddings, protein_ids_np = load_data(path)
    num_points = embeddings.shape[0]
    original_dim = embeddings.shape[1]
    print(f"Time to load data: {time.time() - t_start:.2f} seconds ---")

    # Conditional PCA reduction
    if pca_dim is not None and pca_dim < original_dim:
        print(f"Applying PCA: reducing from {original_dim} to {pca_dim} dimensions")
        t_pca_start = time.time()

        pca = faiss.PCAMatrix(original_dim, pca_dim)
        pca.train(embeddings)
        if hasattr(pca, "eigen_vals") and pca.eigen_vals is not None:
            eigen_vals = pca.eigen_vals
            total_variance = np.sum(eigen_vals)
            variance_kept = np.sum(eigen_vals[:pca_dim])

            if total_variance > 1e-9:
                retained_variance_percent = (variance_kept / total_variance) * 100
                print(f"PCA Variance Retained: {retained_variance_percent:.2f}%")
        embeddings = pca.apply(embeddings)
        print(f"Time for PCA: {time.time() - t_pca_start:.2f} seconds ---")

    t_iter_start = time.time()
    # Normalize
    faiss.normalize_L2(embeddings)

    # Build FAISS Index
    index = None
    print("Initializing FAISS index...")
    N, d = embeddings.shape
    nlist = get_optimal_nlist(N)
    metric = faiss.METRIC_INNER_PRODUCT

    if not cpu:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexIVFFlat(res, d, nlist, metric)
    else:
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, metric)

    if not index.is_trained:
        t_train_start = time.time()
        index.train(embeddings)

    index.add(embeddings)
    print(f"Time for FAISS indexing: {time.time() - t_iter_start:.2f} seconds ---")

    columns = {
        "protein_id": protein_ids_np.tolist(),
        "strain": [
            str(pid).split("|")[1] if "|" in str(pid) else "" for pid in protein_ids_np
        ],
    }
    total_strains = len(set(s for s in columns["strain"] if s))
    all_strains_np = np.array(columns["strain"], dtype=object)

    if algorithm in ["hdbscan", "dbscan"]:
        if not HAS_HDBSCAN:
            raise ImportError(
                f"Sklearn clustering with HDBSCAN support is required for algorithm '{algorithm}'."
            )

        print(f"Running Custom Sparse Graph Pipeline for {algorithm.upper()}...")

        # --- SCIENTIFIC FIX: ADAPT K & INCREASE NPROBE ---
        # 1. Adapt K: User requested specific k.
        #    We ensure k >= min_cluster_size to satisfy sklearn's input requirements.
        hdbscan_k = max(k, min_cluster_size)
        print(f"Using k={hdbscan_k} for graph construction.")

        # 2. Increase nprobe: Default nprobe=1 is insufficient for high-dimensional IVF.
        #    It causes neighbors to be missed, resulting in empty rows in the sparse matrix.
        #    This triggers the 'ValueError: fewer than min_samples neighbors'.
        #    For k=100, we scale nprobe to ensure we visit enough clusters.
        index.nprobe = max(20, int(hdbscan_k / 2))
        print(
            f"OPTIMIZATION: Setting index.nprobe={index.nprobe} to ensure valid neighbor retrieval."
        )

        t_hdbscan_start = time.time()

        # --- Memory Optimized Sparse Graph Construction ---
        print("Constructing sparse KNN graph from FAISS index (Direct CSR Mode)...")

        total_edges = num_points * (hdbscan_k + 1)
        idx_dtype = np.uint32 if num_points < 4_000_000_000 else np.int64

        # Pre-allocation
        all_cols = np.empty(total_edges, dtype=idx_dtype)
        all_data = np.empty(total_edges, dtype=np.float32)
        indptr = np.zeros(num_points + 1, dtype=np.int64)

        num_batches_calc = math.ceil(num_points / batch_size)
        current_idx = 0

        for batch_idx in tqdm(range(num_batches_calc), desc="Fetching Neighbors"):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_points)
            if start >= end:
                continue

            n_samples_batch = end - start

            # Use the clamped hdbscan_k
            distances, indices = index.search(embeddings[start:end], hdbscan_k + 1)

            # Convert Similarity to Distance
            dist_values = 1.0 - distances
            dist_values = np.clip(dist_values, 1e-7, 2.0)

            flat_indices = indices.flatten().astype(idx_dtype)
            flat_distances = dist_values.flatten().astype(np.float32)

            # Identify valid edges (FAISS returns -1 if not found)
            valid_mask = (flat_indices >= 0) & (flat_indices < num_points)

            # Fill Data Arrays
            valid_count = np.sum(valid_mask)
            all_cols[current_idx : current_idx + valid_count] = flat_indices[valid_mask]
            all_data[current_idx : current_idx + valid_count] = flat_distances[
                valid_mask
            ]
            current_idx += valid_count

            # Update Indptr
            valid_mask_2d = valid_mask.reshape(n_samples_batch, hdbscan_k + 1)
            row_counts = np.sum(valid_mask_2d, axis=1)
            indptr[start + 1 : end + 1] = indptr[start] + np.cumsum(row_counts)

        # Truncate arrays to actual size
        all_cols = all_cols[:current_idx]
        all_data = all_data[:current_idx]

        print("Cleaning up dense memory (Embeddings & Index) to free RAM for CSR...")
        del embeddings
        del index
        if not cpu:
            try:
                res = faiss.StandardGpuResources()
                res.noTempMemory()
            except:
                pass
        gc.collect()

        print("Creating CSR Matrix (Optimized Constructor)...")
        knn_graph = csr_matrix(
            (all_data, all_cols, indptr), shape=(num_points, num_points)
        )

        del all_cols, all_data, indptr
        gc.collect()

        # --- Automatic Symmetry Detection Logic ---
        graph_size_bytes = (
            knn_graph.data.nbytes + knn_graph.indices.nbytes + knn_graph.indptr.nbytes
        )
        print(f"Current graph size in memory: {graph_size_bytes / 1024**3:.2f} GB")

        available_mem = get_available_memory_bytes()
        should_symmetrize = True

        if available_mem is not None:
            required_mem = graph_size_bytes * 3.0
            if required_mem > available_mem:
                print("WARNING: Insufficient memory detected for graph symmetrization.")
                print("Automatically skipping symmetrization to prevent OOM crash.")
                should_symmetrize = False

        if should_symmetrize:
            print("Symmetrizing graph...")
            try:
                gc.collect()
                knn_graph_t = knn_graph.transpose()
                knn_graph = knn_graph.maximum(knn_graph_t)
                del knn_graph_t
                gc.collect()
                print("Symmetrization complete.")
            except MemoryError:
                print("WARNING: Memory limit reached during graph symmetrization.")
                print("Proceeding with DIRECTED k-NN graph.")
                if "knn_graph_t" in locals():
                    del knn_graph_t
                gc.collect()
            except Exception as e:
                print(
                    f"WARNING: Symmetrization failed: {e}. Proceeding with directed graph."
                )
                gc.collect()

        # --- Connectivity Check & Patching ---
        # Both HDBSCAN and our DBSCAN emulation use the same sklearn engine which requires connectivity.
        print("Checking graph connectivity...")
        n_components, labels = connected_components(knn_graph, directed=False)

        if n_components > 1:
            print(
                f"Graph has {n_components} connected components. Patching to ensure connectivity..."
            )

            # Find representatives
            _, rep_indices = np.unique(labels, return_index=True)

            # Create bridge edges: chain representatives with high distance
            # Distance should be high (weak connection) to preserve cluster separation
            max_dist = knn_graph.data.max() if knn_graph.nnz > 0 else 1.0
            patch_dist = max(max_dist, 2.0)  # Cosine distance maxes at 2.0

            source_nodes = rep_indices[:-1]
            target_nodes = rep_indices[1:]
            data_patch = np.full(len(source_nodes), patch_dist, dtype=np.float32)

            rows = np.concatenate([source_nodes, target_nodes])
            cols = np.concatenate([target_nodes, source_nodes])
            data = np.concatenate([data_patch, data_patch])

            patch_matrix = coo_matrix((data, (rows, cols)), shape=knn_graph.shape)

            # Add patch to original graph
            knn_graph = knn_graph + patch_matrix
            print("Graph patched with minimal spanning bridges.")

        print(
            f"Fitting Sklearn {algorithm.upper()} (via HDBSCAN engine) (Input: {knn_graph.shape} Sparse Matrix)..."
        )
        # Ensure we don't pass min_samples > neighbors available
        eff_min_samples = min_cluster_size

        # Parameter Setup
        if algorithm == "dbscan":
            # Emulate DBSCAN by forcing a flat cut at 'eps'
            c_epsilon = eps
            c_method = (
                "leaf"  # 'leaf' usually closer to DBSCAN flat extraction than 'eom'
            )
            print(
                f"Configuring as DBSCAN: epsilon={c_epsilon}, min_samples={eff_min_samples}"
            )
        else:
            # Standard HDBSCAN behavior
            c_epsilon = eps
            c_method = "leaf"

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=eff_min_samples,
            metric="precomputed",
            cluster_selection_method=c_method,
            cluster_selection_epsilon=c_epsilon,
        )
        labels = clusterer.fit_predict(knn_graph)

        if hasattr(clusterer, "probabilities_"):
            probs = clusterer.probabilities_
        else:
            probs = np.ones_like(labels, dtype=float)
            probs[labels == -1] = 0.0

        print(f"Clustering finished in {time.time() - t_hdbscan_start:.2f}s.")
        print(f"Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")

        # -- Post Processing --
        columns["cluster_id"] = labels
        columns["cluster_prob"] = probs

        unique_labels, labels_inverse = np.unique(labels, return_inverse=True)
        cluster_data = {label: {"size": 0, "strains": set()} for label in unique_labels}

        for i, label in enumerate(labels):
            cluster_data[label]["size"] += 1
            strain = all_strains_np[i]
            if strain:
                cluster_data[label]["strains"].add(strain)

        label_to_props = {
            label: {
                "size": data["size"],
                "category": (
                    "core"
                    if (
                        len(data["strains"]) / total_strains if total_strains > 0 else 0
                    )
                    >= core_threshold
                    else (
                        "shell"
                        if (
                            len(data["strains"]) / total_strains
                            if total_strains > 0
                            else 0
                        )
                        >= shell_threshold
                        else "cloud"
                    )
                ),
            }
            for label, data in cluster_data.items()
        }

        prop_array_cat = np.array(
            [label_to_props[l]["category"] for l in unique_labels]
        )
        columns["category"] = prop_array_cat[labels_inverse]
        columns["category_prob"] = probs
        columns["ST_HDBSCAN"] = labels

        results = {"algorithm": algorithm.upper(), "n_clusters": len(unique_labels)}
        return columns, results

    else:
        # === ORIGINAL FUZZY ALGORITHM ===
        similarity_thresholds = np.sort(np.unique(similarity_thresholds))
        num_thresholds = len(similarity_thresholds)

        parents_2d = np.arange(num_points, dtype=np.int64)[None, :] * np.ones(
            (num_thresholds, 1), dtype=np.int64
        )
        ranks_2d = np.zeros((num_thresholds, num_points), dtype=np.int64)

        num_batches_calc = math.ceil(num_points / batch_size)

        t_search_start = time.time()
        total_faiss_search_time = 0
        for batch_idx in tqdm(range(num_batches_calc), desc="Clustering"):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_points)
            if start >= end:
                continue

            t_faiss_call_start = time.time()
            similarities, indices = index.search(embeddings[start:end], k + 1)
            total_faiss_search_time += time.time() - t_faiss_call_start

            process_batch_numba(
                parents_2d,
                ranks_2d,
                similarities,
                indices,
                similarity_thresholds,
                start,
            )

        t_numba_proc_time = time.time() - t_search_start - total_faiss_search_time
        print(
            f"Time for search & processing loop: {time.time() - t_search_start:.2f} seconds ---"
        )
        print(f"Time in actual FAISS search calls: {total_faiss_search_time:.2f} s")
        print(f"Time in Numba-compiled processing: {t_numba_proc_time:.2f} s")

        t_finalize_start = time.time()
        all_cluster_labels = {}
        for i, st_val in enumerate(similarity_thresholds):
            final_labels = np.array(
                [numba_find(parents_2d[i], j) for j in range(num_points)],
                dtype=np.int32,
            )
            all_cluster_labels[f"{st_val:.4f}"] = final_labels
        print(
            f"Time for cluster label finalization: {time.time() - t_finalize_start:.2f} seconds ---"
        )

        t_post_start = time.time()

        for st_val in similarity_thresholds:
            st_key = f"{st_val:.4f}"
            labels = all_cluster_labels[st_key]
            columns[f"ST_{st_key}"] = labels

            unique_labels, labels_inverse = np.unique(labels, return_inverse=True)
            cluster_data = {
                label: {"size": 0, "strains": set()} for label in unique_labels
            }
            for i, label in enumerate(labels):
                cluster_data[label]["size"] += 1
                strain = all_strains_np[i]
                if strain:
                    cluster_data[label]["strains"].add(strain)

            label_to_props = {
                label: {
                    "size": data["size"],
                    "category": (
                        "core"
                        if (
                            len(data["strains"]) / total_strains
                            if total_strains > 0
                            else 0
                        )
                        >= core_threshold
                        else (
                            "shell"
                            if (
                                len(data["strains"]) / total_strains
                                if total_strains > 0
                                else 0
                            )
                            >= shell_threshold
                            else "cloud"
                        )
                    ),
                }
                for label, data in cluster_data.items()
            }

            prop_array_size = np.array(
                [label_to_props[l]["size"] for l in unique_labels]
            )
            prop_array_cat = np.array(
                [label_to_props[l]["category"] for l in unique_labels]
            )

            columns[f"ST_{st_key}_size"] = prop_array_size[labels_inverse]
            columns[f"ST_{st_key}_category"] = prop_array_cat[labels_inverse]

        print(
            f"Time for post-clustering analysis: {time.time() - t_post_start:.2f} seconds ---"
        )

        results, columns = fast_calculate_probabilities_optimized(
            columns, similarity_thresholds, mean, sd
        )
        return columns, results


if __name__ == "__main__":
    t_script_start = time.time()
    parser = argparse.ArgumentParser(
        description="High-performance FAISS-based clustering with Numba.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to input embeddings (.npz or .pt), can use wildcards like '*.npz'",
    )
    parser.add_argument(
        "--output_file", required=True, help="Path to save the final CSV result file"
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="fuzzy",
        choices=["fuzzy", "hdbscan", "dbscan"],
        help="Clustering algorithm to use. 'dbscan' mode uses HDBSCAN with flat selection.",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=10,
        help="Minimum cluster size (HDBSCAN) or min_samples (DBSCAN).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.1,
        help="Epsilon parameter for DBSCAN (distance threshold). Default 0.1",
    )

    parser.add_argument(
        "--core_threshold",
        type=float,
        default=0.95,
        help="Minimum strain fraction for a cluster to be 'core'",
    )
    parser.add_argument(
        "--shell_threshold",
        type=float,
        default=0.15,
        help="Minimum strain fraction for a cluster to be 'shell'",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Use CPU for FAISS indexing instead of GPU"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=500,
        help="Number of nearest neighbors to consider for clustering",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10000, help="Batch size for FAISS search"
    )
    parser.add_argument(
        "--st",
        type=str,
        default="0.825,9.5,25",
        help="Similarity thresholds as a string: 'low,high,steps' (Fuzzy only)",
    )
    parser.add_argument(
        "--mean",
        type=float,
        default=0.86,
        help="Mean for normal distribution weights on thresholds. If unset, weighting is disabled. (Fuzzy only)",
    )
    parser.add_argument(
        "--sd",
        type=float,
        default=0.01,
        help="Standard deviation for normal distribution weights. If unset, weighting is disabled. (Fuzzy only)",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=None,
        help="Target dimension for PCA reduction. If not set, PCA is disabled.",
    )

    args = parser.parse_args()

    if args.algorithm == "fuzzy":
        low, high, steps = map(float, args.st.split(","))
        if args.mean is not None and args.sd is not None:
            print(
                "defining similarity thresholds based on normal distribution parameters"
            )
            min_st = max(0.0, args.mean - 3 * args.sd)
            max_st = min(1.0, args.mean + 3 * args.sd)
            similarity_thresholds = np.linspace(min_st, max_st, 15)
            print(
                f"Using similarity thresholds from {min_st:.4f} to {max_st:.4f} based on normal distribution"
            )
            print(f"similarity thresholds: {similarity_thresholds}")
        else:
            similarity_thresholds = (
                np.linspace(low, high, int(steps))
                if int(steps) > 1
                else np.array([low])
            )

        print("Pre-compiling Numba functions...")
        try:
            dummy_parents = np.arange(2, dtype=np.int64)[None, :]
            dummy_ranks = np.zeros((1, 2), dtype=np.int64)
            process_batch_numba(
                dummy_parents,
                dummy_ranks,
                np.random.rand(1, 2).astype("float32"),
                np.array([[0, 1]], dtype=np.int64),
                np.array([0.5]),
                0,
            )
            print("Compilation complete.")
        except Exception as e:
            print(
                f"Could not pre-compile Numba functions. This may happen on some systems. Continuing without pre-compilation. Error: {e}"
            )
    else:
        similarity_thresholds = None

    columns, results = cluster_faiss_parallel(
        args.input_file,
        similarity_thresholds,
        core_threshold=args.core_threshold,
        shell_threshold=args.shell_threshold,
        cpu=args.cpu,
        k=args.k,
        batch_size=args.batch_size,
        mean=args.mean,
        sd=args.sd,
        pca_dim=args.pca_dim,
        algorithm=args.algorithm,
        min_cluster_size=args.min_cluster_size,
        eps=args.eps,
    )

    t_write_start = time.time()
    df = pd.DataFrame(columns)

    id_cols = ["protein_id", "strain"]
    prob_cols = ["cluster_id", "cluster_prob", "category", "category_prob"]
    st_cols = sorted([col for col in df.columns if col.startswith("ST_")])
    final_order = id_cols + st_cols + prob_cols

    final_order = [c for c in final_order if c in df.columns]

    df = df[final_order]
    df.to_csv(args.output_file, index=False, float_format="%.4f")

    print(f"Time to write CSV: {time.time() - t_write_start:.2f} seconds ---")
    print(
        f"\nTotal script execution time: {time.time() - t_script_start:.2f} seconds ---"
    )
    print(f"Results saved to: {args.output_file}")
