import numpy as np
import faiss
from tqdm import tqdm
import math
import os
import glob
import torch
from collections import OrderedDict, defaultdict
import time
import pandas as pd
import argparse
import numba
from scipy.sparse import csr_matrix
from sklearn.cluster import HDBSCAN, DBSCAN


@numba.njit
def numba_find(parents, u):
    if parents[u] != u:
        parents[u] = numba_find(parents, parents[u])
    return parents[u]


@numba.njit
def numba_union(parents, ranks, u, v):
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
    num_points_in_batch = similarities.shape[0]
    for i in range(num_points_in_batch):
        query_idx = start_idx + i
        for k_idx in range(1, similarities.shape[1]):
            neighbor_idx = indices[i, k_idx]
            if neighbor_idx == -1:
                break
            sim = similarities[i, k_idx]
            num_thresholds_passed = np.searchsorted(
                similarity_thresholds, sim, side="right"
            )
            for j in range(num_thresholds_passed):
                numba_union(parents_2d[j], ranks_2d[j], query_idx, neighbor_idx)


def load_data(path):
    if "*" in path:
        pt_files = glob.glob(path)
        if not pt_files:
            raise ValueError(f"No files found matching pattern: {path}")
        if all(f.lower().endswith(".pt") for f in pt_files):
            all_embeddings = OrderedDict()
            for pt_file in pt_files:
                data = torch.load(pt_file)
                all_embeddings.update(data)
            protein_ids = list(all_embeddings.keys())
            embeddings = np.stack(
                [tensor.cpu().numpy() for tensor in all_embeddings.values()]
            ).astype("float32")
            return embeddings, np.array(protein_ids)
        elif all(f.lower().endswith(".npz") for f in pt_files):
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
            raise ValueError("Inconsistent file types in glob pattern.")
    else:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npz":
            data = np.load(path)
            return data["embeddings"].astype("float32"), data["protein_ids"]
        elif ext == ".pt":
            data = torch.load(path)
            protein_ids = list(data.keys())
            embeddings = np.stack(
                [tensor.cpu().numpy() for tensor in data.values()]
            ).astype("float32")
            return embeddings, np.array(protein_ids)
    raise ValueError(f"Unsupported format: {path}")


def normal_pdf(x, mu, sigma):
    if sigma == 0:
        return 1.0 if x == mu else 0.0
    var = float(sigma) ** 2
    denom = (2 * math.pi * var) ** 0.5
    num = math.exp(-((float(x) - float(mu)) ** 2) / (2 * var))
    return num / denom


def fast_calculate_probabilities_optimized(columns, sim_thresholds, mean=None, sd=None):
    use_weights = mean is not None and sd is not None
    st_weights = {}

    if use_weights:
        for st in sim_thresholds:
            st_weights[st] = normal_pdf(st, mean, sd)
        total_weight_sum = sum(st_weights.values())
        if total_weight_sum == 0:
            use_weights = False
    if not use_weights:
        total_weight_sum = float(len(sim_thresholds))
        for st in sim_thresholds:
            st_weights[st] = 1.0

    protein_ids_list = columns["protein_id"]
    unique_pids, protein_indices = np.unique(protein_ids_list, return_inverse=True)
    n_proteins = len(unique_pids)
    protein_to_idx = {pid: i for i, pid in enumerate(unique_pids)}

    category_types = ["core", "shell", "cloud"]
    cat_to_idx = {cat: i for i, cat in enumerate(category_types)}
    weighted_cat_scores = np.zeros((n_proteins, len(category_types)), dtype=np.float64)

    for st in sim_thresholds:
        weight = st_weights[st]
        cat_array = columns[f"ST_{st:.4f}_category"]
        for i, cat in enumerate(cat_array):
            weighted_cat_scores[protein_indices[i], cat_to_idx[cat]] += weight

    weighted_cluster_scores = [defaultdict(float) for _ in range(n_proteins)]
    for st in sim_thresholds:
        weight = st_weights[st]
        cluster_array = columns[f"ST_{st:.4f}"]
        for i, cl in enumerate(cluster_array):
            weighted_cluster_scores[protein_indices[i]][cl] += weight

    cat_probs = weighted_cat_scores / total_weight_sum
    final_rows = []
    for pid in protein_ids_list:
        idx = protein_to_idx[pid]
        c_dict = weighted_cluster_scores[idx]
        best_cl, cl_score = (
            max(c_dict.items(), key=lambda x: x[1]) if c_dict else ("", 0.0)
        )
        best_cl_prob = cl_score / total_weight_sum

        row_cat_probs = cat_probs[idx]
        best_cat_idx = np.argmax(row_cat_probs)
        final_rows.append(
            (
                best_cl,
                best_cl_prob,
                category_types[best_cat_idx],
                row_cat_probs[best_cat_idx],
            )
        )

    (
        columns["cluster_id"],
        columns["cluster_prob"],
        columns["category"],
        columns["category_prob"],
    ) = zip(*final_rows)
    return {
        "weight_info": {"weighting_enabled": use_weights, "mean": mean, "std_dev": sd}
    }, columns


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
    clust_method: str = "fuzzy",
    min_cluster_size: int = 5,
    eps: float = 0.1,
):
    t_start = time.time()
    embeddings, protein_ids_np = load_data(path)
    num_points = embeddings.shape[0]
    original_dim = embeddings.shape[1]

    if pca_dim is not None and pca_dim < original_dim:
        pca = faiss.PCAMatrix(original_dim, pca_dim)
        pca.train(embeddings)
        embeddings = pca.apply(embeddings)

    faiss.normalize_L2(embeddings)

    if not cpu:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, embeddings.shape[1])
    else:
        index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    columns = {
        "protein_id": protein_ids_np.tolist(),
        "strain": [
            str(pid).split("|")[1] if "|" in str(pid) else "" for pid in protein_ids_np
        ],
    }
    total_strains = len(set(s for s in columns["strain"] if s))
    all_strains_np = np.array(columns["strain"], dtype=object)

    if clust_method in ["hdbscan", "dbscan"]:
        # Convert Cosine Distance (eps) to L2 Distance for the precomputed matrix
        l2_eps = math.sqrt(2 * eps)
        print(
            f"Converting input Cosine eps ({eps:.4f}) to L2 distance threshold: {l2_eps:.4f}"
        )

        print(f"Generating sparse distance matrix (k={k})...")
        t_search_start = time.time()

        rows = []
        cols = []
        data = []

        num_batches = math.ceil(num_points / batch_size)
        for i in tqdm(range(num_batches), desc="FAISS Search"):
            start = i * batch_size
            end = min(start + batch_size, num_points)
            sims, idxs = index.search(embeddings[start:end], k)

            dists = np.sqrt(np.maximum(0, 2 - 2 * sims))

            for j in range(end - start):
                query_idx = start + j
                neighbor_indices = idxs[j]
                neighbor_dists = dists[j]

                mask = neighbor_indices != -1
                valid_indices = neighbor_indices[mask]
                valid_dists = neighbor_dists[mask]

                rows.extend([query_idx] * len(valid_indices))
                cols.extend(valid_indices)
                data.extend(valid_dists)

        print(
            f"FAISS search & distance conversion: {time.time() - t_search_start:.2f}s"
        )

        sparse_dist = csr_matrix((data, (rows, cols)), shape=(num_points, num_points))
        sparse_dist = sparse_dist.maximum(sparse_dist.transpose())

        if clust_method == "hdbscan":
            print(
                f"Fitting HDBSCAN on sparse distance matrix (eps={l2_eps:.4f}, min_samples={min_cluster_size})"
            )
            t_hdb_fit = time.time()
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric="precomputed",
                allow_single_cluster=True,
                min_samples=1,
                cluster_selection_epsilon=l2_eps,
                n_jobs=-3,
            )
            labels = clusterer.fit_predict(sparse_dist)
            probs = clusterer.probabilities_
            print(f"HDBSCAN fit completed in {time.time() - t_hdb_fit:.2f}s")

        elif clust_method == "dbscan":
            print(
                f"Fitting DBSCAN on sparse distance matrix (eps={l2_eps:.4f}, min_samples={min_cluster_size})"
            )
            t_db_fit = time.time()
            clusterer = DBSCAN(
                eps=l2_eps,
                min_samples=min_cluster_size,
                metric="precomputed",
                n_jobs=-3,
            )
            labels = clusterer.fit_predict(sparse_dist)
            # Standard DBSCAN provides hard clustering without probabilities.
            # We enforce 1.0 for clustered entities and 0.0 for noise elements prior to noise reassignment.
            probs = np.where(labels == -1, 0.0, 1.0)
            print(f"DBSCAN fit completed in {time.time() - t_db_fit:.2f}s")

        # Assign unique cluster IDs to noise points (-1)
        noise_mask = labels == -1
        num_noise = np.sum(noise_mask)
        if num_noise > 0:
            max_label = labels.max()
            start_id = 0 if max_label == -1 else max_label + 1
            labels[noise_mask] = np.arange(start_id, start_id + num_noise)

        columns["cluster_id"] = labels
        columns["cluster_prob"] = probs

        unique_labels, labels_inverse = np.unique(labels, return_inverse=True)
        cluster_strains = {label: set() for label in unique_labels}

        for i, label in enumerate(labels):
            strain = all_strains_np[i]
            if strain:
                cluster_strains[label].add(strain)

        label_to_cat = {}
        for label in unique_labels:
            strain_frac = (
                len(cluster_strains[label]) / total_strains if total_strains > 0 else 0
            )
            if strain_frac >= core_threshold:
                label_to_cat[label] = "core"
            elif strain_frac >= shell_threshold:
                label_to_cat[label] = "shell"
            else:
                label_to_cat[label] = "cloud"

        columns["category"] = np.array([label_to_cat[l] for l in labels])
        columns["category_prob"] = probs
        # columns[f"ST_{algorithm.upper()}"] = labels

        return columns, {
            "clust_method": clust_method.upper(),
            "n_clusters": len(unique_labels),
        }

    else:
        # ORIGINAL FUZZY ALGORITHM
        similarity_thresholds = np.sort(np.unique(similarity_thresholds))
        num_thresholds = len(similarity_thresholds)
        parents_2d = np.arange(num_points, dtype=np.int64)[None, :] * np.ones(
            (num_thresholds, 1), dtype=np.int64
        )
        ranks_2d = np.zeros((num_thresholds, num_points), dtype=np.int64)

        num_batches = math.ceil(num_points / batch_size)
        for i in tqdm(range(num_batches), desc="Fuzzy Clustering"):
            start = i * batch_size
            end = min(start + batch_size, num_points)
            sims, idxs = index.search(embeddings[start:end], k + 1)
            process_batch_numba(
                parents_2d, ranks_2d, sims, idxs, similarity_thresholds, start
            )

        all_labels = {}
        for i, st in enumerate(similarity_thresholds):
            all_labels[f"{st:.4f}"] = np.array(
                [numba_find(parents_2d[i], j) for j in range(num_points)],
                dtype=np.int32,
            )

        for st_val in similarity_thresholds:
            st_key = f"{st_val:.4f}"
            labels = all_labels[st_key]
            columns[f"ST_{st_key}"] = labels
            unique_labels, inv = np.unique(labels, return_inverse=True)
            counts = np.bincount(inv)

            cluster_strains = {l: set() for l in unique_labels}
            for i, l in enumerate(labels):
                s = all_strains_np[i]
                if s:
                    cluster_strains[l].add(s)

            l_to_props = {}
            for l in unique_labels:
                s_frac = (
                    len(cluster_strains[l]) / total_strains if total_strains > 0 else 0
                )
                cat = (
                    "core"
                    if s_frac >= core_threshold
                    else ("shell" if s_frac >= shell_threshold else "cloud")
                )
                l_to_props[l] = (counts[np.where(unique_labels == l)[0][0]], cat)

            columns[f"ST_{st_key}_size"] = np.array([l_to_props[l][0] for l in labels])
            columns[f"ST_{st_key}_category"] = np.array(
                [l_to_props[l][1] for l in labels]
            )

        res_meta, columns = fast_calculate_probabilities_optimized(
            columns, similarity_thresholds, mean, sd
        )
        return columns, res_meta


if __name__ == "__main__":
    t_script_start = time.time()
    parser = argparse.ArgumentParser(
        description="High-performance FAISS-based clustering."
    )
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--clust_method",
        type=str,
        default="fuzzy",
        choices=["fuzzy", "hdbscan", "dbscan"],
    )
    parser.add_argument("--min_cluster_size", type=int, default=2)
    parser.add_argument(
        "--eps",
        type=float,
        default=0.075,
        help="Epsilon parameter strictly used for DBSCAN algorithm",
    )
    parser.add_argument("--core_threshold", type=float, default=0.95)
    parser.add_argument("--shell_threshold", type=float, default=0.15)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--k", type=int, default=2000, help="k-NN for HDBSCAN, DBSCAN, or Fuzzy"
    )
    parser.add_argument("--batch_size", type=int, default=150000)
    parser.add_argument("--st", type=str, default="0.8,0.9,10")
    parser.add_argument("--mean", type=float, default=None)
    parser.add_argument("--sd", type=float, default=None)
    parser.add_argument("--pca_dim", type=int, default=None)

    args = parser.parse_args()
    sim_thresholds = None

    if args.clust_method == "fuzzy":
        print("test fuzz")
        if args.mean is not None and args.sd is not None:
            print(f"fuzzt clust, mean: {args.mean}, sd: {args.sd}")
            min_st, max_st = (
                max(0.0, args.mean - 3 * args.sd),
                min(1.0, args.mean + 3 * args.sd),
            )
            sim_thresholds = np.linspace(min_st, max_st, 20)
        else:
            low, high, steps = map(float, args.st.split(","))
            sim_thresholds = np.linspace(low, high, int(steps))

        dummy_p = np.arange(2, dtype=np.int64)[None, :]
        dummy_r = np.zeros((1, 2), dtype=np.int64)
        process_batch_numba(
            dummy_p,
            dummy_r,
            np.random.rand(1, 2).astype("float32"),
            np.array([[0, 1]], dtype=np.int64),
            np.array([0.5]),
            0,
        )

    columns, results = cluster_faiss_parallel(
        args.input_file,
        sim_thresholds,
        args.core_threshold,
        args.shell_threshold,
        args.cpu,
        args.k,
        args.batch_size,
        args.mean,
        args.sd,
        args.pca_dim,
        args.clust_method,
        args.min_cluster_size,
        args.eps,
    )

    df = pd.DataFrame(columns)
    id_cols = ["protein_id", "strain"]
    prob_cols = ["cluster_id", "cluster_prob", "category", "category_prob"]
    st_cols = sorted([col for col in df.columns if col.startswith("ST_")])
    final_order = [c for c in id_cols + st_cols + prob_cols if c in df.columns]

    df[final_order].to_csv(args.output_file, index=False, float_format="%.4f")
    print(
        f"Finished in {time.time() - t_script_start:.2f}s. Final cluster file saved to: {args.output_file}"
    )
