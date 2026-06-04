import argparse
import glob
import math
import os
import time
from collections import OrderedDict, defaultdict

import faiss
import numba
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN, HDBSCAN
from tqdm import tqdm


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


@numba.njit
def resolve_all_parents(parents, out):
    n = parents.shape[0]
    for i in range(n):
        out[i] = numba_find(parents, i)


@numba.njit
def filter_valid_pairs(idxs, dists, start, distance_threshold):
    n_queries = idxs.shape[0]
    k = idxs.shape[1]
    
    count = 0
    for j in range(n_queries):
        for c in range(k):
            neighbor_idx = idxs[j, c]
            if neighbor_idx == -1:
                break
            if dists[j, c] <= distance_threshold:
                count += 1
                
    rows = np.empty(count, dtype=np.int32)
    cols = np.empty(count, dtype=np.int32)
    data = np.empty(count, dtype=np.float32)
    
    idx = 0
    for j in range(n_queries):
        query_idx = start + j
        for c in range(k):
            neighbor_idx = idxs[j, c]
            if neighbor_idx == -1:
                break
            d = dists[j, c]
            if d <= distance_threshold:
                rows[idx] = query_idx
                cols[idx] = neighbor_idx
                data[idx] = d
                idx += 1
                
    return rows, cols, data


@numba.njit
def find_component_representatives(labels, n_components):
    reps = np.empty(n_components, dtype=np.int32)
    seen = np.zeros(n_components, dtype=numba.boolean)
    count = 0
    for i in range(labels.shape[0]):
        c = labels[i]
        if not seen[c]:
            reps[c] = i
            seen[c] = True
            count += 1
            if count == n_components:
                break
    return reps


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
    ) = zip(*final_rows, strict=True)
    return {
        "weight_info": {"weighting_enabled": use_weights, "mean": mean, "std_dev": sd}
    }, columns


def search_index_batched(index, queries, k, cpu=False):
    """
    Search a FAISS index in sub-batches to prevent GPU memory allocation issues.
    """
    sub_batch_size = 65536 if cpu else 4096
    n_queries = queries.shape[0]
    if n_queries <= sub_batch_size:
        return index.search(queries, k)

    sims_list = []
    idxs_list = []
    for start in range(0, n_queries, sub_batch_size):
        end = min(start + sub_batch_size, n_queries)
        sub_sims, sub_idxs = index.search(queries[start:end], k)
        sims_list.append(sub_sims)
        idxs_list.append(sub_idxs)

    return np.vstack(sims_list), np.vstack(idxs_list)


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
    use_ann: bool = False,
    nprobe: int = 32,
    nlist: int = None,
):
    embeddings, protein_ids_np = load_data(path)
    num_points = embeddings.shape[0]
    original_dim = embeddings.shape[1]

    if clust_method in ["hdbscan", "dbscan"] and k > 50:
        print(f"Cap k from {k} to 50 for DBSCAN/HDBSCAN to optimize search speed and host memory usage.")
        k = 50

    if pca_dim is not None and pca_dim < original_dim:
        pca = faiss.PCAMatrix(original_dim, pca_dim)
        max_pca_train = min(num_points, 1000000)
        if num_points > max_pca_train:
            print(f"Training PCA matrix on a representative subset of {max_pca_train} points...")
            rng = np.random.default_rng(42)
            pca_train_indices = rng.choice(num_points, size=max_pca_train, replace=False)
            pca.train(embeddings[pca_train_indices])
        else:
            pca.train(embeddings)
        embeddings = pca.apply(embeddings)

    faiss.normalize_L2(embeddings)

    if use_ann:
        if nlist is None or nlist <= 0:
            nlist = int(4 * math.sqrt(num_points))
        # Ensure we have enough points per centroid for IVF training (at least 39 points per centroid is recommended by FAISS)
        nlist = max(1, min(nlist, num_points // 39))
        nlist = max(1, nlist)

        print(f"Building IVF index with nlist={nlist}, nprobe={nprobe}...")
        quantizer = faiss.IndexFlatIP(embeddings.shape[1])
        cpu_index = faiss.IndexIVFFlat(
            quantizer, embeddings.shape[1], nlist, faiss.METRIC_INNER_PRODUCT
        )

        max_train_points = 50 * nlist
        if not cpu:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.GpuIndexIVFFlat(res, embeddings.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
            t_ivf_start = time.time()
            if num_points > max_train_points:
                print(f"Training IVF index on GPU with a random subset of {max_train_points} points (out of {num_points})...")
                rng = np.random.default_rng(42)
                train_indices = rng.choice(num_points, size=max_train_points, replace=False)
                gpu_index.train(embeddings[train_indices])
            else:
                gpu_index.train(embeddings)
            print(f"IVF index trained on GPU in {time.time() - t_ivf_start:.2f}s")
            
            t_add_start = time.time()
            gpu_index.add(embeddings)
            print(f"IVF index populated on GPU in {time.time() - t_add_start:.2f}s")
            index = gpu_index
        else:
            t_ivf_start = time.time()
            if num_points > max_train_points:
                print(f"Training IVF index on CPU with a random subset of {max_train_points} points (out of {num_points})...")
                rng = np.random.default_rng(42)
                train_indices = rng.choice(num_points, size=max_train_points, replace=False)
                cpu_index.train(embeddings[train_indices])
            else:
                cpu_index.train(embeddings)
            cpu_index.add(embeddings)
            print(f"IVF index built and trained on CPU in {time.time() - t_ivf_start:.2f}s")
            index = cpu_index

        index.nprobe = min(nlist, nprobe)
    else:
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

        all_rows = []
        all_cols = []
        all_data = []

        num_batches = math.ceil(num_points / batch_size)
        for i in tqdm(range(num_batches), desc="FAISS Search"):
            start = i * batch_size
            end = min(start + batch_size, num_points)
            k_search = min(k, num_points)
            sims, idxs = search_index_batched(index, embeddings[start:end], k_search, cpu=cpu)

            dists = np.sqrt(np.maximum(0, 2 - 2 * sims))

            r, c, d = filter_valid_pairs(idxs, dists, start, l2_eps)
            if len(r) > 0:
                all_rows.append(r)
                all_cols.append(c)
                all_data.append(d)

        print(
            f"FAISS search & distance conversion: {time.time() - t_search_start:.2f}s"
        )

        if all_rows:
            rows = np.concatenate(all_rows)
            cols = np.concatenate(all_cols)
            data = np.concatenate(all_data)
        else:
            rows = np.array([], dtype=np.int32)
            cols = np.array([], dtype=np.int32)
            data = np.array([], dtype=np.float32)

        sparse_dist = csr_matrix(
            (data, (rows, cols)), shape=(num_points, num_points)
        )
        sparse_dist = sparse_dist.maximum(sparse_dist.transpose())

        if clust_method == "hdbscan":
            # Find connected components and connect them if there are more than 1
            n_components, cc_labels = connected_components(sparse_dist, directed=False, return_labels=True)
            if n_components > 1:
                print(f"Connecting {n_components} disconnected components to satisfy HDBSCAN single-linkage tree...")
                component_nodes = find_component_representatives(cc_labels, n_components)
                new_rows = []
                new_cols = []
                new_data = []
                for idx in range(n_components - 1):
                    u = component_nodes[idx]
                    v = component_nodes[idx+1]
                    new_rows.extend([u, v])
                    new_cols.extend([v, u])
                    new_data.extend([2.0, 2.0]) # Use maximum possible distance for normalized vectors (2.0)
                
                coo = sparse_dist.tocoo()
                r = np.concatenate([coo.row, np.array(new_rows, dtype=np.int32)])
                c = np.concatenate([coo.col, np.array(new_cols, dtype=np.int32)])
                d = np.concatenate([coo.data, np.array(new_data, dtype=np.float32)])
                sparse_dist = csr_matrix((d, (r, c)), shape=(num_points, num_points))

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

        columns["category"] = np.array([label_to_cat[lbl] for lbl in labels])
        columns["category_prob"] = probs

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
        k_search = min(k + 1, num_points)
        for i in tqdm(range(num_batches), desc="Fuzzy Clustering"):
            start = i * batch_size
            end = min(start + batch_size, num_points)
            sims, idxs = search_index_batched(index, embeddings[start:end], k_search, cpu=cpu)
            process_batch_numba(
                parents_2d, ranks_2d, sims, idxs, similarity_thresholds, start
            )

        all_labels = {}
        for i, st in enumerate(similarity_thresholds):
            labels_arr = np.empty(num_points, dtype=np.int32)
            resolve_all_parents(parents_2d[i], labels_arr)
            all_labels[f"{st:.4f}"] = labels_arr

        for st_val in similarity_thresholds:
            st_key = f"{st_val:.4f}"
            labels = all_labels[st_key]
            columns[f"ST_{st_key}"] = labels
            unique_labels, inv = np.unique(labels, return_inverse=True)
            counts = np.bincount(inv)

            cluster_strains = {lbl: set() for lbl in unique_labels}
            for i, lbl in enumerate(labels):
                s = all_strains_np[i]
                if s:
                    cluster_strains[lbl].add(s)

            l_to_props = {}
            for lbl in unique_labels:
                s_frac = (
                    len(cluster_strains[lbl]) / total_strains
                    if total_strains > 0
                    else 0
                )
                cat = (
                    "core"
                    if s_frac >= core_threshold
                    else ("shell" if s_frac >= shell_threshold else "cloud")
                )
                l_to_props[lbl] = (counts[np.where(unique_labels == lbl)[0][0]], cat)

            columns[f"ST_{st_key}_size"] = np.array(
                [l_to_props[lbl][0] for lbl in labels]
            )
            columns[f"ST_{st_key}_category"] = np.array(
                [l_to_props[lbl][1] for lbl in labels]
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
    parser.add_argument(
        "--use_ann",
        action="store_true",
        help="Use Approximate Nearest Neighbors (ANN) via FAISS IVF index",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=32,
        help="Number of centroids to probe for IVF index (used when --use_ann is set)",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=None,
        help="Number of centroids for IVF index (default: 4 * sqrt(N))",
    )

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
        use_ann=args.use_ann,
        nprobe=args.nprobe,
        nlist=args.nlist,
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
