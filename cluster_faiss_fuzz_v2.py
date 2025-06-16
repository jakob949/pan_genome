import numpy as np
import faiss
from tqdm import tqdm
import math
import os
import glob
import torch
import json
from collections import OrderedDict, defaultdict
import pickle
import time
import csv

class UnionFind:
    """
    Union-Find (Disjoint Set) data structure with path compression and union by rank.
    """
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, u):
        """
        Find the root of the set containing u with path compression.
        """
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # Path compression see wiki
        return self.parent[u]

    def union(self, u, v):
        """
        Union the sets containing u and v using union by rank.
        """
        u_root = self.find(u)
        v_root = self.find(v)
        if u_root == v_root:
            return
        if self.rank[u_root] < self.rank[v_root]:
            self.parent[u_root] = v_root
        else:
            self.parent[v_root] = u_root
            if self.rank[u_root] == self.rank[v_root]:
                self.rank[u_root] += 1


def load_data(path):
    """Load data from either .npz files or one or more .pt files with corresponding headers."""
    if '*' in path:
        pt_files = glob.glob(path)
        if not pt_files:
            raise ValueError(f"No files found matching pattern: {path}")

        dir_path = os.path.dirname(pt_files[0])
        output_file = os.path.join(dir_path, 'combined_embeddings.pt')

        if all(file.lower().endswith('.pt') for file in pt_files):
            print(f"Found {len(pt_files)} .pt files to combine")
            all_embeddings = OrderedDict()
            all_header_info = {}

            for pt_file in pt_files:
                data = torch.load(pt_file)
                header_file = pt_file + '.header.json'
                if not os.path.exists(header_file):
                    raise ValueError(f"Header file not found: {header_file}")

                with open(header_file, 'r') as f:
                    header_info = json.load(f)

                if set(data.keys()) != set(header_info.keys()):
                    print(f"Warning: Mismatch between embeddings and header in {pt_file}")

                all_embeddings.update(data)
                all_header_info.update(header_info)

            torch.save(all_embeddings, output_file)
            with open(output_file + '.header.json', 'w') as f:
                json.dump(all_header_info, f, indent=2)

            protein_ids = list(all_embeddings.keys())
            embeddings = np.stack([tensor.cpu().numpy() for tensor in all_embeddings.values()]).astype('float32')
            print(f"shape of combined embeddings: {embeddings.shape}")
            return embeddings, np.array(protein_ids) 

        elif all(file.lower().endswith('.npz') for file in pt_files):
            all_embeddings_list = [] 
            all_protein_ids_list = [] 
            for npz_file in pt_files:
                data = np.load(npz_file)
                all_embeddings_list.append(data['embeddings'].astype('float32'))
                all_protein_ids_list.extend(data['protein_ids'])
            embeddings = np.vstack(all_embeddings_list)
            protein_ids = np.array(all_protein_ids_list)
            print(f"shape of combined embeddings: {embeddings.shape}")
            return embeddings, protein_ids
        else:
            raise ValueError("All files must be of the same type (.pt or .npz)")
    else:
        file_extension = os.path.splitext(path)[1].lower()
        if file_extension == '.npz':
            data = np.load(path)
            embeddings = data['embeddings'].astype('float32')
            protein_ids = data['protein_ids']
        elif file_extension == '.pt':
            data = torch.load(path)
            header_file = path + '.header.json'
            if not os.path.exists(header_file):
                raise ValueError(f"Header file not found: {header_file}")
            with open(header_file, 'r') as f:
                header_info = json.load(f)

            _protein_ids_list = list(data.keys())
            embeddings = np.stack([tensor.cpu().numpy() for tensor in data.values()]).astype('float32')
            if set(_protein_ids_list) != set(header_info.keys()):
                print(f"Warning: Mismatch between embeddings and header in {path}")
            protein_ids = np.array(_protein_ids_list) 
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Please use .npz or .pt.")
        print(f"shape of combined embeddings: {embeddings.shape}")
        return embeddings, protein_ids


def fast_calculate_probabilities_optimized(columns, sim_thresholds):
    """
    Efficiently calculate probabilities (core/shell/cloud and cluster) using only numpy and dictionaries.
    columns must contain:
      columns["protein_id"], and for each threshold st:
         columns[f"ST_{st}_category"] and columns[f"ST_{st}"].
    """

    t1 = time.time()
    protein_ids_list = columns["protein_id"] 
    protein_ids_np_arr = np.array(protein_ids_list) 
    unique_protein_ids = np.unique(protein_ids_np_arr)
    protein_to_idx = {pid: i for i, pid in enumerate(unique_protein_ids)}
    n_proteins = len(unique_protein_ids)
    n_thresholds = len(sim_thresholds)
    category_types = ['core', 'shell', 'cloud']
    category_to_idx = {cat: i for i, cat in enumerate(category_types)}
    n_rows = len(protein_ids_list)

    # Map each row to the index of its protein
    protein_indices = np.array([protein_to_idx[pid] for pid in protein_ids_list])

    # Count how many times each category occurs for each protein
    category_counts = np.zeros((n_proteins, len(category_types)), dtype=np.int32)
    for st in sim_thresholds:
        cat_array = columns[f"ST_{st}_category"]
        for i, cat in enumerate(cat_array):
            cat_idx = category_to_idx[cat]
            p_idx = protein_indices[i]
            category_counts[p_idx, cat_idx] += 1

    # Count cluster occurrences for each protein
    cluster_counts = [defaultdict(int) for _ in range(n_proteins)]
    for st in sim_thresholds:
        cluster_array = columns[f"ST_{st}"]
        for i, cluster_label in enumerate(cluster_array):
            p_idx = protein_indices[i]
            cluster_counts[p_idx][cluster_label] += 1

    # Calculate category probabilities
    category_probs = category_counts / float(n_thresholds)

    # Build a dict: protein_id -> {cat: prob}
    category_probabilities_d = {}
    for pid, idx in protein_to_idx.items():
        cat_dict = {}
        for cat_name, cat_i in category_to_idx.items():
            cat_dict[cat_name] = category_probs[idx, cat_i]
        category_probabilities_d[pid] = cat_dict

    # Build cluster probability for each protein
    cluster_probabilities = {}
    for pid, idx in protein_to_idx.items():
        total_c = sum(cluster_counts[idx].values())
        if total_c > 0:
            cluster_probabilities[pid] = {
                k: v / float(total_c) for k, v in cluster_counts[idx].items()
            }
        else:
            cluster_probabilities[pid] = {}

    # For each row, compute cluster_id, cluster_prob, category, category_prob
    cluster_id_col = []
    cluster_prob_col = []
    category_col = []
    category_p_col = []

    for i in range(n_rows):
        pid = protein_ids_list[i] # Use the original list for iteration
        c_probs = cluster_probabilities[pid]
        if c_probs:
            best_cluster = max(c_probs, key=c_probs.get)
            best_cluster_prob = c_probs[best_cluster]
        else:
            best_cluster = ""
            best_cluster_prob = 0.0

        cat_probs_dict = category_probabilities_d[pid]
        best_cat = max(cat_probs_dict, key=cat_probs_dict.get)
        best_cat_prob = cat_probs_dict[best_cat]

        cluster_id_col.append(best_cluster)
        cluster_prob_col.append(best_cluster_prob)
        category_col.append(best_cat)
        category_p_col.append(best_cat_prob)

    columns["cluster_id"] = cluster_id_col
    columns["cluster_prob"] = cluster_prob_col
    columns["category"] = category_col
    columns["category_prob"] = category_p_col

    print(f"Fast probability calculation took {time.time() - t1:.2f} seconds")
    results = {
        "cluster_probabilities": cluster_probabilities,
        "category_probabilities": category_probabilities_d
    }
    return results, columns


def cluster_faiss_parallel(
    path: str,
    similarity_thresholds,
    core_threshold: float = 0.95,
    shell_threshold: float = 0.15,
    cpu: bool = False,
    k: int = 1000,
    batch_size: int = 100000,
    n_closest: int = 0,
    n_iter: int = 1,
):
    """
    Perform pseudo clustering using GPU FAISS with parallelism.

    """
    print("Loading data...")
    embeddings, protein_ids_np = load_data(path)
    num_points = embeddings.shape[0]

    # Optionally track nearest neighbors --> makes the final csv too bloated
    if n_closest > 0:
        nearest_neighbors_sims = np.zeros((num_points, n_closest), dtype=np.float32)
        nearest_neighbors_ids = np.zeros((num_points, n_closest), dtype=np.int32)

    #  dictionary that will hold cluster labels for each threshold
    all_cluster_labels = {f"ST_{st}": [None]*num_points for st in similarity_thresholds}

    for iteration in range(n_iter):
        print(f"Starting iteration {iteration + 1}/{n_iter}")

        # Normalize embeddings
        if cpu:
            print("Using CPU for indexing")
            embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings_normalized)
        else:
            print("Using GPU for indexing")
            embeddings_normalized = embeddings.copy()
            faiss.normalize_L2(embeddings_normalized)
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, embeddings.shape[1])
            index.add(embeddings_normalized)

        uf_dict = {st: UnionFind(num_points) for st in similarity_thresholds}
        num_batches_calc = math.ceil(num_points / batch_size)

        print(f"Processing {num_batches_calc} batches...")


        for batch_idx in tqdm(range(num_batches_calc), desc=f"Iteration {iteration+1}"): 
            start = batch_idx * batch_size
            end = min(start + batch_size, num_points)
            if start >= end: 
                continue
            t2 = time.time()
            batch_embeddings = embeddings_normalized[start:end]
            similarities, indices = index.search(batch_embeddings, k+1) # k+1 because first is self
            # print(f"Time taken for faiss search (batch {batch_idx}): {time.time() - t2:.2f} sec") # Optional print
            if iteration == n_iter - 1 and n_closest > 0:
                # avoid out of bounds: k+1 < n_closest+1
                actual_neighbors_to_store = min(n_closest, similarities.shape[1] - 1)
                if actual_neighbors_to_store > 0:
                    nearest_neighbors_sims[start:end, :actual_neighbors_to_store] = similarities[:, 1:actual_neighbors_to_store+1]
                    nearest_neighbors_ids[start:end, :actual_neighbors_to_store] = indices[:, 1:actual_neighbors_to_store+1]


            t_merge = time.time() 
            for i in range(end - start):
                query_idx = start + i
                # Skip the first element (self-similarity) by starting from [i][1:]
                for sim, neighbor_idx in zip(similarities[i][1:], indices[i][1:]):
                    if neighbor_idx == -1: # faiss  return -1 for padding/no neighbor
                        continue
                    for st_val, uf in uf_dict.items():
                        if sim >= st_val:
                            uf.union(query_idx, neighbor_idx)
            # print(f"time taken for merging over thresholds UF (batch {batch_idx}): {time.time() - t_merge:.2f} sec") # Optional print

        # Assign cluster labels
        for st_val, uf in uf_dict.items(): 
            st_key = f"ST_{st_val}"
            for i in range(num_points):
                all_cluster_labels[st_key][i] = uf.find(i)

        # If more iterations remain, update embeddings to cluster centroids
        if iteration < n_iter - 1:
            print("Updating embeddings for next iteration...")
            # Use first threshold's labels
            first_threshold_st_val = similarity_thresholds[0]
            first_threshold_labels = all_cluster_labels[f"ST_{first_threshold_st_val}"]
            unique_labels = np.unique(first_threshold_labels)
            new_embeddings = np.zeros_like(embeddings)

            # label_to_mean = {} for debugging
            for label in unique_labels:
                mask_indices = [i for i, lab in enumerate(first_threshold_labels) if lab == label]
                if mask_indices: 
                    centroid = embeddings[mask_indices].mean(axis=0)
                    for idx_in_mask in mask_indices: # Renamed to avoid conflict
                        new_embeddings[idx_in_mask] = centroid
            embeddings = new_embeddings # Update embeddings for the next iteration

    # Build final columns dictionary
    columns = {}
    columns["protein_id"] = protein_ids_np.tolist()
    # Extract strain from "protein_id" (split by '|')
    # If protein_id is like "xxx|strainName|yyy", the strain is the 2nd element
    strain_col = []
    for pid in columns["protein_id"]:
        parts = str(pid).split('|') # Ensure pid is string
        strain = parts[1] if len(parts) > 1 else ""
        strain_col.append(strain)
    columns["strain"] = strain_col

    # Calculate total number of unique strains
    unique_strains = set(s for s in strain_col if s) # Filter out empty strains
    total_strains = len(unique_strains)
    if total_strains == 0:
        print("Warning: No unique strains found. Category calculation might be affected.")



    #categorize clusters by fraction of strains present
    def fraction_of_strains_in_cluster(indices_in_cluster): # Renamed for clarity
        # Return fraction of unique strains in these indices
        if not total_strains: # Handle division by zero
             return 0.0
        cluster_strains = set(strain_col[i] for i in indices_in_cluster if strain_col[i]) # Filter empty
        return len(cluster_strains) / float(total_strains)


    def categorize(percentage):
        if percentage >= core_threshold:
            return 'core'
        elif percentage >= shell_threshold:
            return 'shell'
        else:
            return 'cloud'

    # For each similarity threshold, fill columns:
    #   ST_{st}, ST_{st}_size, ST_{st}_category
    # We'll need cluster -> indices for each threshold to compute sizes and strain fractions
    for st_val in similarity_thresholds: # Renamed st to st_val
        st_key = f"ST_{st_val}"
        columns[st_key] = all_cluster_labels[st_key]
        # Count how many times each cluster appears
        cluster_count_map = defaultdict(int)
        for c_label_in_col in columns[st_key]: 
            cluster_count_map[c_label_in_col] += 1

        # Build a list for size col
        size_col = []
        for c_label_in_col in columns[st_key]: 
            size_col.append(cluster_count_map[c_label_in_col])
        columns[f"{st_key}_size"] = size_col

        # For strain fraction, group rows by cluster
        cluster_indices_map = defaultdict(list)
        for i, c_label_in_col in enumerate(columns[st_key]): 
            cluster_indices_map[c_label_in_col].append(i)

        # cluster -> fraction_of_strains
        cluster_strain_fraction = {}
        for c_label, idx_list in cluster_indices_map.items():
            frac = fraction_of_strains_in_cluster(idx_list)
            cluster_strain_fraction[c_label] = frac

        # Category col
        cat_col = []
        for c_label_in_col in columns[st_key]: # Renamed
            frac = cluster_strain_fraction[c_label_in_col]
            cat_col.append(categorize(frac))
        columns[f"{st_key}_category"] = cat_col

    # compute probabilities (cluster/category) for each protein
    _, columns = fast_calculate_probabilities_optimized(columns, similarity_thresholds)

    return columns


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Perform FAISS-based clustering without pandas.")
    parser.add_argument("--input_file", default="pan_genome/full_analysis_output/9_genes_base/all_reduced_embeddings_pca450.npz", help="Path to the input file (.npz or .pt)")
    # parser.add_argument("--similarity_threshold", type=float, default=0.98, help="Similarity threshold for clustering") # Removed as --st is used
    parser.add_argument("--core_threshold", type=float, default=0.95, help="Threshold for core category")
    parser.add_argument("--shell_threshold", type=float, default=0.15, help="Threshold for shell category")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for FAISS indexing")
    parser.add_argument("--k", type=int, default=1000, help="Number of nearest neighbors for FAISS search (k+1 returned)")
    parser.add_argument("--n_iter", type=int, default=1, help="Number of clustering iterations")
    parser.add_argument("--n_closest_points", type=int, default=0, help="Track this many nearest neighbor points (0 to disable)")
    parser.add_argument("--batch_size", type=int, default=100000, help="Batch size for FAISS search")
    parser.add_argument("--st", type=str, default="0.7,1.02,15", help="Similarity thresholds to use. Comma-separated: low,high,steps")
    parser.add_argument("--output_file", help="Where to save the final CSV")
    args = parser.parse_args()
    
    try:
        similarity_parts = str(args.st).split(',')
        if len(similarity_parts) != 3:
            raise ValueError("Argument --st must be in the format low,high,steps")
        sim_low = float(similarity_parts[0])
        sim_high = float(similarity_parts[1])
        sim_steps = int(similarity_parts[2])
        if sim_steps < 1:
            raise ValueError("Number of steps for similarity thresholds must be at least 1.")
        if sim_low > sim_high:
             print(f"Warning: Low similarity threshold ({sim_low}) is greater than high threshold ({sim_high}). Swapping them.")
             sim_low, sim_high = sim_high, sim_low
        if sim_steps == 1 and sim_low != sim_high:

            similarity_thresholds = np.array([sim_low]) 
            print(f"Using single similarity threshold: {sim_low}")
        elif sim_low == sim_high:
             similarity_thresholds = np.array([sim_low])
             print(f"Using single similarity threshold: {sim_low}")
        else:
            similarity_thresholds = np.linspace(sim_low, sim_high, sim_steps)
            print(f"Similarity thresholds: low={sim_low:.4f}, high={sim_high:.4f}, steps={sim_steps}")
    except ValueError as e:
        print(f"Error parsing --st argument '{args.st}': {e}")
        exit(1)
        
    
    columns = cluster_faiss_parallel(
        args.input_file,
        similarity_thresholds,
        core_threshold=args.core_threshold,
        shell_threshold=args.shell_threshold,
        cpu=args.cpu,
        k=args.k, # k neighbors means k+1 results from search
        batch_size=args.batch_size,
        n_closest=args.n_closest_points,
        n_iter=args.n_iter,
    )


    if args.output_file is None:
        
        input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
        out_csv = f"{input_basename}_fuzz_clustering_v2.csv"
        print(f"Output file not specified, defaulting to: {out_csv}")
    else:
        out_csv = args.output_file

    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print(f"Created output directory: {out_dir}")


    header = ["protein_id", "strain"]
    for st_val in similarity_thresholds: 
        st_key_formatted = f"ST_{st_val:.4f}" 
        header.append(st_key_formatted)
        header.append(f"{st_key_formatted}_size")
        header.append(f"{st_key_formatted}_category")
    header += ["cluster_id", "cluster_prob", "category", "category_prob"]

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        n_rows = len(columns["protein_id"])

        for i in range(n_rows):
            row = [
                columns["protein_id"][i],
                columns["strain"][i]
            ]
            for st_val in similarity_thresholds: 
                st_key = f"ST_{st_val}" 
                row.append(columns[st_key][i])
                row.append(columns[f"{st_key}_size"][i])
                row.append(columns[f"{st_key}_category"][i])
            row.append(columns["cluster_id"][i])
            # Format probabilities for CSV
            row.append(f"{columns['cluster_prob'][i]:.4f}")
            row.append(columns["category"][i])
            row.append(f"{columns['category_prob'][i]:.4f}")
            writer.writerow(row)

    print(f"Results saved to: {out_csv}")