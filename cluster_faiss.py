import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import math
import os
import glob
import os
import torch
import json
from collections import OrderedDict
from prot_T5 import combine_embedding_files  


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
            self.parent[u] = self.find(self.parent[u])  # Path compression
        return self.parent[u]

    def union(self, u, v):
        """
        Union the sets containing u and v using union by rank.
        """
        u_root = self.find(u)
        v_root = self.find(v)
        if u_root == v_root:
            return
        # Union by rank
        if self.rank[u_root] < self.rank[v_root]:
            self.parent[u_root] = v_root
        else:
            self.parent[v_root] = u_root
            if self.rank[u_root] == self.rank[v_root]:
                self.rank[u_root] += 1

def load_data(path):
    """Load data from either .npz files or one or more .pt files with corresponding headers.
    
    Args:
        path: Either a path to a .npz file, a single .pt file, or a glob pattern for multiple .pt files
            (e.g., 'embeddings/*.pt')
            
    Returns:
        tuple: (embeddings array, protein_ids array)
    """
    
    if '*' in path:
        pt_files = glob.glob(path)
        if not pt_files:
            raise ValueError(f"No files found matching pattern: {path}")
            
        # Get the directory and base pattern for output
        dir_path = os.path.dirname(pt_files[0])
        output_file = os.path.join(dir_path, 'combined_embeddings.pt')
        
        if all(file.lower().endswith('.pt') for file in pt_files):
            print(f"Found {len(pt_files)} .pt files to combine")
            
            # Create temporary combined file
            all_embeddings = OrderedDict()
            all_header_info = {}
            
            for pt_file in pt_files:
                # Load the embeddings OrderedDict
                data = torch.load(pt_file)
                
                header_file = pt_file + '.header.json'
                if not os.path.exists(header_file):
                    raise ValueError(f"Header file not found: {header_file}")
                    
                with open(header_file, 'r') as f:
                    header_info = json.load(f)
                
                # Verify alignment with header 
                if set(data.keys()) != set(header_info.keys()):
                    print(f"Warning: Mismatch between embeddings and header information in {pt_file}")
                
                # Add to combined dictionaries
                all_embeddings.update(data)
                all_header_info.update(header_info)
            
            # Save combined files
            torch.save(all_embeddings, output_file)
            with open(output_file + '.header.json', 'w') as f:
                json.dump(all_header_info, f, indent=2)
            
            protein_ids = list(all_embeddings.keys())
            embeddings = np.stack([tensor.cpu().numpy() for tensor in all_embeddings.values()]).astype('float32')
            
            return embeddings, np.array(protein_ids)
            
        # If all files are .npz, combine them
        elif all(file.lower().endswith('.npz') for file in pt_files):
            all_embeddings = []
            all_protein_ids = []
            for npz_file in pt_files:
                data = np.load(npz_file)
                all_embeddings.append(data['embeddings'].astype('float32'))
                all_protein_ids.extend(data['protein_ids'])
            embeddings = np.vstack(all_embeddings)
            protein_ids = np.array(all_protein_ids)
            return embeddings, protein_ids
        else:
            raise ValueError("All files must be of the same type (.pt or .npz)")
            
    else:
        # Handle single file case
        file_extension = os.path.splitext(path)[1].lower()
        
        if file_extension == '.npz':
            data = np.load(path)
            embeddings = data['embeddings'].astype('float32')
            protein_ids = data['protein_ids']
        elif file_extension == '.pt':
            # Load the embeddings OrderedDict
            data = torch.load(path)
            
            # Load the header information
            header_file = path + '.header.json'
            if not os.path.exists(header_file):
                raise ValueError(f"Header file not found: {header_file}")
                
            with open(header_file, 'r') as f:
                header_info = json.load(f)
            
            # Convert OrderedDict of tensors to numpy array
            protein_ids = list(data.keys())
            embeddings = np.stack([tensor.cpu().numpy() for tensor in data.values()]).astype('float32')
            
            # Verify alignment with header
            if set(protein_ids) != set(header_info.keys()):
                print(f"Warning: Mismatch between embeddings and header information in {path}")
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Please use .npz or .pt files.")
    
    return embeddings, protein_ids

def hierarchical_cluster_faiss_parallel(
    path, 
    similarity_threshold=0.985, 
    core_threshold=0.95, 
    shell_threshold=0.15, 
    cpu=False, 
    k=1000,
    batch_size=10000,
    n_closest=10  # Number of closest neighbors to track
):
    """
    Perform pseudo hierarchical clustering using GPU FAISS with parallelism.

    The algorithm uses a Union-Find data structure to merge clusters based on cosine similarity.
    The clusters are then categorized as core, shell, or cloud based on the percentage of strains they contain.

    Args:
        path (str): Path to the input file (.npz or .pt) containing embeddings and protein IDs
        similarity_threshold (float): Similarity threshold for clustering
        core_threshold (float): Threshold for core category
        shell_threshold (float): Threshold for shell category
        cpu (bool): Use CPU instead of GPU for indexing
        k (int): Number of nearest neighbors to compute
        batch_size (int): Number of embeddings to process in each batch
        n_closest (int): Number of closest neighbors to track

    Returns:
        pd.DataFrame: DataFrame with clustering results
    """
    

    print(f"Similarity threshold: {similarity_threshold}, Core threshold: {core_threshold}, Shell threshold: {shell_threshold}")

    print("Loading data...")
    embeddings, protein_ids = load_data(path)
    num_points = embeddings.shape[0]

    # Initialize arrays to store nearest neighbor information
    nearest_neighbors_sims = np.zeros((num_points, n_closest), dtype=np.float32)
    nearest_neighbors_ids = np.zeros((num_points, n_closest), dtype=np.int32)
    
    if cpu:
        # Normalize embeddings for cosine similarity
        print("Normalizing embeddings...")
        embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

        # Build FAISS index
        d = embeddings.shape[1]
        print("Using CPU for indexing")
        index = faiss.IndexFlatIP(d)
    else:
        # Normalize embeddings for cosine similarity
        print("Normalizing embeddings...")
        faiss.normalize_L2(embeddings)

        # Build FAISS index
        d = embeddings.shape[1]
        print("Using GPU for indexing")
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, d)
    
    print("Adding embeddings to the index...")
    index.add(embeddings)

    # Initialize Union-Find structure
    print("Initializing Union-Find structure...")
    uf = UnionFind(num_points)

    # Calculate number of batches
    num_batches = math.ceil(num_points / batch_size)
    print(f"Clustering embeddings in {num_batches} batches of size {batch_size}...")

    # Iterate over batches
    print("Clustering embeddings...")
    for batch_idx in tqdm(range(num_batches), desc="Clustering Batches"):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_points)
        batch_embeddings = embeddings[start:end]

        # Perform batch search
        similarities, indices = index.search(batch_embeddings, k + 1)  # +1 to include self

        # Iterate through each query in the batch
        for i in range(end - start):
            query_idx = start + i
            
            # Store top n_closest neighbors (excluding self)
            nearest_neighbors_sims[query_idx] = similarities[i][1:n_closest+1]
            nearest_neighbors_ids[query_idx] = indices[i][1:n_closest+1]
            
            # Continue with clustering
            for sim, neighbor_idx in zip(similarities[i][1:], indices[i][1:]):
                if sim >= similarity_threshold:
                    uf.union(query_idx, neighbor_idx)

    # Assign cluster labels
    print("Assigning cluster labels...")
    cluster_labels = [uf.find(i) for i in range(num_points)]

    # Relabel clusters to have consecutive cluster IDs
    unique_labels = np.unique(cluster_labels)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    cluster_labels = np.array([label_mapping[label] for label in cluster_labels])

    # Create DataFrame with nearest neighbor information
    print("Creating DataFrame with clustering results...")
    df = pd.DataFrame({
        'protein_ids': protein_ids,
        'cluster': cluster_labels,
    })


# #################################### CORE/SHELL/CLOUD
    print("Processing core/shell/cloud categorization...")
    # Extract strain information
    df['strain'] = df['protein_ids'].str.split('|').str[1]

    # Calculate strain percentages for each cluster
    all_strains = set(df['strain'])
    total_strains = len(all_strains)

    def calculate_strain_percentage(group):
        return len(set(group['strain'])) / total_strains

    cluster_strain_percentages = df.groupby('cluster').apply(calculate_strain_percentage)

    # Categorize clusters
    def categorize_cluster(cluster):
        percentage = cluster_strain_percentages.get(cluster, 0)
        if percentage >= core_threshold:
            return 'core'
        elif percentage >= shell_threshold:
            return 'shell'
        else:
            return 'cloud'

    df['category'] = df['cluster'].apply(categorize_cluster)



    ###### ADD NEAREST NEIGHBOR INFORMATION

    # Add nearest neighbor information
    for i in range(n_closest):
        df[f'neighbor_{i+1}_similarity'] = nearest_neighbors_sims[:, i]
        df[f'neighbor_{i+1}_id'] = protein_ids[nearest_neighbors_ids[:, i]]

    # # Add average similarity to nearest neighbors
    df['avg_neighbor_similarity'] = nearest_neighbors_sims.mean(axis=1)
    df['min_neighbor_similarity'] = nearest_neighbors_sims.min(axis=1)
    df['max_neighbor_similarity'] = nearest_neighbors_sims.max(axis=1)



    # # Add cluster size and persistence
    cluster_sizes = df['cluster'].value_counts()
    df['cluster_size'] = df['cluster'].map(cluster_sizes)

    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Perform FAISS-based hierarchical clustering for pan-genomic analysis with parallelism")
    parser.add_argument("input_file", help="Path to the input file (.npz or .pt) containing embeddings and protein IDs")
    parser.add_argument("--similarity_threshold", type=float, default=0.98, help="Similarity threshold for clustering")
    parser.add_argument("--core_threshold", type=float, default=0.95, help="Threshold for core category")
    parser.add_argument("--shell_threshold", type=float, default=0.15, help="Threshold for shell category")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU for indexing")
    parser.add_argument("--k", type=int, default=1000, help="Number of nearest neighbors to compute")
    parser.add_argument("--batch_size", type=int, default=10000, help="Number of embeddings to process in each batch")
    parser.add_argument("--output_file", help="Path to save the clustering results")
    args = parser.parse_args()

    result_df = hierarchical_cluster_faiss_parallel(
        args.input_file,
        similarity_threshold=args.similarity_threshold,
        core_threshold=args.core_threshold,
        shell_threshold=args.shell_threshold,
        cpu=args.cpu,
        k=args.k,
        batch_size=args.batch_size
    )
    
    output_file = f'_faiss_UF_{args.similarity_threshold}_mean_cluster.csv'

    output_file = args.input_file.replace(os.path.splitext(args.input_file)[1], output_file)
    result_df.to_csv(output_file, index=False)
    print(f"Clustering results saved to {output_file}")
    print(f"Total clusters: {result_df['cluster'].nunique()}")
    print(result_df['category'].value_counts())
