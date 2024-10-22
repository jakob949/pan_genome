import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from collections import defaultdict
import math

class UnionFind:
    """Union-Find (Disjoint Set) data structure with path compression and union by rank."""
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, u):
        """Find the root of the set containing u with path compression.
        """
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # Path compression
        return self.parent[u]

    def union(self, u, v):
        """Union the sets containing u and v using union by rank."""
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

def hierarchical_cluster_faiss_parallel(
    path, 
    similarity_threshold=0.95, 
    core_threshold=0.95, 
    shell_threshold=0.15, 
    gpu=True, 
    k=1000,
    batch_size=10000  # Define an appropriate batch size
):
    """
    Enhanced hierarchical clustering using FAISS with Union-Find to handle
    order sensitivity and transitivity, utilizing batch processing for parallelism.

    Parameters:
    - path: Path to the .npz file containing 'embeddings' and 'protein_ids'.
    - similarity_threshold: Cosine similarity threshold for clustering.
    - core_threshold: Threshold for categorizing core clusters.
    - shell_threshold: Threshold for categorizing shell clusters.
    - gpu: Boolean indicating whether to use GPU.
    - k: Number of nearest neighbors to compute.
    - batch_size: Number of embeddings to process in each batch.

    Returns:
    - df: A pandas DataFrame containing clustering results and metadata.
    """

    print(f"Similarity threshold: {similarity_threshold}, Core threshold: {core_threshold}, Shell threshold: {shell_threshold}")

    # Load data
    print("Loading data...")
    data = np.load(path)
    embeddings = data['embeddings'].astype('float32')
    protein_ids = data['protein_ids']
    num_points = embeddings.shape[0]

    # Normalize embeddings for cosine similarity
    print("Normalizing embeddings...")
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    d = embeddings.shape[1]
    # if gpu and faiss.get_num_gpus() > 0:
    print("Using GPU for indexing")
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, d)  # Inner product for cosine similarity
# else:
    #     print("Using CPU for indexing")
    #     index = faiss.IndexFlatIP(d)

    print("Adding embeddings to the index...")
    index.add(embeddings)

    # Initialize Union-Find structure
    print("Initializing Union-Find structure...")
    uf = UnionFind(num_points)

    # Calculate number of batches
    num_batches = math.ceil(num_points / batch_size)
    print(f"Clustering embeddings in {num_batches} batches of size {batch_size}...")

    # Iterate over batches to find and merge similar points
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
            for sim, neighbor_idx in zip(similarities[i][1:], indices[i][1:]):  # Exclude self
                if sim >= similarity_threshold:
                    uf.union(query_idx, neighbor_idx)

    # Assign cluster labels
    print("Assigning cluster labels...")
    cluster_labels = [uf.find(i) for i in range(num_points)]

    # Relabel clusters to have consecutive cluster IDs
    unique_labels = np.unique(cluster_labels)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    cluster_labels = np.array([label_mapping[label] for label in cluster_labels])

    # Create DataFrame
    print("Creating DataFrame with clustering results...")
    df = pd.DataFrame({
        'protein_ids': protein_ids,
        'cluster': cluster_labels,
    })

    #################################### CORE/SHELL/CLOUD
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

    # Add cluster size and persistence
    cluster_sizes = df['cluster'].value_counts()
    df['cluster_size'] = df['cluster'].map(cluster_sizes)
    df['cluster_persistence'] = df['cluster_size'] / len(df)

    # Add strain percentage for each cluster
    df['cluster_strain_percentage'] = df['cluster'].map(cluster_strain_percentages)

    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Perform FAISS-based hierarchical clustering for pan-genomic analysis with parallelism")
    parser.add_argument("input_file", help="Path to the input .npz file containing embeddings and protein IDs")
    parser.add_argument("--similarity_threshold", type=float, default=0.98, help="Similarity threshold for clustering")
    parser.add_argument("--core_threshold", type=float, default=0.95, help="Threshold for core category")
    parser.add_argument("--shell_threshold", type=float, default=0.15, help="Threshold for shell category")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration if available")
    parser.add_argument("--k", type=int, default=1000, help="Number of nearest neighbors to compute")
    parser.add_argument("--batch_size", type=int, default=10000, help="Number of embeddings to process in each batch")
    parser.add_argument("--output_file", help="Path to save the clustering results")
    args = parser.parse_args()

    result_df = hierarchical_cluster_faiss_parallel(
        args.input_file,
        similarity_threshold=args.similarity_threshold,
        core_threshold=args.core_threshold,
        shell_threshold=args.shell_threshold,
        gpu=args.gpu,
        k=args.k,
        batch_size=args.batch_size

    )
    
    output_file = f'_faiss_UF_{args.similarity_threshold}.csv'

    output_file = args.input_file.replace('.npz', output_file)
    result_df.to_csv(output_file, index=False)
    print(f"Clustering results saved to {output_file}")
    print(f"Total clusters: {result_df['cluster'].nunique()}")
    print(result_df['category'].value_counts())

    # Print some summary statistics
    print("\nSummary Statistics:")
    print(f"Mean cluster size: {result_df['cluster_size'].mean():.2f}")
    print(f"Median cluster size: {result_df['cluster_size'].median()}")
    print(f"Max cluster size: {result_df['cluster_size'].max()}")
