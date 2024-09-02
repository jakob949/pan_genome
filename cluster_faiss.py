import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from collections import defaultdict

def hierarchical_cluster_faiss(path, similarity_threshold=0.95, core_threshold=0.95, shell_threshold=0.15, gpu=False, n_neighbors=10):
    # Load data
    data = np.load(path)
    embeddings = data['embeddings'].astype('float32')
    protein_ids = data['protein_ids']

    # Normalize embeddings - then the dot product cosine similarity
    faiss.normalize_L2(embeddings)

    # Build FAISS index - using dimension of embeddings
    d = embeddings.shape[1] # 
    
    if gpu and faiss.get_num_gpus() > 0:
        print("Using GPU for indexing")
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, d) # use dot product similarity
    else:
        print("Using CPU for indexing")
        index = faiss.IndexFlatIP(d)  
    
    index.add(embeddings)

    clusters = defaultdict(list)
    used = set()
    nearest_neighbors = []
    nearest_similarities = []

    for i in tqdm(range(len(embeddings)), desc="Clustering"):
        # Find similar proteins for all embeddings 
        # - containing the indices of the nearest neighbors in the original embeddings array
        similarities, indices = index.search(embeddings[i:i+1], n_neighbors + 1)  # +1 itself
        nearest_neighbors.append(indices[0][1:])  # Exclude self
        nearest_similarities.append(similarities[0][1:])  # Exclude self similarity

        if i in used:
            continue
        
        # creates a list of tuples (index, similarity) for neighbors that meet the similarity threshold and not in the used set
        similar = [(idx, sim) for idx, sim in zip(indices[0], similarities[0]) 
                   if sim >= similarity_threshold and idx not in used]
        
        # create new cluster - singleton - no similar members
        if not similar:
            clusters[i].append(i)
            used.add(i)
        else: 
            # create new cluster - multiple members
            cluster = [i] + [idx for idx, _ in similar]
            for idx in cluster:
                clusters[i].append(idx)
                used.add(idx)

    ################## saving

    # Create DataFrame
    df = pd.DataFrame({
        'protein_ids': protein_ids,
        'cluster': [next(c for c, members in clusters.items() if i in members) for i in range(len(protein_ids))],
        'nearest_neighbors': nearest_neighbors,
        'nearest_similarities': nearest_similarities
    })

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
        percentage = cluster_strain_percentages[cluster]
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

    # Add mean similarity to nearest neighbors
    df['mean_similarity_to_neighbors'] = df['nearest_similarities'].apply(np.mean)

    # Add number of neighbors above similarity threshold
    df['neighbors_above_threshold'] = df['nearest_similarities'].apply(lambda x: sum(x >= similarity_threshold))

    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Perform detailed FAISS-based hierarchical clustering for pan-genomic analysis")
    parser.add_argument("input_file", help="Path to the input .npz file containing embeddings and protein IDs")
    parser.add_argument("--similarity_threshold", type=float, default=0.95, help="Similarity threshold for clustering")
    parser.add_argument("--core_threshold", type=float, default=0.95, help="Threshold for core category")
    parser.add_argument("--shell_threshold", type=float, default=0.15, help="Threshold for shell category")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration if available")
    parser.add_argument("--n_neighbors", type=int, default=10, help="Number of nearest neighbors to find")
    args = parser.parse_args()

    result_df = hierarchical_cluster_faiss(args.input_file, args.similarity_threshold, 
                                           args.core_threshold, args.shell_threshold, 
                                           args.gpu, args.n_neighbors)
    output_file = args.input_file.replace('.npz', '_faiss_hierarchical_detailed.csv')
    result_df.to_csv(output_file, index=False)
    print(f"Detailed clustering results saved to {output_file}")
    print(f"Total clusters: {result_df['cluster'].nunique()}")
    print(result_df['category'].value_counts())

    # Print some summary statistics
    print("\nSummary Statistics:")
    print(f"Mean cluster size: {result_df['cluster_size'].mean():.2f}")
    print(f"Mean similarity to neighbors: {result_df['mean_similarity_to_neighbors'].mean():.4f}")
    print(f"Mean neighbors above threshold: {result_df['neighbors_above_threshold'].mean():.2f}")