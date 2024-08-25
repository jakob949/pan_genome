import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from collections import Counter
from typing import List, Tuple, Dict, Any

def hdscan(vectors: np.ndarray, labels: List[str],
           min_cluster_size: int = 2,
           min_samples: int = None,
           n_jobs: int = -1,
           **hdbscan_kwargs: Any) -> Tuple[np.ndarray, Dict[int, Dict[str, int]], pd.DataFrame]:
    """
    Perform HDBSCAN clustering on vectors and analyze which labels are clustered together.
   
    Args:
    vectors (np.ndarray): 2D array of vectors to cluster
    labels (List[str]): List of labels corresponding to each vector
    min_cluster_size (int): Minimum number of samples in a cluster for HDBSCAN
    min_samples (int): Number of samples in a neighborhood for a core point
    n_jobs (int): Number of parallel jobs to run for HDBSCAN. -1 means using all processors.
    **hdbscan_kwargs: Additional keyword arguments for HDBSCAN
   
    Returns:
    Tuple[np.ndarray, Dict[int, Dict[str, int]], pd.DataFrame]:
        - numpy array of cluster labels for each input vector
        - Dictionary of cluster compositions
        - DataFrame with cluster assignments and original labels
    """
    # Perform HDBSCAN clustering
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, 
                        min_samples=min_samples, 
                        core_dist_n_jobs=n_jobs,
                        **hdbscan_kwargs)
    cluster_labels = clusterer.fit_predict(vectors)
    return cluster_labels
    # Create a DataFrame with cluster assignments and original labels
    df = pd.DataFrame({'cluster': cluster_labels, 'label': labels})
   
    # Analyze cluster compositions
    cluster_compositions = {}
    for cluster in df['cluster'].unique():
        cluster_labels = df[df['cluster'] == cluster]['label']
        composition = dict(Counter(cluster_labels))
        cluster_compositions[cluster] = composition
   
    # Sort clusters by size (excluding noise cluster -1)
    sorted_clusters = sorted(
        [c for c in cluster_compositions.keys() if c != -1],
        key=lambda x: sum(cluster_compositions[x].values()),
        reverse=True
    )
   
    # Create a summary DataFrame
    summary_data = []
    for cluster in sorted_clusters + [-1]:  # Add noise cluster at the end
        composition = cluster_compositions[cluster]
        total = sum(composition.values())
        for label, count in composition.items():
            summary_data.append({
                'Cluster': cluster,
                'Label': label,
                'Count': count,
                'Percentage': count / total * 100 if total > 0 else 0
            })
   
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(['Cluster', 'Count'], ascending=[True, False]).reset_index(drop=True)
   
    return cluster_labels, cluster_compositions, summary_df