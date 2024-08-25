
import hdbscan
import pandas as pd
import numpy as np

def cluster_hdscan(path):
    data = np.load(path)
    embeddings = data['embeddings']
    protein_ids = data['protein_ids']
    clusterer1 = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, allow_single_cluster=True, core_dist_n_jobs=-1)
    clusterer1.fit(embeddings)
    cluster_persistence_dict = dict(enumerate(clusterer1.cluster_persistence_))
    df = pd.DataFrame({'protein_ids': protein_ids, 'cluster': clusterer1.labels_, 'probs': clusterer1.probabilities_})
    df['cluster_persistence'] = df['cluster'].map(cluster_persistence_dict)
    df.loc[df['cluster'] == -1, 'cluster_persistence'] = np.nan
    df['strain'] = df['protein_ids'].str.split('|').str[1]


    cluster_persistence_dict = dict(enumerate(clusterer1.cluster_persistence_))

    # Create the initial DataFrame
    df = pd.DataFrame({
        'protein_ids': protein_ids, 
        'cluster': clusterer1.labels_, 
        'probs': clusterer1.probabilities_
    })

    df['cluster_persistence'] = df['cluster'].map(cluster_persistence_dict)
    df.loc[df['cluster'] == -1, 'cluster_persistence'] = np.nan
    df['strain'] = df['protein_ids'].str.split('|').str[1]
    all_strains = set(df['strain'].unique())
    total_strains = len(all_strains)

    def strain_percentage(group):
        return len(set(group['strain'])) / total_strains * 100

    cluster_strain_percentages = df.groupby('cluster').apply(strain_percentage)

    # Function to categorize clusters
    def categorize_cluster(cluster):
        if cluster == -1:
            return 'cloud'
        percentage = cluster_strain_percentages[cluster]
        if percentage == 100:
            return 'core'
        elif percentage > 20:
            return 'shell'
        else:
            return 'cloud'

    df['category'] = df['cluster'].apply(categorize_cluster)

    return df