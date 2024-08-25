import torch
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

import numpy as np
from sklearn.decomposition import PCA

def reduce_embeddings_with_pca(embeddings, n_components=450):
    print(f"Input embeddings shape: {embeddings.shape}")
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    print(f"Reduced embeddings shape: {reduced_embeddings.shape}")
    return reduced_embeddings

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reduce dimensionality of protein embeddings using PCA")
    parser.add_argument("embedding_file", help="Path to the protein embedding file")
    parser.add_argument("--components", type=int, default=450, help="Number of PCA components to keep")
    
    args = parser.parse_args()
    
    reduce_embeddings_with_pca(args.embedding_file, args.components)