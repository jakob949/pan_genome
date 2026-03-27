# Using protein language models for pangenome construction

The workflow of this approach is as follows:

1. Load genome data in one of the following formats:
   - GenBank
   - FASTA
   - GFF3

2. Perform homology reduction genome-wise (simlarity threshold: 95%)
   The homology reduction is performed via CD-hit. The reason for this step is to reduce the number of genes which we have to compute embeddings for.
   The homology reduction is performed within each genome, meaning that similar genes within the same genome is treated as one.  
   To disable homology reduction via CD-hit, use the flag:
   --disable_cd_hit
3. Calculate embeddings using an protein language model (encoder).
   For the encoder is it possible use huggingface ids like:
   "Synthyra/ESM2-3B", "Rostlab/prot_t5_xl_uniref50"

   --model_name <"hugging face id>
   Remember "" marks
   If your model gives an error, please report it to us then we will add the logic.

   It is possible to chose "onnx", "quantization" or none, accelerations for the creation of embeddings.
   --acceleration "string"
   Some models may not support any of the two accelerations.
   "Rostlab/prot_t5_xl_uniref50" works well with "onnx" and "Synthyra/ESM2-XX" do not support any of the accelerations.

4. Perform PCA dimension reduction (default: 455 dimensions)
   It is possible to decide the number dimensions of the PCA, by using:
   --pca_dim <int>
   To disable PCA set the flag to zero: --pca_dim 0
5. Clustring
   There is 3 different clustering methods availbel: fuzzy, DBSCAN, and HDBSCAN
   fuzzy refers to "weighted single linkage clustering" described in the paper <link>
   To set which clustering algoritm to use, chose between "fuzzy", "hdbscan", "dbscan". Defualt is "fuzzy"
   --algorithm <string>

   To set the distance threshold (epsilon) for DBSCAN and HDBSCAN. For DBSCAN, this defines the strict global maximum radius for neighborhood formation. For HDBSCAN, this acts as the cluster_selection_epsilon, preventing cluster splits below this distance during hierarchical tree condensation. Default 0.1
   --eps <float>

   To set the minimum cluster size for only for HDBSCAN (default=5) use the flag:
   --min_cluster_size <int>

6. Return a pandas DataFrame with all genes and their cluster assignments

## Features

- Flexible input formats (GenBank, FASTA, GFF3)
- Customizable homology reduction threshold
- State-of-the-art protein embedding using Rostlab/prot_t5_xl_uniref50
- Efficient dimension reduction with PCA
- Fast and hierarchical clustering with HDBSCAN
- Easy-to-use output in pandas DataFrame format

## How to install development version

- editable project installation
- faiss options on conda-forge are `faiss-cpu` and `faiss-gpu`. The `faiss` package is
  a meta-package that will install either the CPU or GPU version depending on your
  system. If you want to specify which one to install, you can replace `
faiss` with either `faiss-cpu` or `faiss-gpu` in the command below.

```bash
conda create -n panlm -c conda-forge python=3.10 faiss-gpu -y
pip install -e ".[dev,docs]"
```

## To run

```bash
panlm --help
panlm --input_dir <path to folder with genomes in it>
```
