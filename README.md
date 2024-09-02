# Pan Genome Analysis

This is an experimental approach to find the pan-genome of an collection of geneomes.

## Workflow

The simple workflow of this approach is as follows:

1. Load genome data in one of the following formats:
   - GenBank
   - FASTA
   - GFF3

2. Perform homology reduction genome-wise (default threshold: 98%)

3. Calculate embeddings using the Rostlab/prot_t5_xl_uniref50 model

4. Perform PCA dimension reduction (default: 455 dimensions)

5. Perform HDBSCAN clustering (fast and hierarchical)

6. Return a pandas DataFrame with all genes and their cluster assignments

## Features

- Flexible input formats (GenBank, FASTA, GFF3)
- Customizable homology reduction threshold
- State-of-the-art protein embedding using Rostlab/prot_t5_xl_uniref50
- Efficient dimension reduction with PCA
- Fast and hierarchical clustering with HDBSCAN
- Easy-to-use output in pandas DataFrame format

## To run

```bash
git clone [repository URL]
python3 main.py --input_dir <path to folder with genomes in it>
```
