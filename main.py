import os
import argparse
import glob
from Bio import SeqIO
from cd_hit import run_cd_hit
from file_parsing import parse_single_file, file_exists_check, write_proteins_to_fasta
from prot_T5 import calculate_embeddings, combine_embedding_files, concatenate_embeddings
from pca_reduction import reduce_embeddings_with_pca
from clustering import cluster_hdscan
import torch
import numpy as np
import time
from lsh_cluster import cluster_lsh
from cluster_faiss import hierarchical_cluster_faiss

s1 = time.time()

def main(input_dir, output_dir, max_seq_length, target_batch_size, use_bf16, pca_components, break_point=np.inf):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    input_files = glob.glob(input_dir)
    if not input_files:
        print(f"No files found matching the pattern: {input_dir}")
        return

    print("Parsing genome files...")
    for i, file in enumerate(input_files):
        if i > break_point:
            break
        file_name = os.path.basename(file)
        file_base = os.path.splitext(file_name)[0]
        print(f"Processing {file_name}")
        
        all_proteins_fasta = os.path.join(output_dir, f"{file_base}_all_proteins.fasta")
        if file_exists_check(all_proteins_fasta):
            print(f"All proteins file already exists: {all_proteins_fasta}")
        else:
            try:
                proteins = parse_single_file(file)
                if not proteins:
                    print(f"No proteins found in {file_name}")
                    continue

                write_proteins_to_fasta(proteins, all_proteins_fasta)
                print(f"All proteins written to {all_proteins_fasta}")
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                continue

        clustered_fasta = os.path.join(output_dir, f"{file_base}_clustered_proteins.fasta")
        if file_exists_check(clustered_fasta):
            print(f"Clustered proteins file already exists: {clustered_fasta}")
        else:
            print("Running CD-HIT on all proteins...")
            success = run_cd_hit(all_proteins_fasta, clustered_fasta)
    
            if success:
                print(f"CD-HIT clustering completed. Results saved to {clustered_fasta}")
                cluster_count = sum(1 for _ in SeqIO.parse(clustered_fasta, "fasta"))
                print(f"Number of protein clusters for {file_name}: {cluster_count}")
            else:
                print(f"CD-HIT clustering failed for {file_name}")
    print("CD-HIT clustering completed.")

    print("Creating embeddings for clustered proteins...")
    clustered_files = glob.glob(os.path.join(output_dir, "*_clustered_proteins.fasta"))
    embedding_files = []
    for clustered_file in clustered_files:
        print(f"\nProcessing: {clustered_file}")
        output_name = os.path.splitext(os.path.basename(clustered_file))[0] + "_embeddings_prot_t5.pt"
        output_file = os.path.join(output_dir, output_name)
        
        if file_exists_check(output_file):
            print(f"Embedding file already exists: {output_file}")
        else:
            total_proteins = calculate_embeddings(clustered_file, output_file, max_seq_length, target_batch_size, use_bf16)
            print("Combining embedding files...")
            combine_embedding_files(output_file)
            print(f"Completed processing {clustered_file}. Total proteins: {total_proteins}")
        
        embedding_files.append(output_file)
    print("\nEmbeddings creation completed.")

    print("Concatenating all embeddings...")
    all_embeddings_array, all_protein_ids = concatenate_embeddings(embedding_files)

    print(f"Total number of proteins: {len(all_protein_ids)}")
    print(f"Shape of concatenated embeddings: {all_embeddings_array.shape}")
    # save concatenated embeddings
    all_embeddings_file = os.path.join(output_dir, "all_embeddings.npz")
    np.savez(all_embeddings_file, embeddings=all_embeddings_array, protein_ids=all_protein_ids)
    # save protein ids
    all_protein_ids_file = os.path.join(output_dir, "all_protein_ids.txt")
    with open(all_protein_ids_file, "w") as f:
        f.write("\n".join(all_protein_ids))

    # print(f"Performing PCA reduction to {pca_components} components...")
    # reduced_embeddings = reduce_embeddings_with_pca(all_embeddings_array, n_components=pca_components)

    # # Save the reduced embeddings
    # reduced_file = os.path.join(output_dir, f"all_reduced_embeddings_pca{pca_components}.npz")
    # np.savez(reduced_file, embeddings=reduced_embeddings, protein_ids=all_protein_ids)
    # print(f"Reduced embeddings saved to: {reduced_file}")
    s_c = time.time()
    print("Clustering reduced embeddings...")
    # cluster_df = cluster_lsh(reduced_file, threshold=0.9, hash_size=6, num_hashtables=5)
    # cluster_df.to_csv(os.path.join(output_dir, "clustered_proteins_lsh.csv"), index=False)
    # cluster_df = cluster_hdscan(reduced_file)
    # cluster_df.to_csv(os.path.join(output_dir, "clustered_proteins.csv"), index=False)

    print("Clustering reduced embeddings...")
    cluster_df = hierarchical_cluster_faiss(all_embeddings_file, 
                                        similarity_threshold=0.95, 
                                        core_threshold=0.95, 
                                        shell_threshold=0.15, 
                                        gpu=True,
                                        n_neighbors=10)    
    cluster_df.to_csv(os.path.join(output_dir, "clustered_proteins.csv"), index=False)
    print(f"Clustering completed. Time taken: {round((time.time()-s_c)/60,2)} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pan-genome analysis, create protein embeddings, and reduce dimensionality")
    parser.add_argument("--input_dir", 
                        help="Glob pattern for input genome files (GFF3, GenBank, FASTA)",
                        default="/data/nilar/pan_genome/genomes/selection/*.gff3")
    parser.add_argument("--output_dir", 
                        help="Directory to save output files",
                        default="/data/nilar/pan_genome/full_analysis_output/9_genes_large")
    parser.add_argument("--max_seq_length", type=int, default=12500, 
                        help="Maximum sequence length for protein embeddings")
    parser.add_argument("--target_batch_size", type=int, default=1, 
                        help="Target batch size for embedding calculations")
    parser.add_argument("--use_bf16", action="store_false", default=True, 
                        help="Use bfloat16 precision for embeddings (default: True)")
    parser.add_argument("--pca_components", type=int, default=450,
                        help="Number of components to keep after PCA reduction")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.max_seq_length, args.target_batch_size, args.use_bf16, args.pca_components)

s2 = time.time()
print(f"Total time: {round((s2-s1)/60,2)} minutes")
