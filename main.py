import os
import argparse
import glob
from Bio import SeqIO
from cd_hit import run_cd_hit
from file_parsing import parse_single_file
from prot_T5 import calculate_embeddings, combine_embedding_files
from pca_reduction import reduce_embeddings_with_pca
# from clustering import hdbscan
import torch
import numpy as np

def file_exists_check(file_path):
    return os.path.isfile(file_path) and os.path.getsize(file_path) > 0

def write_proteins_to_fasta(proteins, output_file):
    with open(output_file, 'w') as f:
        for seq, protein_id, source_file in proteins:
            f.write(f">{protein_id}|{source_file}\n{seq}\n")

def concatenate_embeddings(embedding_files):
    all_embeddings = []
    all_protein_ids = []
    for file in embedding_files:
        print(f"Reading embeddings from: {file}")
        embeddings = torch.load(file)
        for protein_id, embedding in embeddings.items():
            all_embeddings.append(embedding.numpy())
            all_protein_ids.append(protein_id)
    return np.array(all_embeddings), all_protein_ids

def main(input_dir, output_dir, max_seq_length, target_batch_size, use_bf16, pca_components, break_point=2):
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

    # Concatenate all embeddings
    print("Concatenating all embeddings...")
    all_embeddings_array, all_protein_ids = concatenate_embeddings(embedding_files)

    print(f"Total number of proteins: {len(all_protein_ids)}")
    print(f"Shape of concatenated embeddings: {all_embeddings_array.shape}")

    # Perform PCA reduction on the concatenated embeddings
    print(f"Performing PCA reduction to {pca_components} components...")
    reduced_embeddings = reduce_embeddings_with_pca(all_embeddings_array, n_components=pca_components)

    # Save the reduced embeddings
    reduced_file = os.path.join(output_dir, f"all_reduced_embeddings_pca{pca_components}.npz")
    np.savez(reduced_file, embeddings=reduced_embeddings, protein_ids=all_protein_ids)
    print(f"Reduced embeddings saved to: {reduced_file}")

    print("Pan-genome analysis, embedding creation, and dimensionality reduction completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pan-genome analysis, create protein embeddings, and reduce dimensionality")
    parser.add_argument("--input_dir", 
                        help="Glob pattern for input genome files (GFF3, GenBank, FASTA)",
                        default="/data/nilar/pan_genome/genomes/*.gff3")
    parser.add_argument("--output_dir", 
                        help="Directory to save output files",
                        default="/data/nilar/pan_genome/full_analysis_output")
    parser.add_argument("--max_seq_length", type=int, default=12500, 
                        help="Maximum sequence length for protein embeddings")
    parser.add_argument("--target_batch_size", type=int, default=2, 
                        help="Target batch size for embedding calculations")
    parser.add_argument("--use_bf16", action="store_true", 
                        help="Use bfloat16 precision for embeddings if available")
    parser.add_argument("--pca_components", type=int, default=450,
                        help="Number of components to keep after PCA reduction")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.max_seq_length, args.target_batch_size, args.use_bf16, args.pca_components)