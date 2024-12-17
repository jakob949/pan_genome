import os
import argparse
import glob
from Bio import SeqIO
import numpy as np
import time
import json
from cd_hit import run_cd_hit, add_cdhit_filtered_sequences_to_clusters
from file_parsing import parse_single_file, file_exists_check, write_proteins_to_fasta
from prot_T5 import calculate_embeddings, combine_embedding_files, concatenate_embeddings
from pca_reduction import reduce_embeddings_with_pca
from cluster_faiss import hierarchical_cluster_faiss_parallel

s1 = time.time()
times = []

def main(input_dir, output_dir, max_seq_length, acceleration, cd_hit, break_point=np.inf, pca_components=450):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    input_files = glob.glob(input_dir)
    if not input_files:
        print(f"No files found matching the pattern: {input_dir}")
        return

    print("Parsing genome files...")
    for i, file in enumerate(input_files):
        if i >= break_point:
            print(f"Break point reached. Stopping.")
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
        elif cd_hit:
            print("Running CD-HIT on all proteins...")
            success, filtered_seqs = run_cd_hit(all_proteins_fasta, clustered_fasta)
            
            # Store the filtered sequences mapping for later use
            filtered_seqs_file = os.path.join(output_dir, f"{file_base}_filtered_sequences.json")
            with open(filtered_seqs_file, 'w') as f:
                json.dump(filtered_seqs, f)
    
            if success:
                print(f"CD-HIT clustering completed. Results saved to {clustered_fasta}")
                cluster_count = sum(1 for _ in SeqIO.parse(clustered_fasta, "fasta"))
                print(f"Number of protein clusters for {file_name}: {cluster_count}")
            else:
                print(f"CD-HIT clustering failed for {file_name}")
        else:
            print("CD-HIT clustering is skipped.")



    print("CD-HIT clustering completed.")
    times.append(["CD-hit", round(s1-time.time(), 2)])
    s2 = time.time()
    
    print("Creating embeddings for clustered proteins...")
    if cd_hit:
        clustered_files = glob.glob(os.path.join(output_dir, "*_clustered_proteins.fasta"))
    else:
        clustered_files = glob.glob(os.path.join(output_dir, "*_all_proteins.fasta"))
    
    embedding_files = []
    for clustered_file in clustered_files:
        print(f"\nProcessing: {clustered_file}")
        output_name = os.path.splitext(os.path.basename(clustered_file))[0] + "_embeddings_prot_t5.pt"
        output_file = os.path.join(output_dir, output_name)
        
        if file_exists_check(output_file):
            print(f"Embedding file already exists: {output_file}")
        else:
            total_proteins = calculate_embeddings(
                                fasta_file=clustered_file,
                                output_file=output_file,
                                max_seq_length=max_seq_length,
                                max_batch_tokens=14500,
                                max_batch_size=128,  
                                acceleration=acceleration
                            )
            print("Combining embedding files...")
            combine_embedding_files(output_file)
            print(f"Completed processing {clustered_file}. Total proteins: {total_proteins}")
        
        embedding_files.append(output_file)
    print("\nEmbeddings creation completed.")
    times.append(["Embeddings", round(s2-time.time(), 2)])
    print("Concatenating all embeddings...")
    all_embeddings_array, all_protein_ids = concatenate_embeddings(embedding_files)

    print(f"Total number of proteins: {len(all_protein_ids)}")
    print(f"Shape of concatenated embeddings: {all_embeddings_array.shape}")
    # # save concatenated embeddings
    all_embeddings_file = os.path.join(output_dir, "all_embeddings.npz")
    np.savez(all_embeddings_file, embeddings=all_embeddings_array, protein_ids=all_protein_ids)
    # save protein ids
    all_protein_ids_file = os.path.join(output_dir, "all_protein_ids.txt")
    with open(all_protein_ids_file, "w") as f:
        f.write("\n".join(all_protein_ids))
    s_c = time.time()
    
    print("Clustering reduced embeddings...")
    cluster_df = hierarchical_cluster_faiss_parallel(all_embeddings_file, 
                                        similarity_threshold=0.99, 
                                        core_threshold=0.95, 
                                        shell_threshold=0.15, 
                                        cpu=False, 
                                        k=1000,
                                        batch_size=10000,
                                        n_closest=4)    

    
    all_filtered_seqs = {}
    filtered_seq_files = glob.glob(os.path.join(output_dir, "*_filtered_sequences.json"))
    for filtered_file in filtered_seq_files:
        with open(filtered_file, 'r') as f:
            all_filtered_seqs.update(json.load(f))
    
    # Add filtered sequences to the clustering results
    cluster_df = add_cdhit_filtered_sequences_to_clusters(cluster_df, all_filtered_seqs)
    
    # Add stats 
    cluster_sizes = cluster_df['cluster'].value_counts()
    cluster_df['cluster_size'] = cluster_df['cluster'].map(cluster_sizes)

    # Save the updated DataFrame
    cluster_df.to_csv(os.path.join(output_dir, f"clustered_proteins.csv"), index=False)

    print(f"Clustering completed. Time taken: {round((time.time()-s_c)/60,2)} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pan-genome analysis, create protein embeddings, and reduce dimensionality")

    parser.add_argument("--input_dir", 
                        help="Glob pattern for input genome files (GFF3, GenBank, FASTA)",
                        default="/data/nilar/pan_genome/genomes/*.gff3")
    parser.add_argument("--output_dir", 
                        help="Directory to save output files",
                        default="/data/nilar/pan_genome/full_analysis_output/NBC0000_special")
    parser.add_argument("--max_seq_length", type=int, default=7500, 
                        help="Maximum sequence length for protein embeddings")
    parser.add_argument("--acceleration", default="None", 
                        help="Use ONNX <onnx> or <quantization> for acceleration")
    parser.add_argument("--pca_components", type=int, default=450,
                        help="Number of components to keep after PCA reduction")
    parser.add_argument("--break_point", type=int, default=np.inf,
                        help="Number of files to process before stopping")
    parser.add_argument("--cd_hit", action="store_true",
                        help="Run CD-HIT clustering on proteins - to remove highly similar sequences")
    
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.max_seq_length, args.acceleration, args.cd_hit, args.break_point, args.pca_components)

s2 = time.time()
print(f"Total time: {round((s2-s1)/60,2)} minutes")
with open(os.path.join(args.output_dir, "times.txt"), "w") as f:
    for name, seconds in times:
        f.write(f"{name}: {seconds} seconds\n")
    f.write(f"Total time: {round((s2-s1),2)} seconds\n")
