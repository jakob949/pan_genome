#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import panlm.cluster_faiss_fuzz_v2_7 as cff
from panlm.cd_hit import add_cdhit_filtered_sequences_to_clusters, run_cd_hit
from panlm.embs import (
    calculate_embeddings,
    concatenate_embeddings,
)
from panlm.file_parsing import (
    file_exists_check,
    parse_single_file,
    write_proteins_to_fasta,
)
from panlm.heaps_law import compute_heaps_law, plot_heaps_biplot, plot_heaps_law

# Pre‑calibrated mean and sd parameters for fuzzy clustering
DEFAULT_PARAMS: dict[str, Tuple[float, float]] = {
    "Rostlab/prot_t5_xl_uniref50": (0.856, 0.1259),
    "Synthyra/ESM2-3B": (0.9935, 0.0076),
    # "Synthyra/ESMplusplus_large": (0.848, 0.024),
}
N_THRESHOLDS: int = 25  # number of similarity steps


def main(
    input_dir: str,
    output_dir: str,
    max_seq_length: int,
    acceleration: str | None,
    disable_cd_hit: bool,
    break_point: int,
    model_name: str,
    clean: bool,
    lora_adapter_path: str | None,
    heaps_law: bool,
    clust_method: str,
    min_cluster_size: int,
    pca_dim: int | None,
    eps: float,
    mean: float | None = None,
    sd: float | None = None,
    use_ann: bool = False,
    nprobe: int = 32,
    nlist: int | None = None,
):
    times, t0 = [], time.time()
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    input_files = glob.glob(input_dir)
    if not input_files:
        print(f"No files found matching the pattern: {input_dir}")
        return

    fasta_for_embedding = None
    filtered_json_path = os.path.join(output_dir, "all_filtered_sequences.json")

    if not disable_cd_hit:
        print("CD-HIT mode enabled: Processing each genome individually.")
        all_representative_proteins = []
        all_filtered_sequences_map = {}

        temp_fasta_dir = os.path.join(output_dir, "temp_genome_fastas")
        os.makedirs(temp_fasta_dir, exist_ok=True)
        print(f"Using temporary directory: {temp_fasta_dir}")

        if break_point < len(input_files):
            print(f"Break point reached - processing first {break_point} files.")
            input_files = input_files[:break_point]

        for file in tqdm(input_files, desc="Parsing and running CD-HIT per file"):
            try:
                # Parse proteins from a single file
                proteins = parse_single_file(file)
                if not proteins:
                    print(f"No proteins found in {file}, skipping.")
                    continue

                # Write proteins to a temporary, per-genome FASTA
                file_basename = os.path.basename(file)
                temp_input_fasta = os.path.join(
                    temp_fasta_dir, f"{file_basename}.fasta"
                )
                write_proteins_to_fasta(proteins, temp_input_fasta)

                # Run CD-HIT on that single FASTA
                temp_output_fasta = os.path.join(
                    temp_fasta_dir, f"{file_basename}_clustered.fasta"
                )
                ok, filtered_map = run_cd_hit(temp_input_fasta, temp_output_fasta)

                if not ok:
                    print(f"CD-HIT failed on {file}. Skipping this file's proteins.")
                    continue

                # Collect representative proteins
                reps = parse_single_file(temp_output_fasta)
                fixed_reps = []
                for seq, combined_id, _ in reps:
                    # The ID in temp output has the format: protein_id|original_filename
                    if "|" in combined_id:
                        orig_id, orig_file = combined_id.rsplit("|", 1)
                        fixed_reps.append((seq, orig_id, orig_file))
                    else:
                        fixed_reps.append((seq, combined_id, file_basename))
                all_representative_proteins.extend(fixed_reps)

                # Collect the map of filtered sequences
                all_filtered_sequences_map.update(filtered_map)

            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

        if not all_representative_proteins:
            print("No representative proteins found after CD-HIT. Exiting.")
            return

        print(
            f"Total representative proteins from all genomes: {len(all_representative_proteins)}"
        )
        print(f"Total filtered (redundant) proteins: {len(all_filtered_sequences_map)}")

        # Write the combined list of *all representatives* to the file for embedding
        fasta_for_embedding = os.path.join(
            output_dir, "all_proteins_representatives.fasta"
        )
        write_proteins_to_fasta(all_representative_proteins, fasta_for_embedding)

        # Save the combined map of *all filtered sequences* for later
        with open(filtered_json_path, "w") as fh:
            json.dump(all_filtered_sequences_map, fh)

        # Clean up temporary FASTA directory
        print(f"Cleaning up temporary directory: {temp_fasta_dir}")
        shutil.rmtree(temp_fasta_dir)

    else:
        # Parse all files directly if CD-HIT is disabled
        print("CD-HIT mode disabled. Parsing and combining all files.")
        all_proteins = []

        if break_point < len(input_files):
            print(f"Break point reached - processing first {break_point} files.")
            input_files = input_files[:break_point]

        for file in tqdm(input_files, desc="Parsing files"):
            try:
                proteins = parse_single_file(file)
                if proteins:
                    all_proteins.extend(proteins)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

        if not all_proteins:
            print("No proteins found in any input file. Exiting.")
            return

        print(f"Found a total of {len(all_proteins)} proteins.")

        combined_fasta_path = os.path.join(output_dir, "all_proteins_combined.fasta")
        write_proteins_to_fasta(all_proteins, combined_fasta_path)

        fasta_for_embedding = combined_fasta_path
        # No filtered JSON path in this case
        filtered_json_path = None

    times.append(["Parsing & CD-HIT", round(time.time() - t0, 2)])

    print("Creating embeddings...")
    t_emb = time.time()

    model_suffix = model_name.replace("/", "_")
    final_embedding_file = os.path.join(output_dir, f"all_embeddings_{model_suffix}.pt")

    if not file_exists_check(final_embedding_file):
        print(
            f"Model: {model_name}, Input FASTA: {os.path.basename(fasta_for_embedding)}"
        )
        calculate_embeddings(
            fasta_file=fasta_for_embedding,
            output_file=final_embedding_file,
            max_seq_length=max_seq_length,
            max_batch_tokens=12_500,
            max_batch_size=128,
            acceleration=acceleration,
            model_name=model_name,
            lora_adapter_path=lora_adapter_path,
        )

    else:
        print(
            f"Found existing embeddings file: {os.path.basename(final_embedding_file)}. Skipping calculation."
        )

    print(f"Embeddings completed in {round((time.time() - t_emb) / 60, 2)} min")
    times.append(["Embeddings", round(time.time() - t_emb, 2)])

    all_emb_arr, all_ids = concatenate_embeddings([final_embedding_file])
    npz_file = os.path.join(output_dir, "all_embeddings.npz")
    np.savez(npz_file, embeddings=all_emb_arr, protein_ids=all_ids)

    with open(os.path.join(output_dir, "all_protein_ids.txt"), "w") as fh:
        fh.write("\n".join(all_ids))

    # --- Threshold Logic ---
    # Derive threshold range if fuzzy clustering is used.
    if clust_method == "fuzzy":
        if mean is not None and sd is not None:
            print(
                f"Using user-supplied mean ({mean}) and sd ({sd}) for fuzzy clustering."
            )
        elif model_name in DEFAULT_PARAMS:
            mean, sd = DEFAULT_PARAMS[model_name]
            print(
                f"Using pre-calibrated mean ({mean}) and sd ({sd}) for model {model_name}."
            )
        else:
            mean, sd = 0.8, 0.05
            print(
                f"WARNING: No default mean and sd for model '{model_name}'. "
                "Provide --mean and --sd. Using fallback values: mean=0.8, sd=0.05."
            )

        low, up = max(0.0, mean - 3 * sd), min(1.0, mean + 3 * sd)
        sim_thresholds = np.linspace(low, up, N_THRESHOLDS)
        print(
            f"Final similarity thresholds: {low:.4f} → {up:.4f} ({N_THRESHOLDS} steps)"
        )
    else:
        # Dummy thresholds for other clustering methods
        sim_thresholds = np.array([0.5])
        print(f"Clustering method is {clust_method}, skipping strict threshold checks.")

    t_clust = time.time()
    print(f"Starting clustering using clustering method: {clust_method}")
    columns, _ = cff.cluster_faiss_parallel(
        npz_file,
        sim_thresholds,
        core_threshold=0.95,
        shell_threshold=0.15,
        cpu=False,
        k=1_500,
        batch_size=100000,
        mean=mean,
        sd=sd,
        clust_method=clust_method,
        min_cluster_size=min_cluster_size,
        pca_dim=pca_dim,
        eps=eps,
        use_ann=use_ann,
        nprobe=nprobe,
        nlist=nlist,
    )
    cluster_df = pd.DataFrame(columns)

    if not disable_cd_hit:
        if file_exists_check(filtered_json_path):
            print(
                f"Re-integrating {len(all_filtered_sequences_map)} filtered sequences..."
            )
            with open(filtered_json_path) as fh:
                all_filt = json.load(fh)
            cluster_df = add_cdhit_filtered_sequences_to_clusters(cluster_df, all_filt)
        else:
            print(
                "Warning: CD-HIT enabled, but filtered sequences JSON not found during post-processing."
            )

    cluster_df["cluster_size"] = cluster_df["cluster_id"].map(
        cluster_df["cluster_id"].value_counts()
    )

    if heaps_law:
        results = compute_heaps_law(cluster_df)
        plot_heaps_biplot(results, output_dir)
        plot_heaps_law(results, output_dir)
    final_output_path = os.path.join(output_dir, "clustered_proteins.csv")
    cluster_df.to_csv(final_output_path, index=False)
    print(f"Final cluster file saved to: {final_output_path}")

    print(
        f"Clustering finished in {round((time.time() - t_clust) / 60, 2)} min "
        f"(total proteins = {len(cluster_df)})"
    )
    times.append(["Clustering", round(time.time() - t_clust, 2)])

    # Timing summary
    total = round(time.time() - t0, 2)
    with open(os.path.join(output_dir, "times.txt"), "w") as fh:
        for name, secs in times:
            fh.write(f"{name}: {secs} sec\n")
        fh.write(f"Total: {total} sec\n")
    print(f"Total runtime: {round(total / 60, 2)} min")

    # Optional ONNX model cleanup (if not cleaning everything)
    if acceleration == "onnx" and not clean:
        safe_name = model_name.replace("/", "_")
        onnx_dir = os.path.join(output_dir, f"{safe_name}_onnx")
        if os.path.exists(onnx_dir):
            print(f"Cleaning up ONNX model directory: {onnx_dir}")
            shutil.rmtree(onnx_dir, ignore_errors=True)

    # Optional cleanup
    if clean:
        for f in os.listdir(output_dir):
            if f != "clustered_proteins.csv":
                path = os.path.join(output_dir, f)
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
        print("Removed intermediate files, kept only clustered_proteins.csv")


def cli():
    parser = argparse.ArgumentParser(
        description=(
            "Pan‑genome analysis with flexible encoder models and "
            "FAISS/Levenshtein clustering."
        )
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Glob pattern for input genome annotation files (*.gff3, *.fasta, )",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write all output files",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4_500,
        help="Maximum sequence length fed to the encoder",
    )
    parser.add_argument(
        "--acceleration",
        choices=["onnx", "quantization", "None"],
        default=None,
        help="Encoder acceleration backend (onnx / quantization / none)",
    )
    parser.add_argument(
        "--break_point",
        type=int,
        default=np.inf,
        help="Process at most N genome files (for debugging)",
    )
    parser.add_argument(
        "--disable_cd_hit",
        action="store_true",
        help="Disable CD-HIT pre-filtering of redundant proteins. (CD-HIT is enabled by default)",
    )

    parser.add_argument(
        "--model_name",
        default="Synthyra/ESM2-3B",
        help="HuggingFace encoder model identifier - Synthyra/ESM2-3B",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove intermediate files, keep only final CSV file",
    )
    parser.add_argument(
        "--lora_adapter_path",
        help="Path to the directory containing fine-tuned LoRA adapter weights",
    )
    parser.add_argument(
        "--heaps_law",
        action="store_true",
        help="Compute Heaps' law, and output the alpha, gamma parameter, and plot (number of clusters vs number of genomes)",
    )
    parser.add_argument(
        "--clust_method",
        choices=["fuzzy", "hdbscan", "dbscan"],
        default="fuzzy",
        help="Clustering method to use: 'fuzzy', 'hdbscan', or 'dbscan'.",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=2,
        help="Minimum cluster size for HDBSCAN.",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=455,
        help="Target dimension for PCA reduction. Set to 0 to disable PCA.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.0475,
        help="Distance threshold (epsilon). For DBSCAN, this defines the strict global maximum radius for neighborhood formation. For HDBSCAN, this acts as the cluster_selection_epsilon, preventing cluster splits below this distance during hierarchical tree condensation. Default 0.1",
    )
    parser.add_argument(
        "--mean",
        type=float,
        default=None,
        help="Mean for fuzzy clustering threshold generation",
    )
    parser.add_argument(
        "--sd",
        type=float,
        default=None,
        help="Standard deviation for fuzzy clustering threshold generation",
    )
    parser.add_argument(
        "--use_ann",
        action="store_true",
        help="Use Approximate Nearest Neighbors (ANN) via FAISS IVF index. Otherwize exact nearest neighbors are used",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=32,
        help="Number of centroids to probe for IVF index (used when --use_ann is set)",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=None,
        help="Number of centroids for IVF index (default: 4 * sqrt(N))",
    )

    args = parser.parse_args()

    pca_dim_to_pass = args.pca_dim if args.pca_dim > 0 else None

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        acceleration=args.acceleration,
        disable_cd_hit=args.disable_cd_hit,
        break_point=args.break_point,
        model_name=args.model_name,
        clean=args.clean,
        lora_adapter_path=args.lora_adapter_path,
        heaps_law=args.heaps_law,
        clust_method=args.clust_method,
        min_cluster_size=args.min_cluster_size,
        pca_dim=pca_dim_to_pass,
        eps=args.eps,
        mean=args.mean,
        sd=args.sd,
        use_ann=args.use_ann,
        nprobe=args.nprobe,
        nlist=args.nlist,
    )


if __name__ == "__main__":
    cli()
