#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from typing import Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

import panlm.cluster_faiss_fuzz_v2_4 as cff
import panlm.configure_of_cluster_st as conf
from panlm.cd_hit import add_cdhit_filtered_sequences_to_clusters, run_cd_hit
from panlm.cluster_faiss_fuzz_tools import cluster_at_thresholds
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

# Pre‑calibrated similarity threshold ranges for selected models (low, high)
DEFAULT_THRESHOLDS: dict[str, Tuple[float, float]] = {
    "Rostlab/prot_t5_xl_uniref50": (0.825, 1.0),
    # Rostlab/prot_t5_xl_uniref50:(mean = 0,943, SD = 0,02)
    # "Synthyra/ESM2-3B": (mean = 0.796, SD = 0.046),
    # "Synthyra/ESMplusplus_large": (mean = 0.848, SD = 0.024),
}
# 0.825,1.0,25"
N_THRESHOLDS: int = 25  # number of similarity steps


def calibrate_thresholds(pt_files, lower_bound=0.015, upper_bound=0.25):
    """Derive a similarity range for *pt_files* that yields roughly
    *lower_bound* ≤ cluster-share ≤ *upper_bound*."""

    thresholds = np.linspace(0.70, 1.005, 75)
    cpu, k, batch_size = False, 100, 10_000

    lows, ups = [], []
    for pt in pt_files:
        try:
            emb, _ = conf.concatenate_embeddings_cpu([pt])
        except ValueError as e:
            print(f"Warning: skipping {pt} due to embedding dimension mismatch: {e}")
            continue

        res = cluster_at_thresholds(emb, thresholds, cpu, k, batch_size)
        n_emb = len(emb)
        pct = np.asarray([len(set(lbls)) for lbls in res.values()]) / n_emb

        lows.append(thresholds[np.argmin(np.abs(pct - lower_bound))])
        ups.append(thresholds[np.argmin(np.abs(pct - upper_bound))])

    if not lows:
        raise RuntimeError(
            "Calibration failed: no valid embeddings to derive thresholds."
        )

    return float(np.mean(lows)), float(np.mean(ups))


def main(
    input_dir: str,
    output_dir: str,
    max_seq_length: int,
    acceleration: str | None,
    disable_cd_hit: bool,
    break_point: int,
    calibrate: bool,
    model_name: str,
    sim_low: float | None,
    sim_high: float | None,
    clean: bool,
    lora_adapter_path: str | None,
    heaps_law: bool,
    algorithm: str,
    min_cluster_size: int,
    pca_dim: int | None,
    eps: float,
):
    times, t0 = [], time.time()
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    input_files = glob.glob(input_dir)
    if not input_files:
        print(f"No files found matching the pattern: {input_dir}")
        return

    # --- START: Modified Logic ---

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
                # 1. Parse proteins from a single file
                proteins = parse_single_file(file)
                if not proteins:
                    print(f"No proteins found in {file}, skipping.")
                    continue

                # 2. Write proteins to a temporary, per-genome FASTA
                file_basename = os.path.basename(file)
                temp_input_fasta = os.path.join(
                    temp_fasta_dir, f"{file_basename}.fasta"
                )
                write_proteins_to_fasta(proteins, temp_input_fasta)

                # 3. Run CD-HIT on that single FASTA
                temp_output_fasta = os.path.join(
                    temp_fasta_dir, f"{file_basename}_clustered.fasta"
                )
                ok, filtered_map = run_cd_hit(temp_input_fasta, temp_output_fasta)

                if not ok:
                    print(f"CD-HIT failed on {file}. Skipping this file's proteins.")
                    continue

                # 4. Collect representative proteins
                # We parse the FASTA file that CD-HIT *created*, which contains only representatives
                reps = parse_single_file(temp_output_fasta)
                all_representative_proteins.extend(reps)

                # 5. Collect the map of filtered sequences
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
        # This is the original logic, executed if --disable_cd_hit is used
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

    print(f"Embeddings completed in {round((time.time()-t_emb)/60,2)} min")
    times.append(["Embeddings", round(time.time() - t_emb, 2)])

    all_emb_arr, all_ids = concatenate_embeddings([final_embedding_file])
    npz_file = os.path.join(output_dir, "all_embeddings.npz")
    np.savez(npz_file, embeddings=all_emb_arr, protein_ids=all_ids)

    with open(os.path.join(output_dir, "all_protein_ids.txt"), "w") as fh:
        fh.write("\n".join(all_ids))

    # --- Threshold Logic ---
    # Note: Even if using HDBSCAN, we keep this logic to allow the script to pass valid arguments,
    # though HDBSCAN ignores the thresholds inside cluster_faiss_parallel.
    if sim_low is not None and sim_high is not None:
        low, up = sim_low, sim_high
        print(
            f"Using user‑supplied similarity thresholds: {low:.4f} → {up:.4f} ({N_THRESHOLDS} steps)"
        )
    elif calibrate:
        print("Downloading calibration data & computing thresholds…")
        calib_pt = conf.ensure_calibration_embeddings(
            dest_dir=os.path.join(output_dir, "fastas_for_calibration"),
            acceleration=acceleration,
            model_name=model_name,
        )
        low, up = calibrate_thresholds(calib_pt)
    elif model_name in DEFAULT_THRESHOLDS:
        low, up = DEFAULT_THRESHOLDS[model_name]
        print(
            f"Using pre‑calibrated thresholds for {model_name}: {low:.4f} → {up:.4f} ({N_THRESHOLDS} steps)"
        )
    else:
        # If using HDBSCAN, we can tolerate missing threshold defaults,
        # but for safety, we exit if 'fuzzy' is chosen and no thresholds exist.
        if algorithm == "fuzzy":
            print(
                "WARNING: No default similarity thresholds for model "
                f"'{model_name}'. Provide --sim_low/--sim_high or run the "
                "configuration script (configure_of_cluster_st.py) to derive them."
            )
            sys.exit(1)
        else:
            # For HDBSCAN, use dummies if not provided, just to pass the variable creation
            low, up = 0.0, 1.0
            print(f"Algorithm is {algorithm}, skipping strict threshold checks.")

    sim_thresholds = np.linspace(low, up, N_THRESHOLDS)
    if algorithm == "fuzzy":
        print(
            f"Final similarity thresholds: {low:.4f} → {up:.4f} ({len(sim_thresholds)} steps)"
        )

    t_clust = time.time()
    print(f"Starting clustering using algorithm: {algorithm}")
    columns, _ = cff.cluster_faiss_parallel(
        npz_file,
        sim_thresholds,
        core_threshold=0.95,
        shell_threshold=0.15,
        cpu=False,
        k=1_500,
        batch_size=100000,
        mean=0.923,  # Corrected from weight_mean to mean
        sd=0.2,  # Corrected from weight_sd to sd
        algorithm=algorithm,
        min_cluster_size=min_cluster_size,
        pca_dim=pca_dim,
        eps=eps,
    )
    cluster_df = pd.DataFrame(columns)

    # --- START: Modified Post-Clustering ---
    # Now, this step will only run if disable_cd_hit was False and the file was created
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
    # --- END: Modified Post-Clustering ---

    cluster_df["cluster_size"] = cluster_df["cluster_id"].map(
        cluster_df["cluster_id"].value_counts()
    )

    if heaps_law:
        results = compute_heaps_law(cluster_df)
        plot_heaps_biplot(results, output_dir)
        plot_heaps_law(results, output_dir)

    cluster_df.to_csv(os.path.join(output_dir, "clustered_proteins.csv"), index=False)

    print(
        f"Clustering finished in {round((time.time()-t_clust)/60,2)} min "
        f"(total proteins = {len(cluster_df)})"
    )
    times.append(["Clustering", round(time.time() - t_clust, 2)])

    # Timing summary
    total = round(time.time() - t0, 2)
    with open(os.path.join(output_dir, "times.txt"), "w") as fh:
        for name, secs in times:
            fh.write(f"{name}: {secs} sec\n")
        fh.write(f"Total: {total} sec\n")
    print(f"Total runtime: {round(total/60,2)} min")

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
        default="/data/nilar/go_cafa5/all_proteins_combined.fasta",
        help="Glob pattern for input genome annotation files (*.gff3, *.fasta, )",
    )
    parser.add_argument(
        "--output_dir",
        default="/data/nilar/pan_genome/synthetic_dataset/indivual_genomes/our/test_output/del",
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
        "--calibrate",
        action="store_true",
        help="Re‑calibrate similarity thresholds using reference FASTA files",
    )
    parser.add_argument(
        "--model_name",
        default="Synthyra/ESM2-3B",
        help="HuggingFace encoder model identifier - Synthyra/ESM2-3B",
    )
    parser.add_argument(
        "--sim_low",
        type=float,
        help="Lower bound of similarity threshold range (overrides default)",
    )
    parser.add_argument(
        "--sim_high",
        type=float,
        help="Upper bound of similarity threshold range (overrides default)",
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
    # --- New Arguments ---
    parser.add_argument(
        "--algorithm",
        choices=["fuzzy", "hdbscan", "dbscan"],
        default="fuzzy",
        help="Clustering algorithm to use: 'fuzzy' (original), 'hdbscan', or 'dbscan'.",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=5,
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
        default=0.1,
        help="Distance threshold (epsilon). For DBSCAN, this defines the strict global maximum radius for neighborhood formation. For HDBSCAN, this acts as the cluster_selection_epsilon, preventing cluster splits below this distance during hierarchical tree condensation. Default 0.1",
    )
    # ---------------------

    args = parser.parse_args()

    pca_dim_to_pass = args.pca_dim if args.pca_dim > 0 else None

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        acceleration=args.acceleration,
        disable_cd_hit=args.disable_cd_hit,
        break_point=args.break_point,
        calibrate=args.calibrate,
        model_name=args.model_name,
        sim_low=args.sim_low,
        sim_high=args.sim_high,
        clean=args.clean,
        lora_adapter_path=args.lora_adapter_path,
        heaps_law=args.heaps_law,
        algorithm=args.algorithm,
        min_cluster_size=args.min_cluster_size,
        pca_dim=pca_dim_to_pass,
        eps=args.eps,
    )
