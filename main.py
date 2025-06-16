#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import shutil
from typing import Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO  # noqa: F401 – imported for downstream helpers

from cd_hit import add_cdhit_filtered_sequences_to_clusters, run_cd_hit

from file_parsing import (
    file_exists_check,
    parse_single_file,
    write_proteins_to_fasta,
)
from prot_T5 import (
    calculate_embeddings,
    combine_embedding_files,
    concatenate_embeddings,
)

import configure_of_cluster_st as conf
from cluster_faiss_fuzz_tools import cluster_at_thresholds
import cluster_faiss_fuzz_v2 as cff


# Pre‑calibrated similarity threshold ranges for selected models (low, high)
DEFAULT_THRESHOLDS: dict[str, Tuple[float, float]] = {
    "Rostlab/prot_t5_xl_uniref50": (0.7432, 0.9509),
    # "facebook/esm2_t6_8M_UR50D": (0.95, 0.999),
}

N_THRESHOLDS: int = 55  # number of similarity steps


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
        raise RuntimeError("Calibration failed: no valid embeddings to derive thresholds.")

    return float(np.mean(lows)), float(np.mean(ups))


def main(
    input_dir: str,
    output_dir: str,
    max_seq_length: int,
    acceleration: str | None,
    cd_hit: bool,
    break_point: int,
    calibrate: bool,
    model_name: str,
    sim_low: float | None,
    sim_high: float | None,
    clean: bool,
):
    times, t0 = [], time.time()
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    input_files = glob.glob(input_dir)
    if not input_files:
        print(f"No files found matching the pattern: {input_dir}")
        return

    print("Parsing genome files…")
    for i, file in enumerate(input_files):
        if i >= break_point:
            print("Break point reached – stopping.")
            break

        fn = os.path.basename(file)
        base = os.path.splitext(fn)[0]
        all_fasta = os.path.join(output_dir, f"{base}_all_proteins.fasta")
        clustered = os.path.join(output_dir, f"{base}_clustered_proteins.fasta")

        if not file_exists_check(all_fasta):
            try:
                proteins = parse_single_file(file)
                if not proteins:
                    print(f"No proteins found in {fn}")
                    continue
                write_proteins_to_fasta(proteins, all_fasta)
            except Exception as e:  
                print(f"Error processing {fn}: {e}")
                continue

        if cd_hit and not file_exists_check(clustered):
            ok, filt = run_cd_hit(all_fasta, clustered)
            if ok:
                with open(
                    os.path.join(output_dir, f"{base}_filtered_sequences.json"),
                    "w",
                ) as fh:
                    json.dump(filt, fh)
            else:
                print(f"CD‑HIT failed for {fn}")
        elif not cd_hit:
            clustered = all_fasta

    times.append(["CD‑HIT/parsing", round(time.time() - t0, 2)])


    print("Creating embeddings…")
    fasta_glob = "*_clustered_proteins.fasta" if cd_hit else "*_all_proteins.fasta"
    fasta_files = glob.glob(os.path.join(output_dir, fasta_glob))

    embedding_files, t_emb = [], time.time()
    for fasta in fasta_files:
        out_pt = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(fasta))[0]
            + f"_embeddings_{model_name.split('/')[-1]}.pt",
        )
        if not file_exists_check(out_pt):
            calculate_embeddings(
                fasta_file=fasta,
                output_file=out_pt,
                max_seq_length=max_seq_length,
                max_batch_tokens=12_500,
                max_batch_size=128,
                acceleration=acceleration,
                model_name=model_name,
            )
            combine_embedding_files(out_pt)
        embedding_files.append(out_pt)

    print(f"Embeddings completed in {round((time.time()-t_emb)/60,2)} min")
    times.append(["Embeddings", round(time.time() - t_emb, 2)])


    all_emb_arr, all_ids = concatenate_embeddings(embedding_files)
    npz_file = os.path.join(output_dir, "all_embeddings.npz")
    np.savez(npz_file, embeddings=all_emb_arr, protein_ids=all_ids)

    with open(os.path.join(output_dir, "all_protein_ids.txt"), "w") as fh:
        fh.write("\n".join(all_ids))

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
        print(
            "WARNING: No default similarity thresholds for model "
            f"'{model_name}'. Provide --sim_low/--sim_high or run the "
            "configuration script (configure_of_cluster_st.py) to derive them."
        )
        sys.exit(1)

    sim_thresholds = np.linspace(low, up, N_THRESHOLDS)
    print(
        f"Final similarity thresholds: {low:.4f} → {up:.4f} ({len(sim_thresholds)} steps)"
    )


    t_clust = time.time()
    columns = cff.cluster_faiss_parallel(
        npz_file,
        sim_thresholds,
        core_threshold=0.95,
        shell_threshold=0.15,
        cpu=False,
        k=1_000,
        batch_size=10_000,
        n_closest=4,
    )
    cluster_df = pd.DataFrame(columns)


    if cd_hit:
        filt_files = glob.glob(os.path.join(output_dir, "*_filtered_sequences.json"))
        all_filt: dict[str, str] = {}
        for ff in filt_files:
            with open(ff) as fh:
                all_filt.update(json.load(fh))
        cluster_df = add_cdhit_filtered_sequences_to_clusters(cluster_df, all_filt)

    # Add cluster sizes and write
    cluster_df["cluster_size"] = (
        cluster_df["cluster_id"].map(cluster_df["cluster_id"].value_counts())
    )
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Pan‑genome analysis with flexible encoder models and "
            "FAISS/Levenshtein clustering."
        )
    )
    parser.add_argument(
        "--input_dir",
        default="/data/nilar/pan_genome/full_analysis_output/bp_25_all_peps_seq_large/*.fasta",
        help="Glob pattern for input genome annotation files (*.gff3, *.fasta)",
    )
    parser.add_argument(
        "--output_dir",
        default="/data/nilar/pan_genome/full_analysis_output/bp_25_all_peps_seq_large/esm/",
        help="Directory to write all output files",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=5_500,
        help="Maximum sequence length fed to the encoder",
    )
    parser.add_argument(
        "--acceleration",
        choices=["onnx", "quantization", None],
        default="onnx",
        help="Encoder acceleration backend (onnx / quantization / none)",
    )
    parser.add_argument(
        "--break_point",
        type=int,
        default=np.inf,
        help="Process at most N genome files (for debugging)",
    )
    parser.add_argument(
        "--cd_hit",
        action="store_true",
        help="Enable CD‑HIT pre‑filtering of redundant proteins",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Re‑calibrate similarity thresholds using reference FASTA files",
    )
    parser.add_argument(
        "--model_name",
        default="Rostlab/prot_t5_xl_uniref50",
        help="HuggingFace encoder model identifier - facebook/esm2_t6_8M_UR50D",
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
    args = parser.parse_args()
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        acceleration=args.acceleration,
        cd_hit=args.cd_hit,
        break_point=args.break_point,
        calibrate=args.calibrate,
        model_name=args.model_name,
        sim_low=args.sim_low,
        sim_high=args.sim_high,
        clean=args.clean,
    )

