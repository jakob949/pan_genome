#!/usr/bin/env python3
import io
import shutil
import urllib.request
import zipfile
import os
import glob
import argparse

import numpy as np
import torch
from collections import OrderedDict, Counter # Added Counter here

from cluster_faiss_fuzz_tools import cluster_at_thresholds, assign_categories
from prot_T5 import calculate_embeddings, combine_embedding_files

# --------------------------------------------------------------------------- #
# Download calibration FASTA files                                            #
# --------------------------------------------------------------------------- #
_REPO_ZIP = "https://github.com/jakob949/pan_genome/archive/refs/heads/main.zip"
_CALIB_DIR = "fastas_for_calibration" # Default subdirectory for downloaded calibration files

def _download_calibration_fastas(dest_dir: str = _CALIB_DIR) -> list[str]:
    os.makedirs(dest_dir, exist_ok=True)
    existing = glob.glob(os.path.join(dest_dir, "*.fna"))
    if existing:
        print(f"Found existing calibration FASTA files in {dest_dir}.")
        return sorted(map(os.path.abspath, existing))

    print(f"Downloading calibration FASTA files to {dest_dir}…")
    try:
        with urllib.request.urlopen(_REPO_ZIP) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                prefix = "pan_genome-main/fastas_for_calibration/"
                downloaded_any = False
                for member in zf.namelist():
                    if member.startswith(prefix) and member.endswith(".fna"):
                        fname = os.path.basename(member)
                        if not fname: # Skip directory entries if any
                            continue
                        out_path = os.path.join(dest_dir, fname)
                        with zf.open(member) as src, open(out_path, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        downloaded_any = True
                if not downloaded_any:
                    print(f"Warning: No files matching '*.fna' found within the '{prefix}' path in the repository ZIP.")

    except urllib.error.URLError as e:
        print(f"Error downloading calibration FASTA files: {e}")
        # If download fails, return any existing files or an empty list
        return sorted(glob.glob(os.path.join(dest_dir, "*.fna")))
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        return sorted(glob.glob(os.path.join(dest_dir, "*.fna")))

    final_fasta_files = sorted(glob.glob(os.path.join(dest_dir, "*.fna")))
    if not final_fasta_files:
        print(f"Warning: Download process completed, but no '*.fna' files were ultimately found in {dest_dir}.")
    return final_fasta_files

def ensure_calibration_embeddings(
    dest_dir: str, # Changed default, will be set by main based on args
    acceleration: str = "onnx",
    max_seq_length: int = 7500,
    max_batch_tokens: int = 8500,
    max_batch_size: int = 512,
    model_name: str = "Rostlab/prot_t5_xl_uniref50",
    user_fasta_input_pattern: str = None,
) -> list[str]:
    os.makedirs(dest_dir, exist_ok=True) # Ensure dest_dir (output for embeddings) exists

    input_pattern_for_embeddings: str
    output_prefix_for_embeddings: str # This will be like os.path.join(dest_dir, "prefix")

    if user_fasta_input_pattern:
        print(f"Using user-provided FASTA files matching pattern: {user_fasta_input_pattern}")
        # Check if the pattern itself yields any files before proceeding
        # This check is now primarily done inside ensure_embeddings
        input_pattern_for_embeddings = user_fasta_input_pattern
        output_prefix_for_embeddings = os.path.join(dest_dir, "user")
    else:
        print(f"Using default calibration FASTA files. Target directory for FASTAs: {dest_dir}")
        # Note: If dest_dir is e.g. /my/outputs, calibration FASTAs will be downloaded to /my/outputs/
        # (or a subfolder if _CALIB_DIR was more complexly joined with dest_dir, but here dest_dir IS the target)
        # For clarity, one might use a subfolder for calibration fastas within dest_dir
        # e.g., calib_fasta_subdir = os.path.join(dest_dir, "calibration_fastas")
        # and pass calib_fasta_subdir to _download_calibration_fastas
        # For now, dest_dir is used directly as per previous logic with _CALIB_DIR.
        
        calib_fasta_storage_dir = os.path.join(dest_dir, "calibration_source_fastas") # Store downloaded fastas here
        downloaded_fasta_files = _download_calibration_fastas(calib_fasta_storage_dir)
        
        if not downloaded_fasta_files:
            raise FileNotFoundError(
                f"Failed to download or locate calibration FASTA files in {calib_fasta_storage_dir}. "
                f"Please check network connectivity, the repository at {_REPO_ZIP}, or provide files via --user_fasta_dir."
            )
        input_pattern_for_embeddings = os.path.join(calib_fasta_storage_dir, "*.fna")
        output_prefix_for_embeddings = os.path.join(dest_dir, "calib")

    print(f"Embeddings will be generated with prefix: {output_prefix_for_embeddings}_<basename>.pt")
    pt_files = ensure_embeddings(
        input_pattern=input_pattern_for_embeddings,
        output_prefix=output_prefix_for_embeddings, # e.g., "/path/to/dest_dir/user" or "/path/to/dest_dir/calib"
        max_seq_length=max_seq_length,
        max_batch_tokens=max_batch_tokens,
        max_batch_size=max_batch_size,
        acceleration=acceleration,
        model_name=model_name,
    )
    return pt_files

def ensure_embeddings(
    input_pattern: str,
    output_prefix: str, # e.g., "/path/to/dest_dir/user" or "/path/to/dest_dir/calib"
    max_seq_length: int = 7500,
    max_batch_tokens: int = 8500,
    max_batch_size: int = 512,
    acceleration: str = "onnx",
    model_name: str = "Rostlab/prot_t5_xl_uniref50",
) -> list[str]:
    fasta_files = sorted(glob.glob(input_pattern))
    if not fasta_files:
        raise FileNotFoundError(f"No FASTA files found matching input pattern: {input_pattern}")
    
    print(f"Found {len(fasta_files)} FASTA files to process from pattern: {input_pattern}")

    pt_files = []
    for fasta in fasta_files:
        base = os.path.splitext(os.path.basename(fasta))[0]
        # output_prefix is now a full path prefix like dest_dir/user or dest_dir/calib
        out_pt = f"{output_prefix}_{base}.pt" # Results in dest_dir/user_base.pt
        out_pt_abs = os.path.abspath(out_pt)

        if not os.path.isfile(out_pt_abs):
            print(f"Generating embeddings for {fasta} -> {out_pt_abs}")
            calculate_embeddings(
                fasta_file=fasta,
                output_file=out_pt_abs,
                max_seq_length=max_seq_length,
                max_batch_tokens=max_batch_tokens,
                max_batch_size=max_batch_size,
                acceleration=acceleration,
                model_name=model_name,
            )
            combine_embedding_files(out_pt_abs) # Assuming this is still needed for individual .pt files
        else:
            print(f"Embeddings file {out_pt_abs} already exists. Skipping generation.")
        pt_files.append(out_pt_abs)
    return pt_files

def concatenate_embeddings_cpu(embedding_files: list[str]) -> tuple[np.ndarray, list[str]]:
    if not embedding_files:
        raise ValueError("No embedding files provided")
    all_emb, all_ids = [], []
    expected_dim = None

    for f in embedding_files:
        data = torch.load(f, map_location="cpu")
        for pid, emb in data.items():
            arr = emb.cpu().numpy()
            # determine vector length
            vec_len = arr.shape[0] if arr.ndim == 1 else arr.shape[-1]
            if expected_dim is None:
                expected_dim = vec_len
            elif vec_len != expected_dim:
                print(f"Warning: skipping {pid} in {f} (dim {vec_len} != expected {expected_dim})")
                continue
            # ensure each is a row vector for stacking
            row = arr.reshape(1, -1) if arr.ndim == 1 else arr
            all_emb.append(row)
            all_ids.append(pid)

    if not all_emb:
        raise ValueError(f"No valid embeddings to concatenate in: {embedding_files}")

    emb_matrix = np.vstack(all_emb)
    return emb_matrix, all_ids

# Assuming these functions are available from your original script:
# from cluster_faiss_fuzz_tools import cluster_at_thresholds
# from your_module import concatenate_embeddings_cpu # if not in the same file

def find_valid_st_range_coverage_criteria(
    pt_files: list[str],
    alpha_dominant_cluster: float = 0.20,
    target_P_high_coverage: float = 0.90,
    target_P_medium_coverage: float = 0.60,
    # Assuming cluster_at_thresholds and concatenate_embeddings_cpu are accessible
    # Or pass them as arguments if they are in different modules
) -> tuple[dict[str, tuple[int,int]], dict[str, list[float]], int, int]:
    """
    Finds a valid similarity threshold range based on the proportion of embeddings
    falling into dominant clusters.

    Args:
        pt_files: List of paths to embedding (.pt) files.
        alpha_dominant_cluster: Fraction of total embeddings for a cluster to be "dominant".
        target_P_high_coverage: Target proportion of embeddings in dominant clusters 
                                for the lower similarity bound.
        target_P_medium_coverage: Target proportion of embeddings in dominant clusters
                                 for the higher similarity bound.

    Returns:
        tuple: (
            file2ST_indices: Dict mapping filename to (idx_coarse, idx_fine),
            file2P_dominant_embs: Dict mapping filename to list of P_dominant_embs values,
            overall_min_idx: Overall minimum index for the threshold range,
            overall_max_idx: Overall maximum index for the threshold range
        )
    """
    threshold_values = np.linspace(0.70, 1.005, 75)
    file2ST_indices = {}
    file2P_dominant_embs_log = {} # For logging/debugging

    all_idx_coarse_list = []
    all_idx_fine_list = []

    for pt_file in pt_files:
        # This is a placeholder for where concatenate_embeddings_cpu would be called.
        # emb, _ = concatenate_embeddings_cpu([pt_file]) 
        # This is a placeholder for where cluster_at_thresholds would be called.
        # cluster_results_all_s = cluster_at_thresholds(emb, threshold_values, cpu=False, k=100, batch_size=10000)
        
        # --- Mocking data for emb and cluster_results_all_s for demonstration ---
        # In a real scenario, these would come from your actual data and functions.
        # Ensure `concatenate_embeddings_cpu` and `cluster_at_thresholds` are correctly called.
        print(f"Processing {pt_file} for ST range finding...")
        try:
            emb_data = torch.load(pt_file, map_location="cpu") # Example of loading
            if not emb_data:
                print(f"Warning: No data in {pt_file}, skipping ST range calculation for this file.")
                file2P_dominant_embs_log[pt_file] = [0.0] * len(threshold_values)
                continue

            temp_embs = [e.cpu().numpy().reshape(1, -1) for e in emb_data.values() if e.numel() > 0 and e.ndim > 0] # ensure e is not scalar
            if not temp_embs:
                print(f"Warning: No valid embeddings to process in {pt_file} after loading, skipping ST range calculation.")
                file2P_dominant_embs_log[pt_file] = [0.0] * len(threshold_values)
                continue
            emb = np.vstack(temp_embs)

            # Mocking cluster_at_thresholds behavior for the example
            # To use actual clustering, replace this block with:
            # emb_for_clustering, _ = concatenate_embeddings_cpu([pt_file]) # Or use 'emb' if already suitable
            # cluster_results_all_s = cluster_at_thresholds(emb_for_clustering, threshold_values, cpu=True, k=100, batch_size=10000)
            # print(f"Mocking cluster results for {pt_file}") # Keep this print if mock is active
            cluster_results_all_s = {}
            n_emb_file_mock = emb.shape[0]
            for s_val_idx, s_val in enumerate(threshold_values):
                if n_emb_file_mock == 0:
                    cluster_results_all_s[s_val] = [] # s_val as key
                else:
                    # Simulate more clusters with higher similarity (lower s_val means more merging)
                    # This mock logic might need adjustment to reflect reality better:
                    # Higher s_val (similarity threshold) -> more, smaller clusters
                    # Lower s_val (similarity threshold) -> fewer, larger clusters
                    # P_dominant_embs should generally decrease as s_val increases (labels become more fragmented)
                    
                    # Mock clusters: Let's say at 0.7, 10% of N are clusters. At 1.0, N clusters.
                    # fraction_clusters = 0.1 + 0.9 * (s_val - 0.70) / (1.005 - 0.70)
                    # num_clusters_mock = max(1, int(n_emb_file_mock * fraction_clusters))

                    # Simpler mock: more clusters as s_val increases (index increases)
                    if s_val < 0.75: num_clusters_mock = max(1, n_emb_file_mock // 10)
                    elif s_val < 0.85: num_clusters_mock = max(1, n_emb_file_mock // 5)
                    elif s_val < 0.95: num_clusters_mock = max(1, n_emb_file_mock // 2)
                    else: num_clusters_mock = n_emb_file_mock

                mock_labels = np.random.randint(0, max(1, num_clusters_mock), n_emb_file_mock).tolist()
                cluster_results_all_s[s_val] = mock_labels # Use s_val as key as per original structure
        except Exception as e:
            print(f"Error processing file {pt_file} for ST range: {e}. Skipping this file.")
            file2P_dominant_embs_log[pt_file] = [0.0] * len(threshold_values)
            continue
        # --- End Mocking ---

        n_emb_file = emb.shape[0]
        if n_emb_file == 0: # Should have been caught by temp_embs check
            print(f"Warning: Zero embeddings in {pt_file} after processing, skipping index calculation.")
            file2P_dominant_embs_log[pt_file] = [0.0] * len(threshold_values)
            continue

        P_dominant_embs_values = []
        for s_idx, s_val in enumerate(threshold_values):
            # labels = cluster_results_all_s[s_idx] # Original was using s_idx as key - this is likely wrong
            labels = cluster_results_all_s[s_val] # Should use s_val as key if dict is {s_val: labels}
            if not labels: 
                P_dominant_embs_values.append(0.0)
                continue
            
            cluster_sizes = Counter(labels)
            sum_embs_in_dominant_clusters = 0
            for cluster_id, size in cluster_sizes.items():
                if size > alpha_dominant_cluster * n_emb_file:
                    sum_embs_in_dominant_clusters += size
            
            P_val = (sum_embs_in_dominant_clusters / n_emb_file) if n_emb_file > 0 else 0.0
            P_dominant_embs_values.append(P_val)
        
        file2P_dominant_embs_log[pt_file] = P_dominant_embs_values
        
        abs_diff_high_coverage = np.abs(np.array(P_dominant_embs_values) - target_P_high_coverage)
        idx_coarse_file = np.argmin(abs_diff_high_coverage)
        
        abs_diff_medium_coverage = np.abs(np.array(P_dominant_embs_values) - target_P_medium_coverage)
        idx_fine_file = np.argmin(abs_diff_medium_coverage)

        file2ST_indices[pt_file] = (idx_coarse_file, idx_fine_file)
        all_idx_coarse_list.append(idx_coarse_file)
        all_idx_fine_list.append(idx_fine_file)

    if not all_idx_coarse_list or not all_idx_fine_list: 
        print("Warning: No valid ST indices found from any processed files. Cannot determine overall range.")
        # Return empty/default values that won't cause crashes later
        return {}, file2P_dominant_embs_log, 0, len(threshold_values) - 1


    overall_min_idx = min(all_idx_coarse_list)
    overall_max_idx = max(all_idx_fine_list)
    
    if overall_min_idx > overall_max_idx:
        print(f"Warning: Overall min_idx ({overall_min_idx}) > overall_max_idx ({overall_max_idx}). "
              f"This might indicate issues with target P values, alpha_dominant, data characteristics, or mock data. "
              f"Swapping to form a valid range: [{overall_max_idx}, {overall_min_idx}]. Review parameters and data behavior.")
        overall_min_idx, overall_max_idx = overall_max_idx, overall_min_idx

    return file2ST_indices, file2P_dominant_embs_log, overall_min_idx, overall_max_idx

def main_new():
    parser = argparse.ArgumentParser(description="Determine similarity threshold range using dominant cluster coverage.")
    parser.add_argument("--dest_dir", default="/data/nilar/pan_genome/seqs_oth_threshold/esm/non", # Default from original
                        help="Directory for storing downloaded calibration FASTAs (if used) and all generated embeddings.")
    parser.add_argument("--user_fasta_dir", type=str, default=None,
                        help="Glob pattern to user-provided FASTA files (e.g., '/path/to/fastas/*.fna'). "
                             "If provided, calibration FASTAs will not be downloaded. "
                             "Embeddings will be stored in --dest_dir with a 'user_' prefix.")
    parser.add_argument("--acceleration", choices=["onnx","quantization",None],
                        default="onnx", help="Embedding acceleration mode")
    parser.add_argument("--max_seq_length", type=int, default=7500)
    parser.add_argument("--max_batch_tokens", type=int, default=8500)
    parser.add_argument("--max_batch_size", type=int, default=32) # Adjusted from 512 to 32 as per example
    parser.add_argument("--model_name", default="Rostlab/prot_t5_xl_uniref50",
                        help="HuggingFace model ID for embeddings")
    
    parser.add_argument("--alpha_dominant", type=float, default=0.15,
                        help="Fraction of total embeddings for a cluster to be 'dominant'")
    parser.add_argument("--target_P_high_coverage", type=float, default=0.80,
                        help="Target P(embeddings in dominant clusters) for the coarser clustering bound")
    parser.add_argument("--target_P_medium_coverage", type=float, default=0.50,
                        help="Target P(embeddings in dominant clusters) for the finer clustering bound")
    args = parser.parse_args()

    print("Step 1: Preparing/Generating embedding files...")
    pt_files = []
    try:
        pt_files = ensure_calibration_embeddings(
            dest_dir=args.dest_dir,
            acceleration=args.acceleration,
            max_seq_length=args.max_seq_length,
            max_batch_tokens=args.max_batch_tokens,
            max_batch_size=args.max_batch_size,
            model_name=args.model_name,
            user_fasta_input_pattern=args.user_fasta_dir 
        )
    except FileNotFoundError as e:
        print(f"Error: Essential FASTA files not found or could not be prepared: {e}")
        print("Please check your --user_fasta_dir pattern or network connection for calibration file download.")
        return 
    except Exception as e:
        print(f"An unexpected error occurred during embedding preparation/generation: {e}")
        import traceback
        traceback.print_exc()
        return
            
    if not pt_files:
        print("No embedding (.pt) files were found or generated. Cannot proceed. Exiting.")
        return
    
    print(f"\nStep 2: Computing valid similarity threshold ranges using {len(pt_files)} embedding file(s).")
    print("Using parameters:")
    print(f"  Alpha for dominant cluster: {args.alpha_dominant}")
    print(f"  Target P for high coverage (coarse clustering): {args.target_P_high_coverage}")
    print(f"  Target P for medium coverage (fine clustering): {args.target_P_medium_coverage}")

    file2ST_indices, file2P_log, overall_min_idx, overall_max_idx = find_valid_st_range_coverage_criteria(
        pt_files,
        args.alpha_dominant,
        args.target_P_high_coverage,
        args.target_P_medium_coverage
    )
    
    threshold_values = np.linspace(0.70, 1.005, 75)

    print("\n--- Results ---")
    if not file2ST_indices:
        print("No per-file ST indices could be determined.")
    else:
        print("\nPer-file threshold indices (coarse_idx, fine_idx) and corresponding similarity values:")
        mean_threshold_coarse_vals = []
        mean_threshold_fine_vals = []
        for pt, (idx_coarse, idx_fine) in file2ST_indices.items():
            sim_coarse = threshold_values[idx_coarse] if 0 <= idx_coarse < len(threshold_values) else float('nan')
            sim_fine = threshold_values[idx_fine] if 0 <= idx_fine < len(threshold_values) else float('nan')
            print(f"  {os.path.basename(pt)}: coarse_idx={idx_coarse} (sim={sim_coarse:.4f}), "
                  f"fine_idx={idx_fine} (sim={sim_fine:.4f})")
            if not np.isnan(sim_coarse): mean_threshold_coarse_vals.append(sim_coarse)
            if not np.isnan(sim_fine): mean_threshold_fine_vals.append(sim_fine)
        
        if mean_threshold_coarse_vals:
            mean_coarse_sim = np.mean(mean_threshold_coarse_vals)
            print(f"  Mean of per-file coarse similarity thresholds: {mean_coarse_sim:.4f}")
        if mean_threshold_fine_vals:
            mean_fine_sim = np.mean(mean_threshold_fine_vals)
            print(f"  Mean of per-file fine similarity thresholds: {mean_fine_sim:.4f}")

    print(f"\nOverall recommended similarity threshold range (based on min/max of per-file indices):")
    lower_sim_overall = threshold_values[overall_min_idx] if 0 <= overall_min_idx < len(threshold_values) else float('nan')
    higher_sim_overall = threshold_values[overall_max_idx] if 0 <= overall_max_idx < len(threshold_values) else float('nan')
    
    print(f"  Lower similarity threshold bound: {lower_sim_overall:.4f} (index {overall_min_idx})")
    print(f"  Higher similarity threshold bound: {higher_sim_overall:.4f} (index {overall_max_idx})")


    if file2P_log:
        first_file_key = next(iter(file2P_log))
        # print(f"\nLog of P_dominant_embs for {os.path.basename(first_file_key)} across thresholds:")
        # for s_val, p_val in zip(threshold_values, file2P_log[first_file_key]):
        #     print(f"  Sim: {s_val:.4f}, P_dominant: {p_val:.4f}")
    print("\nDone.")

if __name__ == "__main__":
    main_new()