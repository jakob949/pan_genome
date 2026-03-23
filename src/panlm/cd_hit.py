import os
import subprocess

import pandas as pd


def run_cd_hit(input_fasta, output_fasta, identity=0.92):
    """
    Run CD-HIT on the input FASTA file and save both representative sequences
    and information about filtered sequences.

    Args:
        input_fasta (str): Path to the input FASTA file containing protein sequences.
        output_fasta (str): Path to save the output FASTA file with representative sequences.
        identity (float): Sequence identity threshold for clustering (default: 0.9).

    Returns:
        tuple: (bool, dict) - Success status and dictionary mapping filtered sequences to their representatives
    """
    try:
        # Run CD-HIT
        cmd = [
            "cd-hit",
            "-i",
            input_fasta,
            "-o",
            f"{output_fasta}.temp",
            "-c",
            str(identity),
            "-T",
            "0",
            "-M",
            "0",
            "-d",
            "0",
        ]
        # subprocess.run(cmd, check=True)
        # subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(cmd, check=True)

        # Dictionary to store mapping of filtered sequences to their representatives
        filtered_to_rep = {}

        # Parse cluster file to get filtered sequences
        with open(f"{output_fasta}.temp.clstr", "r") as cluster_file:
            current_rep = None
            for line in cluster_file:
                if not line.startswith(">"):
                    # Extract sequence ID and representative status
                    seq_info = line.strip().split()
                    seq_id = seq_info[2].replace(">", "").replace("...", "")
                    is_representative = "*" in line

                    if is_representative:
                        current_rep = seq_id
                    else:
                        filtered_to_rep[seq_id] = current_rep

        # Extract representative sequences
        with (
            open(f"{output_fasta}.temp", "r") as temp_file,
            open(output_fasta, "w") as out_file,
        ):
            write_sequence = False
            for line in temp_file:
                if line.startswith(">"):
                    write_sequence = True
                    out_file.write(line)
                elif write_sequence:
                    out_file.write(line)

        # Clean up temporary files
        os.remove(f"{output_fasta}.temp")
        os.remove(f"{output_fasta}.temp.clstr")

        return True, filtered_to_rep

    except subprocess.CalledProcessError:
        print(f"Error: CD-HIT failed to run on {input_fasta}")
        return False, {}
    except Exception as e:
        print(f"Error: {str(e)}")
        return False, {}


def add_cdhit_filtered_sequences_to_clusters(cluster_df, filtered_sequences_dict):
    """
    Add filtered sequences from CD-HIT to their corresponding clusters in the DataFrame.

    Args:
        cluster_df (pd.DataFrame): DataFrame containing clustering results
        filtered_sequences_dict (dict): Dictionary mapping filtered sequences to their representatives

    Returns:
        pd.DataFrame: Updated DataFrame including filtered sequences
    """

    # Create a dictionary from the cluster_df for O(1) lookups.
    # We only need the columns that will be copied to the filtered sequences.

    # Identify neighbor columns to set them to None later
    neighbor_cols = [col for col in cluster_df.columns if "neighbor" in col]

    # Select columns to keep (all except neighbor columns)
    cols_to_keep = [col for col in cluster_df.columns if col not in neighbor_cols]

    # Create the fast lookup map
    # This maps rep_seq -> {column_name: value}
    print("Creating representative sequence lookup map...")
    rep_map = cluster_df[cols_to_keep].set_index("protein_id").to_dict("index")
    print("Lookup map created.")

    # Create rows for filtered sequences
    new_rows = []

    print(f"Processing {len(filtered_sequences_dict)} filtered sequences...")
    for filtered_seq, rep_seq in filtered_sequences_dict.items():
        # Find the cluster data of the representative sequence using the map
        rep_data = rep_map.get(rep_seq)

        if rep_data:
            # Create a new row for the filtered sequence
            new_row = rep_data.copy()

            # Set neighbor columns to None (as in original code)
            for col in neighbor_cols:
                new_row[col] = None

            # Update the protein_id and flags
            new_row["protein_id"] = filtered_seq
            new_row["is_cd_hit_filtered"] = True
            new_row["cd_hit_representative"] = rep_seq
            new_rows.append(new_row)
    # --- OPTIMIZATION END ---

    print("Filtered sequence processing complete.")

    # Add flag for original sequences
    cluster_df["is_cd_hit_filtered"] = False
    cluster_df["cd_hit_representative"] = None

    # Append new rows if any exist
    if new_rows:
        print("Concatenating original and filtered DataFrames...")
        filtered_df = pd.DataFrame(new_rows)
        # Re-order columns in filtered_df to match cluster_df for clean concatenation
        filtered_df = filtered_df[cluster_df.columns]

        updated_df = pd.concat([cluster_df, filtered_df], ignore_index=True)
        print("Concatenation complete.")
        return updated_df

    return cluster_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CD-HIT on a FASTA file")
    parser.add_argument("input_fasta", help="Path to the input FASTA file")
    parser.add_argument("output_fasta", help="Path to save the output FASTA file")
    parser.add_argument(
        "--identity",
        type=float,
        default=0.98,
        help="Sequence identity threshold (default: 0.98)",
    )
    args = parser.parse_args()

    success, clusters = run_cd_hit(args.input_fasta, args.output_fasta, args.identity)
