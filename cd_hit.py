import subprocess
import os

def run_cd_hit(input_fasta, output_fasta, identity=0.98):
    """
    Run CD-HIT on the input FASTA file and save the first sequence in each cluster.
    
    Args:
    input_fasta (str): Path to the input FASTA file containing protein sequences.
    output_fasta (str): Path to save the output FASTA file with representative sequences.
    identity (float): Sequence identity threshold for clustering (default: 0.98).
    
    Returns:
    bool: True if CD-HIT ran successfully, False otherwise.
    """
    try:
        # Run CD-HIT
        cmd = [
            "cd-hit",
            "-i", input_fasta,
            "-o", f"{output_fasta}.temp",
            "-c", str(identity),
            "-T", "0",  # Use all available CPU cores
            "-M", "0",  # Use all available memory
        ]
        subprocess.run(cmd, check=True)

        # Extract the first sequence from each cluster
        with open(f"{output_fasta}.temp", "r") as temp_file, open(output_fasta, "w") as out_file:
            write_sequence = False
            for line in temp_file:
                if line.startswith(">"):
                    if not write_sequence:
                        write_sequence = True
                        out_file.write(line)
                elif write_sequence:
                    out_file.write(line)
                    write_sequence = False

        # Clean up temporary files
        os.remove(f"{output_fasta}.temp")
        os.remove(f"{output_fasta}.temp.clstr")

        return True
    except subprocess.CalledProcessError:
        print(f"Error: CD-HIT failed to run on {input_fasta}")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run CD-HIT on a FASTA file")
    parser.add_argument("input_fasta", help="Path to the input FASTA file")
    parser.add_argument("output_fasta", help="Path to save the output FASTA file")
    parser.add_argument("--identity", type=float, default=0.98, help="Sequence identity threshold (default: 0.98)")
    args = parser.parse_args()

    success = run_cd_hit(args.input_fasta, args.output_fasta, args.identity)
    if success:
        print(f"CD-HIT completed successfully. Output saved to {args.output_fasta}")
    else:
        print("CD-HIT failed to complete.")