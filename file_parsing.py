# file_parsing.py
import os
import itertools  # Import itertools
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from BCBio import GFF

# --- Start of optimizations ---
try:
    import pyfastx
    PYFASTX_AVAILABLE = True
except ImportError:
    PYFASTX_AVAILABLE = False
# --- End of optimizations ---


def parse_single_file(file_path):
    file_extension = os.path.splitext(file_path)[-1].lower()
    if file_extension in ['.gff3', '.gff']:
        return parse_gff3(file_path)
    elif file_extension in ['.gb', '.gbk', '.genbank', '.gbff']:
        return parse_genbank(file_path)
    elif file_extension in ['.fna', '.faa', '.fasta', '.fa']:
        return parse_fasta(file_path)
    else:
        print(f"Unsupported file format: {file_path}")
        return []

def parse_gff3(gff3_file):
    proteins = []
    file_name = os.path.basename(gff3_file)
    try:
        with open(gff3_file) as handle:
            for rec in GFF.parse(handle):
                for feature in rec.features:
                    if feature.type == "CDS":
                        if 'translation' in feature.qualifiers:
                            translation = feature.qualifiers['translation'][0]
                        else:
                            # extract and translate from sequence
                            try:
                                dna_seq = feature.extract(rec.seq)
                                translation = str(dna_seq.translate(to_stop=True))
                            except Exception:
                                # no sequence available; skip
                                continue
                        protein_id = (
                            feature.qualifiers.get('protein_id', [None])[0]
                            or feature.qualifiers.get('locus_tag', ['unknown'])[0]
                        )
                        proteins.append((translation, protein_id, file_name))
    except Exception:
        # Fallback for GFF3 without sequence or FASTA section: manual parse
        with open(gff3_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                cols = line.strip().split('\t')
                if len(cols) < 9 or cols[2] != 'CDS':
                    continue
                attr_str = cols[8]
                attrs = {}
                for kv in attr_str.split(';'):
                    if '=' in kv:
                        key, val = kv.split('=', 1)
                        attrs[key] = val
                translation = attrs.get('translation')
                if not translation:
                    # cannot translate without sequence
                    continue
                protein_id = attrs.get('protein_id') or attrs.get('locus_tag') or 'unknown'
                proteins.append((translation, protein_id, file_name))
    return proteins

def parse_fasta(fasta_file):
    proteins = []
    file_name = os.path.basename(fasta_file)
    
    # --- Start of optimized FASTA parsing ---
    if PYFASTX_AVAILABLE:
        try:
            # pyfastx is significantly faster for large FASTA files
            fa = pyfastx.Fasta(fasta_file)
            for name, seq in fa:
                proteins.append((seq, name, file_name))
        except Exception as e:
            # Fallback to SeqIO if pyfastx fails (e.g., malformed file)
            print(f"pyfastx failed on {fasta_file} ({e}), falling back to Bio.SeqIO.")
            try:
                for record in SeqIO.parse(fasta_file, "fasta"):
                    proteins.append((str(record.seq), record.id, file_name))
            except Exception as e_bio:
                print(f"Error parsing FASTA file {fasta_file} with Bio.SeqIO: {str(e_bio)}")
    else:
        # Original logic if pyfastx is not installed
        try:
            for record in SeqIO.parse(fasta_file, "fasta"):
                proteins.append((str(record.seq), record.id, file_name))
        except Exception as e:
            print(f"Error parsing FASTA file {fasta_file}: {str(e)}")
    # --- End of optimized FASTA parsing ---
            
    return proteins

def parse_genbank(genbank_file):
    proteins = []
    file_name = os.path.basename(genbank_file)
    try:
        for record in SeqIO.parse(genbank_file, "genbank"):
            for feature in record.features:
                if feature.type == "CDS" and "translation" in feature.qualifiers:
                    translation = feature.qualifiers["translation"][0]
                    protein_id = (
                        feature.qualifiers.get("protein_id", [None])[0]
                        or feature.qualifiers.get("locus_tag", ["unknown"])[0]
                    )
                    proteins.append((translation, protein_id, file_name))
    except Exception as e:
        print(f"Error parsing GenBank file {genbank_file}: {str(e)}")
    return proteins

def parse_multiple_files_parallel(genome_files: list):
    """
    Parses a list of genome files in parallel using a ProcessPoolExecutor.
    
    Args:
        genome_files (list): A list of file paths to parse.

    Returns:
        list: A list of all extracted (protein_seq, protein_id, source_file) tuples.
    """
    # Note: all_proteins = [] is no longer needed here
    with ProcessPoolExecutor() as executor:
        # The executor.map call is the part that runs in parallel
        results = list(tqdm(
            executor.map(parse_single_file, genome_files),
            total=len(genome_files),
            desc="Parsing genome files"
        ))

    all_proteins = list(itertools.chain.from_iterable(results))
    
    return all_proteins

def file_exists_check(file_path):
    return os.path.isfile(file_path) and os.path.getsize(file_path) > 0

def write_proteins_to_fasta(proteins, output_file):
    with open(output_file, 'w') as f:
        for seq, protein_id, source_file in proteins:
            f.write(f">{protein_id}|{source_file}\n{seq}\n")