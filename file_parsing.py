import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from BCBio import GFF

def parse_single_file(file_path):
    file_extension = os.path.splitext(file_path)[-1].lower()
    if file_extension in ['.gff3', '.gff']:
        return parse_gff3(file_path)
    elif file_extension in ['.gb', '.gbk', '.genbank']:
        return parse_genbank(file_path)
    elif file_extension in ['.fna', '.faa', '.fasta', '.fa']:
        return parse_fasta(file_path)
    else:
        print(f"Unsupported file format: {file_path}")
        return []

def parse_gff3(gff3_file):
    proteins = []
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
                        proteins.append((translation, protein_id, os.path.basename(gff3_file)))
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
                proteins.append((translation, protein_id, os.path.basename(gff3_file)))
    return proteins

def parse_fasta(fasta_file):
    proteins = []
    try:
        for record in SeqIO.parse(fasta_file, "fasta"):
            proteins.append((str(record.seq), record.id, os.path.basename(fasta_file)))
    except Exception as e:
        print(f"Error parsing FASTA file {fasta_file}: {str(e)}")
    return proteins

def parse_genbank(genbank_file):
    proteins = []
    try:
        for record in SeqIO.parse(genbank_file, "genbank"):
            for feature in record.features:
                if feature.type == "CDS" and "translation" in feature.qualifiers:
                    translation = feature.qualifiers["translation"][0]
                    protein_id = (
                        feature.qualifiers.get("protein_id", [None])[0]
                        or feature.qualifiers.get("locus_tag", ["unknown"])[0]
                    )
                    proteins.append((translation, protein_id, os.path.basename(genbank_file)))
    except Exception as e:
        print(f"Error parsing GenBank file {genbank_file}: {str(e)}")
    return proteins

def parse_multiple_files(directory):
    genome_files = [
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.endswith(('.gff3', '.gff', '.gb', '.gbk', '.genbank', '.fna', '.faa', '.fasta', '.fa'))
    ]
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(parse_single_file, genome_files),
            total=len(genome_files),
            desc="Parsing genome files"
        ))
    all_proteins = []
    for result in results:
        all_proteins.extend(result)
    return all_proteins

def file_exists_check(file_path):
    return os.path.isfile(file_path) and os.path.getsize(file_path) > 0

def write_proteins_to_fasta(proteins, output_file):
    with open(output_file, 'w') as f:
        for seq, protein_id, source_file in proteins:
            f.write(f">{protein_id}|{source_file}\n{seq}\n")

