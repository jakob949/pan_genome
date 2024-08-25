import os
import torch
import re
import json
from transformers import T5EncoderModel, T5Tokenizer
from concurrent.futures import ProcessPoolExecutor
import argparse
from tqdm import tqdm
from Bio import SeqIO
from collections import OrderedDict

def parse_single_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.gff3':
        return parse_gff3(file_path)
    elif file_extension in ['.gb', '.gbk', '.genbank']:
        return parse_genbank(file_path)
    elif file_extension in ['.fst', '.fa', '.fasta']:
        return parse_fasta(file_path)
    else:
        print(f"Unsupported file format: {file_path}")
        return []

def parse_gff3(gff3_file):
    proteins = []
    with open(gff3_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            fields = line.strip().split('\t')
            if len(fields) != 9 or fields[2] != 'CDS':
                continue
            
            attributes = dict(item.split('=') for item in fields[8].split(';') if '=' in item)
            if 'translation' in attributes:
                protein_id = attributes.get('protein_id', attributes.get('locus_tag', 'unknown'))
                translation = attributes['translation']
                
                proteins.append((translation, protein_id, os.path.basename(gff3_file)))
    
    return proteins

def parse_fasta(fasta_file):
    proteins = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        protein_id = record.id
        translation = str(record.seq)
        proteins.append((translation, protein_id, os.path.basename(fasta_file)))
    
    return proteins


def parse_genbank(genbank_file):
    proteins = []
    for record in SeqIO.parse(genbank_file, "genbank"):
        for feature in record.features:
            if feature.type == "CDS":
                if "translation" in feature.qualifiers:
                    translation = feature.qualifiers["translation"][0]
                    protein_id = feature.qualifiers.get("protein_id", [feature.qualifiers.get("locus_tag", ["unknown"])[0]])[0]
                    proteins.append((translation, protein_id, os.path.basename(genbank_file)))
    
    return proteins

def parse_multiple_files(directory):
    all_proteins = []
    genome_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.gff3', '.gb', '.gbk', '.genbank'))]
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(parse_single_file, genome_files), total=len(genome_files), desc="Parsing genome files"))
    
    for result in results:
        all_proteins.extend(result)
    
    return all_proteins