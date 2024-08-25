import os
import torch
import re
import json
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm
from Bio import SeqIO
from collections import OrderedDict
import argparse

def parse_fasta(fasta_file):
    proteins = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        protein_id = record.id
        translation = str(record.seq)
        proteins.append((translation, protein_id, os.path.basename(fasta_file)))
    return proteins

def preprocess_sequences(sequences, max_length=None):
    processed = []
    for seq in sequences:
        seq = re.sub(r"[UZOB]", "X", seq)
        if max_length and len(seq) > max_length:
            seq = seq[:max_length]
        processed.append(" ".join(list(seq)))
    return processed

def initialize_model(use_bf16, model_name="Rostlab/prot_t5_xl_half_uniref50-enc"):
    print("Loading ProtT5 model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  

    model.eval()
    print(f"Model parameter dtype: {next(model.parameters()).dtype}")
    return model, tokenizer, device

def calculate_embeddings(fasta_file, output_file, max_seq_length, target_batch_size, use_bf16):
    model, tokenizer, device = initialize_model(use_bf16)
    
    proteins = parse_fasta(fasta_file)
    print(f"Found {len(proteins)} proteins")
    
    proteins.sort(key=lambda x: len(x[0]), reverse=True)
    
    total_proteins = 0
    batch = []
    batch_seq_length = 0
    header_info = OrderedDict()
    
    with torch.no_grad():
        for protein in tqdm(proteins, desc="Calculating embeddings"):
            sequence, protein_id, source_file = protein
            seq_length = len(sequence)
            
            if seq_length > max_seq_length:
                process_batch([(sequence, protein_id, source_file)], model, tokenizer, device, output_file, total_proteins, max_length=max_seq_length)
                header_info[protein_id] = {"length": min(seq_length, max_seq_length), "source": source_file}
                total_proteins += 1
            else:
                if batch and (batch_seq_length + seq_length) ** 2 > (target_batch_size * max_seq_length ** 2):
                    process_batch(batch, model, tokenizer, device, output_file, total_proteins)
                    for seq, pid, src in batch:
                        header_info[pid] = {"length": len(seq), "source": src}
                    total_proteins += len(batch)
                    batch = []
                    batch_seq_length = 0
                
                batch.append(protein)
                batch_seq_length += seq_length
        
        if batch:
            process_batch(batch, model, tokenizer, device, output_file, total_proteins)
            for seq, pid, src in batch:
                header_info[pid] = {"length": len(seq), "source": src}
            total_proteins += len(batch)
    
    header_file = f"{output_file}.header.json"
    with open(header_file, 'w') as f:
        json.dump(header_info, f, indent=2)
    
    return total_proteins

def process_batch(batch, model, tokenizer, device, output_file, batch_number, max_length=None):
    batch_sequences, batch_ids, batch_sources = zip(*batch)
    
    preprocessed_sequences = preprocess_sequences(batch_sequences, max_length)
    ids = tokenizer.batch_encode_plus(preprocessed_sequences, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    
    embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
    
    batch_embeddings = OrderedDict()
    for j, (seq, pid, src) in enumerate(zip(batch_sequences, batch_ids, batch_sources)):
        seq_length = min(len(seq), max_length) if max_length else len(seq)
        emb = embedding_repr.last_hidden_state[j, :seq_length]
        per_protein_emb = emb.mean(dim=0)
        batch_embeddings[pid] = per_protein_emb.cpu()
    
    torch.save(batch_embeddings, f"{output_file}.part{batch_number:04d}")
    
    del embedding_repr, input_ids, attention_mask, batch_embeddings
    torch.cuda.empty_cache()

def combine_embedding_files(output_file):
    path = os.path.dirname(output_file)
    base_name = os.path.basename(output_file)
    
    part_files = sorted([f for f in os.listdir(path) if f.startswith(base_name) and re.search(r'\.part\d+$', f)])
    
    all_embeddings = OrderedDict()
    total_proteins = 0
    embedding_dim = None
    
    for part_file in tqdm(part_files, desc="Combining embedding files"):
        full_path = os.path.join(path, part_file)
        try:
            embeddings = torch.load(full_path)
            for pid, emb in embeddings.items():
                all_embeddings[pid] = emb
                total_proteins += 1
                if embedding_dim is None:
                    embedding_dim = emb.shape[0]
            os.remove(full_path)
        except Exception as e:
            print(f"Error processing {full_path}: {str(e)}")
    
    print(f"Total number of proteins: {total_proteins}")
    print(f"Embedding dimension: {embedding_dim}")
    
    torch.save(all_embeddings, output_file)
    print(f"Saved combined embeddings for {total_proteins} proteins to {output_file}")

    header_file = f"{output_file}.header.json"
    with open(header_file, 'r') as f:
        header_info = json.load(f, object_pairs_hook=OrderedDict)
    
    if list(all_embeddings.keys()) == list(header_info.keys()):
        print("Embeddings and header information are perfectly aligned.")
    else:
        print("Warning: Embeddings and header information are not aligned.")

def main(fasta_input, output_directory, max_seq_length, target_batch_size, use_bf16):
    if not os.path.isfile(fasta_input):
        raise ValueError(f"Input file not found: {fasta_input}")
    
    print(f"Processing FASTA file: {fasta_input}")
    input_name = os.path.splitext(os.path.basename(fasta_input))[0]
    
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, f"{input_name}_protein_embeddings_prot_t5.pt")
    
    print("Calculating ProtT5 embeddings...")
    total_proteins = calculate_embeddings(fasta_input, output_file, max_seq_length, target_batch_size, use_bf16)
    
    print("Combining embedding files...")
    combine_embedding_files(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate ProtT5 embeddings for proteins in FASTA files")
    parser.add_argument("fasta_input", help="Input FASTA file containing protein sequences")
    parser.add_argument("output_directory", help="Directory to save the output embeddings")
    parser.add_argument("--max_seq_length", type=int, default=12500, help="Maximum sequence length to process")
    parser.add_argument("--target_batch_size", type=int, default=2, help="Target batch size for full-length sequences")
    parser.add_argument("--use_bf16", action="store_true", help="Use bfloat16 precision if available")
    args = parser.parse_args()
    
    main(args.fasta_input, args.output_directory, args.max_seq_length, args.target_batch_size, args.use_bf16)