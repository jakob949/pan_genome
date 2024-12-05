import os
import torch
import re
import json
from transformers import T5EncoderModel, T5Tokenizer, BitsAndBytesConfig
from tqdm import tqdm
from Bio import SeqIO
from collections import OrderedDict
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from DBSampler import DynamicBatchSampler
import time


def concatenate_embeddings(embedding_files):
    all_embeddings = []
    all_protein_ids = []
    for file in embedding_files:
        print(f"Reading embeddings from: {file}")
        embeddings = torch.load(file)
        for protein_id, embedding in embeddings.items():
            all_embeddings.append(embedding.numpy())
            all_protein_ids.append(protein_id)
    return np.array(all_embeddings), all_protein_ids

def parse_fasta(fasta_file):
    proteins = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        protein_id = record.id
        translation = str(record.seq)
        proteins.append((translation, protein_id, os.path.basename(fasta_file)))
    return proteins

def preprocess_sequence(seq):
    seq = re.sub(r"[UZOB]", "X", seq)
    return " ".join(list(seq))

class ProteinDataset(Dataset):
    def __init__(self, proteins, tokenizer, max_length=None):
        self.proteins = proteins
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.len_dict = {}  # For DynamicBatchSampler
        
        # Pre-compute lengths and tokenization
        for idx, (sequence, _, _) in enumerate(proteins):
            if max_length:
                sequence = sequence[:max_length]
            processed_seq = preprocess_sequence(sequence)
            tokens = self.tokenizer.encode(processed_seq, add_special_tokens=True)
            self.len_dict[idx] = len(tokens)

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        sequence, protein_id, source_file = self.proteins[idx]
        if self.max_length:
            sequence = sequence[:self.max_length]
        
        processed_seq = preprocess_sequence(sequence)
        tokens = self.tokenizer.encode(processed_seq, add_special_tokens=True)
        return tokens, protein_id, source_file, len(tokens)

def padding_collator(batch):
    if not batch:
        raise ValueError("Empty batch received!")
        
    # Unzip the batch
    tokens, protein_ids, source_files, lengths = zip(*batch)
    
    # Find max length in this batch
    max_len = max(lengths)
    
    # Debug information
    print(f"Batch size: {len(batch)}, Max length in batch: {max_len}")
    
    # Pad sequences
    padded_tokens = []
    attention_masks = []
    
    for seq in tokens:
        padding_len = max_len - len(seq)
        padded_seq = seq + [1] * padding_len  # 1 is the pad token ID for T5
        attention_mask = [1] * len(seq) + [0] * padding_len
        
        padded_tokens.append(padded_seq)
        attention_masks.append(attention_mask)
    
    # Convert to tensors
    padded_tokens = torch.tensor(padded_tokens)
    attention_masks = torch.tensor(attention_masks)
    
    return padded_tokens, attention_masks, protein_ids, source_files

def initialize_model(use_quantization, model_name="Rostlab/prot_t5_xl_uniref50"):
    """Initialize ProtT5 model with optional quantization"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            # Use 8-bit precision 
            load_in_8bit=True,
            # Use 4-bit precision 
            # load_in_4bit=True,


            torch_dtype=torch.float16,
        )
        
        model = T5EncoderModel.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )
        print("Quantization model\n", model)
    else:
        model = T5EncoderModel.from_pretrained(model_name)
        model.to(device)
    
    model.eval()
    return model, tokenizer, device

def calculate_embeddings(fasta_file, output_file, max_seq_length, max_batch_tokens, max_batch_size, use_quantization):
    model, tokenizer, device = initialize_model(use_quantization)
    
    proteins = parse_fasta(fasta_file)
    print(f"Found {len(proteins)} proteins")
    
    # Sort proteins by length for more efficient batching
    proteins.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Create dataset with tokenizer for proper length calculations
    dataset = ProteinDataset(proteins, tokenizer, max_length=max_seq_length)
    
    # Print some statistics about the sequence lengths
    lengths = list(dataset.len_dict.values())
    print(f"\nSequence length statistics:")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Mean length: {sum(lengths)/len(lengths):.2f}")
    print(f"Number of sequences exceeding max_seq_length: {sum(1 for l in lengths if l > max_seq_length)}\n")
    
    # Initialize DynamicBatchSampler
    # For single GPU processing, use num_replicas=1 and rank=0
    batch_sampler = DynamicBatchSampler(
        num_replicas=1,  # Single GPU processing
        rank=0,          # Single GPU processing
        length_dict=dataset.len_dict,
        num_buckets=128, # Number of buckets for length-based batching
        min_len=0,       # No minimum length constraint
        max_len=max_seq_length,  # Maximum sequence length
        max_batch_tokens=max_batch_tokens,  # Maximum tokens per batch
        max_batch_size=max_batch_size,      # Maximum sequences per batch
        shuffle=False,   # Keep False for sorted processing
        seed=0,         # Random seed
        drop_last=False # Don't drop the last batch
    )
    
    # Initialize the sampler for the first epoch
    batch_sampler.set_epoch(0)
    
    # Verify that batches are being created
    print(f"Number of batches: {len(batch_sampler)}")
    if len(batch_sampler) == 0:
        print("Warning: No batches created! Adjusting parameters...")
        # Try with more relaxed constraints
        batch_sampler = DynamicBatchSampler(
            num_replicas=1,
            rank=0,
            length_dict=dataset.len_dict,
            num_buckets=64,  # Reduced number of buckets
            min_len=0,
            max_len=max_seq_length,
            max_batch_tokens=max_batch_tokens * 2,  # Increased token limit
            max_batch_size=max_batch_size * 2,      # Increased batch size
            shuffle=False,
            seed=0,
            drop_last=False
        )
        batch_sampler.set_epoch(0)
        print(f"After adjustment - Number of batches: {len(batch_sampler)}")
        if len(batch_sampler) == 0:
            raise ValueError("Still no batches created after adjustment. Please check the sequence lengths and batch constraints.")
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=padding_collator,
        num_workers=4
    )
    
    total_proteins = 0
    header_info = OrderedDict()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating embeddings"):
            input_ids, attention_mask, protein_ids, source_files = batch
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
            
            batch_embeddings = OrderedDict()
            for j, (pid, src) in enumerate(zip(protein_ids, source_files)):
                # Get actual sequence length from attention mask
                seq_length = attention_mask[j].sum().item()
                emb = embedding_repr.last_hidden_state[j, :seq_length]
                per_protein_emb = emb.mean(dim=0)
                batch_embeddings[pid] = per_protein_emb.to(torch.float32).cpu()
                header_info[pid] = {"length": seq_length, "source": src}
            
            torch.save(batch_embeddings, f"{output_file}.part{total_proteins:04d}")
            total_proteins += len(batch_embeddings)
            
            del embedding_repr, input_ids, attention_mask, batch_embeddings
            torch.cuda.empty_cache()
    
    # Save header information
    header_file = f"{output_file}.header.json"
    with open(header_file, 'w') as f:
        json.dump(header_info, f, indent=2)
    
    return total_proteins

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

def main(fasta_input, output_directory, max_seq_length, max_batch_tokens, max_batch_size, use_quantization):
    if not os.path.isfile(fasta_input):
        raise ValueError(f"Input file not found: {fasta_input}")
    
    print(f"Processing FASTA file: {fasta_input}")
    input_name = os.path.splitext(os.path.basename(fasta_input))[0]
    
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, f"{input_name}_protein_embeddings_prot_t5.pt")
    
    print("Calculating ProtT5 embeddings...")
    total_proteins = calculate_embeddings(
        fasta_input, 
        output_file, 
        max_seq_length, 
        max_batch_tokens, 
        max_batch_size, 
        use_quantization
    )
    
    print("Combining embedding files...")
    combine_embedding_files(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate ProtT5 embeddings for proteins in FASTA files")
    parser.add_argument("fasta_input", help="Input FASTA file containing protein sequences")
    parser.add_argument("output_directory", help="Directory to save the output embeddings")
    parser.add_argument("--max_seq_length", type=int, default=12500, help="Maximum sequence length to process")
    parser.add_argument("--max_batch_tokens", type=int, default=22500, help="Maximum tokens per batch")
    parser.add_argument("--max_batch_size", type=int, default=512, help="Maximum batch size")
    parser.add_argument("--use_quantization", action="store_true", help="Use 8-bit quantization for reduced memory usage")
    args = parser.parse_args()
    
    main(
        args.fasta_input,
        args.output_directory,
        args.max_seq_length,
        args.max_batch_tokens,
        args.max_batch_size,
        args.use_quantization
    )
