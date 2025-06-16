#!/usr/bin/env python
import os
import torch
import json
import re
import argparse
from tqdm import tqdm
from Bio import SeqIO
from collections import OrderedDict
import numpy as np
from torch.utils.data import Dataset, DataLoader
import onnxruntime as ort

from DBSampler import DynamicBatchSampler
from transformers import BitsAndBytesConfig

def parse_fasta(fasta_file):
    proteins = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        proteins.append((str(record.seq), record.id, os.path.basename(fasta_file)))
    return proteins

def preprocess_sequence(seq):
    seq = re.sub(r"[UZOB]", "X", seq)
    return " ".join(list(seq))

def concatenate_embeddings(embedding_files):
    all_embeddings = []
    all_protein_ids = []
    for file in embedding_files:
        print(f"Reading embeddings from: {file}")
        embeddings = torch.load(file)
        for pid, emb in embeddings.items():
            all_embeddings.append(emb.numpy())
            all_protein_ids.append(pid)
    return np.array(all_embeddings), all_protein_ids

class ProteinDataset(Dataset):
    def __init__(self, proteins, tokenizer, max_length=None):
        self.proteins = proteins
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.len_dict = {}
        for idx, (seq, _, _) in enumerate(proteins):
            seq = seq[:max_length] if max_length else seq
            tokens = tokenizer.encode(preprocess_sequence(seq), add_special_tokens=True)
            self.len_dict[idx] = len(tokens)

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        seq, pid, src = self.proteins[idx]
        seq = seq[: self.max_length] if self.max_length else seq
        tokens = self.tokenizer.encode(preprocess_sequence(seq), add_special_tokens=True)
        return tokens, pid, src, len(tokens)

def padding_collator(batch):
    if not batch:
        raise ValueError("Empty batch received!")
    tokens, pids, srcs, lengths = zip(*batch)
    max_len = max(lengths)
    padded, masks = [], []
    for seq in tokens:
        pad_len = max_len - len(seq)
        padded.append(seq + [1] * pad_len)
        masks.append([1] * len(seq) + [0] * pad_len)
    return (
        torch.tensor(padded),
        torch.tensor(masks),
        list(pids),
        list(srcs),
    )

def initialize_model(acceleration=None, model_name="Rostlab/prot_t5_xl_uniref50", onnx_base_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Acceleration: {acceleration}, Model: {model_name}")

    # prepare safe folder name
    safe_name = model_name.replace("/", "_")
    # base directory for ONNX: same as input FASTA if provided, else cwd
    base_dir = onnx_base_dir or os.getcwd()
    onnx_dir = os.path.join(base_dir, f"{safe_name}_onnx")
    onnx_path = os.path.join(onnx_dir, f"{safe_name}.onnx")

    if "t5" in model_name.lower():
        from transformers import T5EncoderModel, T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        if acceleration == "onnx":
            os.makedirs(onnx_dir, exist_ok=True)
            if not os.path.exists(onnx_path):
                model = T5EncoderModel.from_pretrained(model_name).eval()
                dummy = tokenizer("A K L M", return_tensors="pt")
                torch.onnx.export(
                    model,
                    (dummy["input_ids"], dummy["attention_mask"]),
                    onnx_path,
                    input_names=["input_ids", "attention_mask"],
                    output_names=["last_hidden_state"],
                    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}},
                    opset_version=12,
                )
                del model
            sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
            model = sess
        elif acceleration == "quantization":
            quant = BitsAndBytesConfig(load_in_8bit=True, torch_dtype=torch.float16)
            model = T5EncoderModel.from_pretrained(model_name, quantization_config=quant, device_map="auto")
        else:
            model = T5EncoderModel.from_pretrained(model_name).to(device)

    else:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        if acceleration == "onnx":
            os.makedirs(onnx_dir, exist_ok=True)
            if not os.path.exists(onnx_path):
                model = AutoModel.from_pretrained(model_name).eval()
                dummy = tokenizer("A K L M", return_tensors="pt")
                torch.onnx.export(
                    model,
                    (dummy["input_ids"], dummy["attention_mask"]),
                    onnx_path,
                    input_names=["input_ids", "attention_mask"],
                    output_names=["last_hidden_state"],
                    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}},
                    opset_version=12,
                )
                del model
            sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
            model = sess
        elif acceleration == "quantization":
            quant = BitsAndBytesConfig(load_in_8bit=True, torch_dtype=torch.float16)
            model = AutoModel.from_pretrained(model_name, quantization_config=quant, device_map="auto")
        else:
            model = AutoModel.from_pretrained(model_name).to(device)

    if not isinstance(model, ort.InferenceSession):
        model.eval()
    return model, tokenizer, device

def sliding_window_chunks(seq, window_size, stride):
    chunks = []
    for i in range(0, len(seq) - window_size + 1, stride):
        chunks.append((seq[i : i + window_size], i))
    if len(seq) % stride != 0:
        chunks.append((seq[-window_size:], len(seq) - window_size))
    return chunks

def process_long_sequence(seq, pid, model, tokenizer, device, max_seq_length, acceleration):
    window, stride = max_seq_length, max_seq_length // 2
    embs = []
    for chunk, _ in sliding_window_chunks(seq, window, stride):
        tokens = tokenizer.encode(preprocess_sequence(chunk), add_special_tokens=True, return_tensors="pt")
        mask = torch.ones_like(tokens)
        if acceleration == "onnx":
            out = model.run(None, {"input_ids": tokens.numpy(), "attention_mask": mask.numpy()})[0]
            emb = torch.from_numpy(out)
        else:
            tokens, mask = tokens.to(device), mask.to(device)
            with torch.inference_mode():
                emb = model(input_ids=tokens, attention_mask=mask).last_hidden_state
        embs.append(emb.mean(dim=1).cpu())
    return torch.mean(torch.stack(embs), dim=0)[0]

def calculate_embeddings(
    fasta_file,
    output_file,
    max_seq_length,
    max_batch_tokens,
    max_batch_size,
    acceleration,
    model_name="Rostlab/prot_t5_xl_uniref50",
):
    base_dir = os.path.dirname(fasta_file)
    model, tokenizer, device = initialize_model(acceleration, model_name, onnx_base_dir=base_dir)
    proteins = parse_fasta(fasta_file)
    print(f"Found {len(proteins)} proteins")

    long, normal = [], []
    for seq, pid, src in proteins:
        (long if len(seq) > max_seq_length else normal).append((seq, pid, src))
    print(f"Long: {len(long)}, Normal: {len(normal)}")

    dataset = ProteinDataset(normal, tokenizer, max_length=max_seq_length)
    sampler = DynamicBatchSampler(
        num_replicas=1,
        rank=0,
        length_dict=dataset.len_dict,
        num_buckets=32,
        min_len=0,
        max_len=max_seq_length,
        max_batch_tokens=max_batch_tokens,
        max_batch_size=max_batch_size,
        shuffle=False,
        seed=0,
        drop_last=False,
    )
    sampler.set_epoch(0)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=padding_collator,
        num_workers=10,
        pin_memory=True,
    )

    total = 0
    header = OrderedDict()
    with torch.inference_mode():
        for input_ids, attention_mask, pids, srcs in loader:
            if acceleration == "onnx":
                try:
                    out = model.run(None, {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()})
                    outputs = torch.from_numpy(out[0])
                except Exception as e:
                    if "memory" in str(e).lower():
                        print("ONNX OOM, switch to quantization")
                        del model; torch.cuda.empty_cache()
                        acceleration = "quantization"
                        model, tokenizer, device = initialize_model(acceleration, model_name, onnx_base_dir=base_dir)
                        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                    else:
                        raise
            else:
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            mask = attention_mask.unsqueeze(-1).float()
            summed = (outputs * mask).sum(dim=1)
            lengths = mask.sum(dim=1)
            mean_emb = summed / lengths
            batch = {pid: emb for pid, emb in zip(pids, mean_emb)}
            torch.save(batch, f"{output_file}.part{total:04d}")
            total += len(batch)
            del outputs, input_ids, attention_mask, batch
            torch.cuda.empty_cache()

    if long:
        long_batch = OrderedDict()
        for seq, pid, src in tqdm(long, desc="Long seqs"):
            emb = process_long_sequence(seq, pid, model, tokenizer, device, max_seq_length, acceleration)
            long_batch[pid] = emb
            header[pid] = {"length": len(seq), "source": src, "processed_as": "long_sequence"}
            if len(long_batch) >= max_batch_size:
                torch.save(long_batch, f"{output_file}.part{total:04d}")
                total += len(long_batch)
                long_batch = OrderedDict()
        if long_batch:
            torch.save(long_batch, f"{output_file}.part{total:04d}")
            total += len(long_batch)

    with open(f"{output_file}.header.json", "w") as hf:
        json.dump(header, hf, indent=2)

    return total

def combine_embedding_files(output_file):
    path, base = os.path.dirname(output_file), os.path.basename(output_file)
    parts = sorted(f for f in os.listdir(path) if f.startswith(base) and re.search(r"\.part\d+$", f))
    all_emb = OrderedDict()
    count = 0
    dim = None
    for part in tqdm(parts, desc="Combining parts"):
        data = torch.load(os.path.join(path, part))
        for pid, emb in data.items():
            all_emb[pid] = emb
            count += 1
            dim = dim or emb.shape[0]
        os.remove(os.path.join(path, part))
    print(f"Total proteins: {count}, Embedding dim: {dim}")
    torch.save(all_emb, output_file)
    print(f"Saved {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Calculate protein embeddings")
    parser.add_argument("fasta_input", help="Input FASTA file")
    parser.add_argument("output_directory", help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=7500)
    parser.add_argument("--max_batch_tokens", type=int, default=12500)
    parser.add_argument("--max_batch_size", type=int, default=512)
    parser.add_argument("--acceleration", choices=["onnx", "quantization", None], default=None)
    parser.add_argument(
        "--model_name",
        default="Rostlab/prot_t5_xl_uniref50",
        help="HuggingFace model ID",
    )
    args = parser.parse_args()

    os.makedirs(args.output_directory, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.fasta_input))[0]
    output_file = os.path.join(args.output_directory, f"{base}_embeddings.pt")

    total = calculate_embeddings(
        args.fasta_input,
        output_file,
        args.max_seq_length,
        args.max_batch_tokens,
        args.max_batch_size,
        args.acceleration,
        args.model_name,
    )
    print(f"Total embeddings: {total}")

    if args.acceleration == "onnx":
        onnx_base = os.path.dirname(args.fasta_input)
        safe_name = args.model_name.replace("/", "_")
        onnx_dir = os.path.join(onnx_base, f"{safe_name}_onnx")
        if os.path.exists(onnx_dir):
            import shutil
            shutil.rmtree(onnx_dir, ignore_errors=True)

    print("Combining parts...")
    combine_embedding_files(output_file)

if __name__ == "__main__":
    main()
