#!/usr/bin/env python

import argparse
import json
import os
import re
from collections import OrderedDict, defaultdict
import numpy as np
import onnxruntime as ort
import torch
from Bio import SeqIO
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BitsAndBytesConfig



def parse_fasta(fasta_file):
    proteins = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        proteins.append((str(record.seq), record.id, os.path.basename(fasta_file)))
    return proteins


def preprocess_sequence(seq, model_name):
    # Replace uncommon amino acids with X
    seq = re.sub(r"[UZOB]", "X", seq)
    # T5 expects space-separated characters
    if "t5" in model_name.lower():
        return " ".join(list(seq))
    return seq


def sliding_window_chunks(seq, window_size, stride):
    chunks = []
    for i in range(0, max(len(seq) - window_size + 1, 0), stride):
        chunks.append((seq[i : i + window_size], i))
    if len(seq) > window_size and (len(seq) - window_size) % stride != 0:
        # Add the tail so last position is covered
        chunks.append((seq[-window_size:], len(seq) - window_size))
    return chunks


def concatenate_embeddings(embedding_files):
    """
    Load one or more *.pt embedding files (dict: protein_id -> 1D tensor), stack into
    a single array and return (embeddings_np, protein_ids_list).
    """
    all_embeddings = []
    all_protein_ids = []
    for file in embedding_files:
        print(f"Reading embeddings from: {file}")
        embeddings = torch.load(file, map_location="cpu")
        for pid, emb in embeddings.items():
            all_embeddings.append(emb.numpy())
            all_protein_ids.append(pid)

    if not all_embeddings:
        raise ValueError("No embeddings found in the provided files.")

    # Validate consistent dimensionality
    dim0 = all_embeddings[0].shape[-1]
    for idx, a in enumerate(all_embeddings):
        if a.shape[-1] != dim0:
            raise ValueError(
                f"Embedding dimension mismatch at index {idx}: {a.shape[-1]} != {dim0}"
            )

    return np.array(all_embeddings), all_protein_ids


class ProteinDataset(Dataset):
    """
    Tokenizes sequences and records token lengths for dynamic batching.

    proteins: list of (sequence_str, protein_id, source_tag)
    """

    def __init__(self, proteins, tokenizer, model_name, max_length=None):
        self.proteins = proteins
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.len_dict = {}

        # Number of special tokens the tokenizer adds for a single sequence
        self._num_special = (
            tokenizer.num_special_tokens_to_add(pair=False) if max_length else 0
        )

        # Precompute token lengths for DynamicBatchSampler
        for idx, (seq, _, _) in enumerate(proteins):
            if self.max_length:
                keep = max(self.max_length - self._num_special, 0)
                seq = seq[:keep]
            proc = preprocess_sequence(seq, self.model_name)
            toks = self.tokenizer(
                proc,
                add_special_tokens=True,
                return_attention_mask=False,
                truncation=True,
                max_length=self.max_length,
            )
            token_ids = toks["input_ids"]
            self.len_dict[idx] = len(token_ids)

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        seq, pid, src = self.proteins[idx]
        if self.max_length:
            keep = max(self.max_length - self._num_special, 0)
            seq = seq[:keep]
        proc = preprocess_sequence(seq, self.model_name)
        toks = self.tokenizer(
            proc,
            add_special_tokens=True,
            return_attention_mask=False,
            truncation=True,
            max_length=self.max_length,
        )
        token_ids = toks["input_ids"]
        return token_ids, pid, src, len(token_ids)


def make_padding_collator(pad_id: int):
    """
    Returns a collate function that pads to the longest sequence in the batch
    using the provided pad token id, and builds an attention mask.
    """

    def padding_collator(batch):
        if not batch:
            raise ValueError("Empty batch received!")
        tokens, pids, srcs, lengths = zip(*batch, strict=False)
        max_len = max(lengths)

        padded, masks = [], []
        for seq in tokens:
            pad_len = max_len - len(seq)
            padded.append(seq + [pad_id] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)

        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(masks, dtype=torch.long),
            list(pids),
            list(srcs),
        )

    return padding_collator


def initialize_model(
    acceleration=None,
    model_name="Rostlab/prot_t5_xl_uniref50",
    onnx_base_dir=None,
    lora_adapter_path=None,
    revision=None,
):
    """
    Initialize model and tokenizer.

    - T5 branch supports ONNX export/run and 8-bit quantization.
    - Non-T5 (e.g., Synthyra/ESM2) loads with trust_remote_code and uses model.tokenizer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Acceleration: {acceleration}, Model: {model_name}")

    safe_name = model_name.replace("/", "_")
    base_dir = onnx_base_dir or os.getcwd()
    onnx_dir = os.path.join(base_dir, f"{safe_name}_onnx")
    onnx_path = os.path.join(onnx_dir, f"{safe_name}.onnx")

    if "t5" in model_name.lower():
        from transformers import T5EncoderModel, T5Tokenizer

        tokenizer = T5Tokenizer.from_pretrained(
            model_name, do_lower_case=False, revision=revision
        )

        if acceleration == "onnx":
            if lora_adapter_path:
                print(
                    "Warning: ONNX acceleration is not compatible with LoRA adapters here. Ignoring LoRA path."
                )
            os.makedirs(onnx_dir, exist_ok=True)
            if not os.path.exists(onnx_path):
                print(f"Exporting T5 encoder to ONNX at: {onnx_path}")
                model = T5EncoderModel.from_pretrained(
                    model_name, revision=revision
                ).eval()
                dummy_input = " ".join(list("AKLM"))
                dummy = tokenizer(dummy_input, return_tensors="pt")
                torch.onnx.export(
                    model,
                    (dummy["input_ids"], dummy["attention_mask"]),
                    onnx_path,
                    input_names=["input_ids", "attention_mask"],
                    output_names=["last_hidden_state"],
                    dynamic_axes={
                        "input_ids": {0: "batch", 1: "seq"},
                        "attention_mask": {0: "batch", 1: "seq"},
                    },
                    opset_version=12,
                )
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            sess = ort.InferenceSession(
                onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            model = sess

        elif acceleration == "quantization":
            if lora_adapter_path:
                print(
                    "Warning: Quantization is not directly compatible with LoRA adapters here. Ignoring LoRA path."
                )
            quant = BitsAndBytesConfig(load_in_8bit=True, torch_dtype=torch.float16)
            model = T5EncoderModel.from_pretrained(
                model_name,
                quantization_config=quant,
                device_map="auto",
                revision=revision,
            )
        else:
            model = T5EncoderModel.from_pretrained(model_name, revision=revision)
            if lora_adapter_path:
                print(f"Loading fine-tuned LoRA adapters from: {lora_adapter_path}")
                model = PeftModel.from_pretrained(model, lora_adapter_path)
                print("Merging LoRA adapters for efficient inference...")
                model = model.merge_and_unload()
            model.to(device)
    else:
        # Logic for ESM-like models (e.g., Synthyra/ESM2-3B) with custom code and tokenizer
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision=revision,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, revision=revision
            )

        if acceleration == "onnx":
            print(
                "Warning: ONNX acceleration is only implemented for T5 here; falling back to PyTorch for this model."
            )
        if acceleration == "quantization":
            print(
                "Note: Quantization not implemented for this custom model in this script; using PyTorch FP16/FP32."
            )

        if lora_adapter_path:
            print(f"Loading fine-tuned LoRA adapters from: {lora_adapter_path}")
            try:
                model = PeftModel.from_pretrained(model, lora_adapter_path)
                print("Merging LoRA adapters for efficient inference...")
                model = model.merge_and_unload()
            except Exception as e:
                print(
                    f"Warning: Failed to load/merge LoRA adapters for this model: {e}"
                )
        model.to(device)

    if not isinstance(model, ort.InferenceSession):
        model.eval()

    return model, tokenizer, device

def calculate_embeddings(
    fasta_file,
    output_file,
    max_seq_length,
    max_batch_tokens,
    max_batch_size,
    acceleration,
    model_name="Rostlab/prot_t5_xl_uniref50",
    lora_adapter_path=None,
    revision=None,
):
    """
    Compute mean-pooled sequence embeddings and save as a dict (pid -> 1D tensor) to output_file.
    """

    if acceleration == "None":
        acceleration = None

    base_dir = os.path.dirname(fasta_file)
    model, tokenizer, device = initialize_model(
        acceleration,
        model_name,
        onnx_base_dir=base_dir,
        lora_adapter_path=lora_adapter_path,
        revision=revision,
    )

    # Tokenizer-aware effective window to leave room for special tokens
    num_specials = tokenizer.num_special_tokens_to_add(pair=False)
    effective_window = max(max_seq_length - num_specials, 1)

    proteins = parse_fasta(fasta_file)
    print(f"Found {len(proteins)} proteins")

    # Classify using character length vs effective window to avoid truncation in "normal"
    long, normal = [], []
    for seq, pid, src in proteins:
        (long if len(seq) > effective_window else normal).append((seq, pid, src))

    print(f"Long: {len(long)}, Normal: {len(normal)}")

    all_embeddings = OrderedDict()
    header = OrderedDict()

    # Collator uses model's pad token id when available
    pad_id = (
        tokenizer.pad_token_id
        if getattr(tokenizer, "pad_token_id", None) is not None
        else 0
    )
    padding_collator = make_padding_collator(pad_id)

    # Import here to avoid a hard dependency if user changes batching strategy
    from DBSampler import DynamicBatchSampler

    # 1) Normal sequences
    if normal:
        dataset = ProteinDataset(
            normal, tokenizer, model_name, max_length=max_seq_length
        )
        sampler = DynamicBatchSampler(
            num_replicas=1,
            rank=0,
            length_dict=dataset.len_dict,
            num_buckets=32,
            min_len=0,
            max_len=max_seq_length,  # tokenized lengths will be <= max_seq_length
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

        with torch.inference_mode():
            for input_ids, attention_mask, pids, srcs in tqdm(
                loader, desc="Normal seqs"
            ):
                if acceleration == "onnx" and isinstance(model, ort.InferenceSession):
                    try:
                        out = model.run(
                            None,
                            {
                                "input_ids": input_ids.numpy(),
                                "attention_mask": attention_mask.numpy(),
                            },
                        )
                        outputs = torch.from_numpy(out[0])
                    except Exception as e:
                        if "memory" in str(e).lower():
                            print("ONNX OOM; falling back to PyTorch for this batch.")
                            input_ids, attention_mask = input_ids.to(
                                device
                            ), attention_mask.to(device)
                            outputs = model(
                                input_ids=input_ids, attention_mask=attention_mask
                            ).last_hidden_state
                        else:
                            raise
                else:
                    input_ids, attention_mask = input_ids.to(device), attention_mask.to(
                        device
                    )
                    outputs = model(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).last_hidden_state

                mask = attention_mask.unsqueeze(-1).float()
                summed = (outputs * mask).sum(dim=1)
                lengths = mask.sum(dim=1)
                mean_emb = summed / lengths

                seq_lengths = lengths.squeeze().cpu().tolist()
                if not isinstance(seq_lengths, list):
                    seq_lengths = [seq_lengths]

                for pid, src, length in zip(pids, srcs, seq_lengths, strict=False):
                    header[pid] = {
                        "length": int(length),
                        "source": src,
                        "processed_as": "normal_sequence",
                    }

                batch_embeddings = {
                    pid: emb for pid, emb in zip(pids, mean_emb.cpu(), strict=False)
                }
                all_embeddings.update(batch_embeddings)

                del outputs, input_ids, attention_mask
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    # 2) Long sequences with sliding windows
    if long:
        print("Preparing long sequence windows for batched processing...")
        window, stride = effective_window, max(1, effective_window // 2)

        all_windows = []
        long_seq_info = {}

        for seq, pid, src in tqdm(long, desc="Creating windows"):
            if len(seq) > window:
                chunks = sliding_window_chunks(seq, window, stride)
            else:
                # Edge case: len(seq) == window+1 after accounting for specials shouldn't occur, but be safe
                chunks = [(seq[:window], 0)]
            all_windows.extend([(chunk, pid, src) for chunk, _ in chunks])
            long_seq_info[pid] = (len(seq), src)

        long_dataset = ProteinDataset(
            all_windows, tokenizer, model_name, max_length=max_seq_length
        )
        long_sampler = DynamicBatchSampler(
            num_replicas=1,
            rank=0,
            length_dict=long_dataset.len_dict,
            num_buckets=32,
            min_len=0,
            max_len=max_seq_length,  # tokenized lengths will be <= max_seq_length
            max_batch_tokens=max_batch_tokens,
            max_batch_size=max_batch_size,
            shuffle=False,
            seed=0,
            drop_last=False,
        )
        long_sampler.set_epoch(0)

        long_loader = DataLoader(
            long_dataset,
            batch_sampler=long_sampler,
            collate_fn=padding_collator,
            num_workers=10,
            pin_memory=True,
        )

        window_embeddings = defaultdict(list)
        with torch.inference_mode():
            for input_ids, attention_mask, pids, _srcs in tqdm(
                long_loader, desc="Long seqs (batched windows)"
            ):
                if acceleration == "onnx" and isinstance(model, ort.InferenceSession):
                    try:
                        out = model.run(
                            None,
                            {
                                "input_ids": input_ids.numpy(),
                                "attention_mask": attention_mask.numpy(),
                            },
                        )
                        outputs = torch.from_numpy(out[0])
                    except Exception as e:
                        if "memory" in str(e).lower():
                            print("ONNX OOM; falling back to PyTorch for this batch.")
                            input_ids, attention_mask = input_ids.to(
                                device
                            ), attention_mask.to(device)
                            outputs = model(
                                input_ids=input_ids, attention_mask=attention_mask
                            ).last_hidden_state
                        else:
                            raise
                else:
                    input_ids, attention_mask = input_ids.to(device), attention_mask.to(
                        device
                    )
                    outputs = model(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).last_hidden_state

                mask = attention_mask.unsqueeze(-1).float()
                summed = (outputs * mask).sum(dim=1)
                lengths = mask.sum(dim=1)
                mean_emb = summed / lengths

                for pid, emb in zip(pids, mean_emb.cpu(), strict=False):
                    window_embeddings[pid].append(emb)

                del outputs, input_ids, attention_mask
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        print("Averaging long sequence window embeddings...")
        for pid, embs in tqdm(window_embeddings.items(), desc="Aggregating long seqs"):
            avg_emb = torch.mean(torch.stack(embs), dim=0)
            all_embeddings[pid] = avg_emb
            seq_len, src = long_seq_info[pid]
            header[pid] = {
                "length": seq_len,
                "source": src,
                "processed_as": "long_sequence_batched",
            }

    print(f"Saving {len(all_embeddings)} embeddings to {output_file}...")
    torch.save(all_embeddings, output_file)
    with open(f"{output_file}.header.json", "w") as hf:
        json.dump(header, hf, indent=2)

    return len(all_embeddings)


def _cli():
    parser = argparse.ArgumentParser(description="Calculate protein embeddings")

    parser.add_argument("fasta_input", required=True, help="Input FASTA file")
    parser.add_argument("output_directory", required=True, help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=2500)
    parser.add_argument("--max_batch_tokens", type=int, default=22500)
    parser.add_argument("--max_batch_size", type=int, default=2048)

    # Default to None so Synthyra/ESM2 runs on PyTorch by default
    parser.add_argument(
        "--acceleration",
        choices=["onnx", "quantization", "None"],
        default="onnx",
        help="ONNX is implemented only for T5 in this script.",
    )
    parser.add_argument(
        "--model_name",
        default="Synthyra/ESM2-3B",
        help="Hugging Face model ID - 'Synthyra/FastESM2_650', 'Synthyra/ESM2-3B or 'Rostlab/prot_t5_xl_uniref50'",
    )
    parser.add_argument(
        "--revision", default=None, help="Optional model revision or commit hash"
    )
    parser.add_argument(
        "--lora_adapter_path",
        default=None,
        help="Path to LoRA adapter weights (not supported with ONNX/quantization here).",
    )

    args = parser.parse_args()
    os.makedirs(args.output_directory, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.fasta_input))[0]
    output_file = os.path.join(args.output_directory, f"{base}_embeddings.pt")

    total = calculate_embeddings(
        fasta_file=args.fasta_input,
        output_file=output_file,
        max_seq_length=args.max_seq_length,
        max_batch_tokens=args.max_batch_tokens,
        max_batch_size=args.max_batch_size,
        acceleration=args.acceleration,
        model_name=args.model_name,
        lora_adapter_path=args.lora_adapter_path,
        revision=args.revision,
    )

    print(f"Total embeddings calculated: {total}")

    # Clean up T5 ONNX directory after run, if created
    if args.acceleration == "onnx" and "t5" in args.model_name.lower():
        onnx_base = os.path.dirname(args.fasta_input)
        safe_name = args.model_name.replace("/", "_")
        onnx_dir = os.path.join(onnx_base, f"{safe_name}_onnx")
        if os.path.exists(onnx_dir):
            import shutil

            shutil.rmtree(onnx_dir, ignore_errors=True)


if __name__ == "__main__":
    _cli()
