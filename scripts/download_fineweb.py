"""
download_fineweb.py — Downloads the FineWeb sample-10BT dataset from HuggingFace

What this file does:
  Uses the HuggingFace datasets library to download the "sample-10BT" subset
  of HuggingFaceFW/fineweb (~10 billion GPT-2 tokens of cleaned web text).
  Data is cached locally as Parquet files in datasets/fineweb_cache/.

Where it gets its data:
  Streamed from HuggingFace Hub over HTTPS. Resumes automatically if interrupted.

Who consumes its output:
  The cached Parquet dataset can be streamed into the C++ tokenizer via a
  separate Python script that reads the dataset and pipes text to stdout.
  The C++ tokenizer then builds word frequencies from that stream.

Why this file exists:
  FineWeb provides high-quality, deduplicated web text that complements the
  Stack Overflow and Wikipedia datasets. More diverse training data produces
  a more robust BPE vocabulary.

Prerequisites:
  pip install datasets
"""

import os
from datasets import load_dataset

def main():
    # Save it in the datasets folder at the root of the project
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(base_dir, "datasets", "fineweb_cache")
    
    # Load the dataset (sample-10BT)
    dataset = load_dataset(
        "HuggingFaceFW/fineweb", 
        name="sample-10BT", 
        split="train", 
        cache_dir=cache_dir
    )
    
    print(f"Dataset downloaded successfully to {cache_dir}")
    print(dataset)

if __name__ == "__main__":
    main()
