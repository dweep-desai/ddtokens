import os
from datasets import load_dataset

def main():
    print("Downloading HuggingFaceFW/fineweb sample-10BT...")
    
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
