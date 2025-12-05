import os
from datasets import load_dataset
import argparse

def download_and_process_wikitext(output_dir):
    """
    Downloads WikiText-103 and saves it as a raw text file for training.
    """
    print(f"Downloading WikiText-103 to {output_dir}...")
    
    try:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "wikitext_train.txt")

    print("Processing and saving...")
    with open(output_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(dataset):
            text = item['text']
            # Basic filtering: remove empty lines or very short headers
            if text.strip() and len(text.strip()) > 20:
                f.write(text)
            
            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1} lines...", end="\r")
    
    print(f"\nSuccessfully saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess WikiText-103")
    parser.add_argument("--output_dir", type=str, default="data/text_corpus", help="Output directory")
    args = parser.parse_args()

    download_and_process_wikitext(args.output_dir)
