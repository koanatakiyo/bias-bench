from transformers import AutoModel, AutoTokenizer


import argparse
from pathlib import Path

import os

try:
    token_file = Path.home()/".cache"/"huggingface"/"token"
    if token_file.exists():
        with open(token_file, "r") as file:
            hf_token = file.read().strip()
except:
    raise FileNotFoundError("Hugging Face token file not found. Please run 'huggingface-cli login'.")


from huggingface_hub import login
os.environ["HF_TOKEN"] = hf_token
HF_TOKEN=os.getenv('HF_TOKEN')
login(token=HF_TOKEN)

def main():


    parser = argparse.ArgumentParser(description="Process some arguments.")

    parser.add_argument('--model_name', type=str, required=True, help='Name of the Hugging Face model to load')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the model and tokenizer
    # model_name = "bert-base-uncased"  # Replace with your model name
    model = AutoModel.from_pretrained(args.model_name, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, safe_serialization=True)

    # Save the model and tokenizer to a directory
    save_directory = "/home/yandan/LLM-bias/transformers_cache"
    model.save_pretrained(save_directory,safe_serialization=True)
    tokenizer.save_pretrained(save_directory,safe_serialization=True)

if __name__ == "__main__":
    main()