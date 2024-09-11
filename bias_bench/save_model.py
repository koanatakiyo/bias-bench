from transformers import AutoModel, AutoTokenizer


import argparse

def main():

    parser = argparse.ArgumentParser(description="Process some arguments.")

    parser.add_argument('--model_name', type=str, required=True, help='Name of the Hugging Face model to load')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the model and tokenizer
    # model_name = "bert-base-uncased"  # Replace with your model name
    model = AutoModel.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Save the model and tokenizer to a directory
    save_directory = "/home/yandan/LLM-bias/transformers_cache"
    model.save_pretrained(save_directory,safe_serialization=True)
    tokenizer.save_pretrained(save_directory,safe_serialization=True)

if __name__ == "__main__":
    main()