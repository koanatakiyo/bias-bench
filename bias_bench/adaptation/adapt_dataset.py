import argparse
import os
import json

import transformers

from bias_bench.benchmark.crows import CrowSPairsRunner
from bias_bench.benchmark.stereoset import dataloader
from bias_bench.benchmark.stereoset import StereoSetRunner
from bias_bench.model import models
from bias_bench.adaptation import prompt_context



thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs CrowS-Pairs benchmark.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)

parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased",
    # choices=["bert-base-uncased", "albert-base-v2", "roberta-base", "gpt2"],
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)


parser.add_argument(
    "--bias_type",
    action="store",
    default=None,
    # choices=["gender", "race", "religion", ""],
    help="Determines which CrowS-Pairs dataset split to evaluate against.",
)


parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=None,
    help="RNG seed. Used for logging in experiment ID.",
)


parser.add_argument(
    "--cuda",
    type=str,
    default="0",
    choices=["0", "1", "2", "3", "4", "5", "6", "7"],
    help="choose cuda device",
)


parser.add_argument(
    "--percentage",
    type=int,
    default=100,
    help="choose train percentage",
)


if __name__ == "__main__":
    args = parser.parse_args()

    model_short_name = args.model_name_or_path.split("/")[1].split("-")[0]

    print("Running StereoSet:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - batch_size: {args.batch_size}")
    print(f" - seed: {args.seed}")
    print(f" - cuda: {args.cuda}")
    print(f" - percentage: {args.percentage}")

    stereoset_path = "../../data/stereoset/test.json"
    
    with open(stereoset_path, 'r') as file:
        stereoset_data = json.load(file)

# Context simulates documents in Context aware query re-writing



