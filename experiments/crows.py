import argparse
import os
import json

import transformers

from bias_bench.benchmark.crows import CrowSPairsRunner
from bias_bench.model import models
from bias_bench.util import generate_experiment_id, _is_generative

os.environ['TRANSFORMERS_CACHE'] = '/data/yandan/transformers_cache'


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
    "--model_name",
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
    # choices=["gender", "race", "religion"],
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

parser.add_argument(
    "--is_sg",
    type=bool,
)



if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name="crows",
        # model=args.model,
        model_name_or_path=args.model_name,
        bias_type=args.bias_type,
    )

    print("Running CrowS-Pairs benchmark:")
    print(f" - persistent_dir: {args.persistent_dir}")
    # print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name}")
    print(f" - bias_type: {args.bias_type}")
    print(f" - seed: {args.seed}")
    print(f'- is sg: {args.is_sg}')

    # Load model and tokenizer.
    model = getattr(models, "AutoModelForCausalLM")(args.model_name)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)

    if args.is_sg is True:
        input_file = f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_adapted_sg_full.csv"
    else:
        input_file=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized.csv"

    runner = CrowSPairsRunner(
        model=model,
        tokenizer=tokenizer,
        input_file = input_file,
        bias_type=args.bias_type,
        is_generative=_is_generative(args.model_name),
        model_name_or_path=args.model_name, 
        cuda = args.cuda,
        percentage=args.percentage,
        seed=args.seed,
    )
    results = runner()

    print(f"Metric: {results}")


    print(json.dumps(results, indent=4))
    experiment_id = experiment_id.replace("/", "_")

    if args.is_sg is True:
        path = f"{args.persistent_dir}/results/crows/adapted/{experiment_id}.json"
        path_dir = f"{args.persistent_dir}/results/crows/adapted"
    else:
        path = f"{args.persistent_dir}/results/adapted_dataset/crows/{experiment_id}.json"
        path_dir = f"{args.persistent_dir}/results/adapted_dataset/crows"
    
    os.makedirs(path_dir, exist_ok=True)
    with open(path, "w") as f:
        # json.dump(results, f)
        json.dump(results, f, indent=4, sort_keys=True)