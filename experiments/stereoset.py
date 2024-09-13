import argparse
import json
import os

import transformers

from bias_bench.benchmark.stereoset import StereoSetRunner
from bias_bench.model import models
from bias_bench.util import generate_experiment_id, _is_generative

# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import BitsAndBytesConfig

# the cached model
os.environ['TRANSFORMERS_CACHE'] = '/home/yandan/LLM-bias/transformers_cache'


thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs StereoSet benchmark.")
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
    "--batch_size",
    action="store",
    type=int,
    default=1,
    help="The batch size to use during StereoSet intrasentence evaluation.",
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

    # wandb.init(project="wts", name=f"{args.benchmark}_{args.category}_{args.method}", reinit=True)

    args = parser.parse_args()

    model_short_name = args.model_name_or_path.split("/")[1].split("-")[0]

    experiment_id = generate_experiment_id(
        name="stereoset",
        model_name_or_path=model_short_name,
        seed=args.seed,
    )

    print("Running StereoSet:")
    print(f" - persistent_dir: {args.persistent_dir}")
    # print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - batch_size: {args.batch_size}")
    print(f" - seed: {args.seed}")
    print(f" - cuda: {args.cuda}")
    print(f" - percentage: {args.percentage}")
    
    # try_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).bfloat16()


    model = getattr(models, "AutoModelForCausalLM")(args.model_name_or_path)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    # model_start_token = tokenizer.bos_token

    runner = StereoSetRunner(
        intrasentence_model=model,
        tokenizer=tokenizer,
        input_file=f"{args.persistent_dir}/data/stereoset/test.json",
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        is_generative=_is_generative(args.model_name_or_path),
        cuda=args.cuda,
        percentage=args.percentage,
    )
    results = runner()

        
    os.makedirs(f"{args.persistent_dir}/results/stereoset", exist_ok=True)
    with open(
        f"{args.persistent_dir}/results/stereoset/{experiment_id}.json", "w"
    ) as f:
        json.dump(results, f, indent=2)
