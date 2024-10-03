import argparse
import os
import json
import random
from tqdm import tqdm

import pandas as pd
import sys

import re

import logging

from bias_bench.adaptation import validation

import csv


# Custom stream handler to redirect print() to logging
class PrintLogger:
    def write(self, message):
        if message.strip():  # Avoid logging empty messages
            logging.info(message.strip())  # Log print statements as INFO level

    def flush(self):
        pass  # Required method, but can be left empty for now



thisdir = os.path.dirname(os.path.realpath(__file__))

def parse_args():

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
        help="HuggingFace model name or path (e.g., bert-base-uncased. "
        "model is instantiated.",
    )


    parser.add_argument(
        "--bias_type",
        action="store",
        default=None,
        choices=["gender", "race", "religion", "profession"],
        help="Determines which stereoset dataset split to evaluate against.",
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
        # choices=["0", "1", "2", "3", "4", "5", "6", "7"],
        help="choose cuda device",
    )


    parser.add_argument(
        "--percentage",
        # type=int,
        default=100,
        help="choose train percentage",
    )


    parser.add_argument(
        "--intra_inter",
        type=str,
        default="intrasentence",
        choices=["intrasentence", "intersentence"],
        help="choose cuda device",
    )

    parser.add_argument(
        "--data_percent",
        type=str,
        default="100",
        help="dataset percentage",
    )

    parser.add_argument(
        "--part",
        type=int,
        default=None,
        choices=[1,2,3,4,5],
        help="dataset_part",

    )

    return parser.parse_args()


def main():

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")


    #safely import pytorch and transformerssss

    # from bias_bench.benchmark.stereoset import dataloader
    # from bias_bench.benchmark.stereoset import StereoSetRunner
    import torch
    from bias_bench.model import models
    from bias_bench.adaptation import prompt_strings
    from datetime import datetime


    model_short_name = re.sub("-", "_", args.model_name.split("/")[1])

    print("Running StereoSet:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model_name: {args.model_name}")
    print(f" - intra-inter: {args.intra_inter}")
    print(f" - seed: {args.seed}")
    print(f" - parallel cudas: {args.cuda}")
    print(f" - percentage: {args.percentage}")
    print(f" - bias type: {args.bias_type}")
    print(f" - data percentage: {args.data_percent}")
    print(f" - part: {args.part}")


    # Configure the logging settings
    # Get the current date and time
    current_time = datetime.now()
    # Format the current time as a string (optional)
    formatted_time = current_time.strftime("%m_%d_%H_%M_%S")

    log_directory = 'data/stereoset/adapted/log'  # Replace with your desired path
    log_file = os.path.join(log_directory, f'log_{args.intra_inter}_{formatted_time}_{model_short_name}_{args.part}_{args.bias_type}.txt')   
    os.makedirs(log_directory, exist_ok=True)  

    # Set up basic configuration for logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
        handlers=[
            logging.FileHandler(log_file),  # Write logs to a file
            logging.StreamHandler()  # Optionally log to console as well
        ]
    )

    # Redirect stdout (print) to the logging system
    sys.stdout = PrintLogger()

    #write csv
    output_root = "data/stereoset/adapted"

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # with open(f'{output_root}/test_adapted_sg_comparison_{args.intra_inter}_{model_short_name}_part_{args.part}_{args.bias_type}.csv', 'w', newline='', encoding='utf-8') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=data.keys())
    #     writer.writeheader()

    if args.part is None:
        stereoset_path = "data/stereoset/test.json"
        
    else:
        stereoset_path = f"data/stereoset/adapt_part/test_part_{args.part}.json"


    visible_cuda_num = torch.cuda.device_count() # the to device cuda
    devices = []

    for i in range(visible_cuda_num):
        devices += [torch.device(f"cuda:{i}")]

    try:
        with open(stereoset_path, 'r') as file:
            stereoset_data = json.load(file)
    except:
        print(os.getcwd())

    
    if args.part is None:
        stereoset_data = stereoset_data['data'][args.intra_inter]
    

    if args.percentage != 100:
        sample_size = int(len(stereoset_data) * (float(args.percentage) / 100))
        stereoset_data = random.sample(stereoset_data, sample_size)
    

    if args.bias_type is not None:
        stereoset_data = [item for item in stereoset_data if item["bias_type"] == args.bias_type]

    # generator = models.Private_Generator(args.model_name, to_device_cuda)
    generator = models.Private_Generator(args.model_name)
    
    # prompt generate
    # add in if the prompt is not working
    if args.intra_inter == "intersentence":
        batch_prompts = prompt_strings.get_inter_stereo_prompt(stereoset_data)
    elif args.intra_inter == "intrasentence":
        batch_prompts = prompt_strings.get_intra_stereo_prompt(stereoset_data)

    adapted = []
    comparison = []

    for i in tqdm(range(len(batch_prompts))):
        try_time = 0
        
        prompt = batch_prompts[i]
        original_set = stereoset_data[i]
        batch_response = generator.response_generator(prompt)
        key_result = prompt_strings.extract_sample_from_response("stereoset", batch_response[0], try_time)


        try:    
            assert key_result['context'] != ''
        except:
            while key_result['context'] == '':
                try_time += 1
                batch_response = generator.response_generator(prompt)
                key_result = prompt_strings.extract_sample_from_response("stereoset", batch_response[0], try_time)
                # print("what is in here?")
                # print(key_result)
                if try_time > 5:
                    key_result['target'] = ""
                    key_result['context'] = ""
                    key_result['sentence_1'] = ""
                    key_result['label_1'] = ''
                    key_result['sentence_2'] = ""
                    key_result['label_2'] = ''
                    key_result['sentence_3'] = ""
                    key_result['label_3'] = ''
                    # print("what is behind here?")
                    # print(key_result)
                    break
                    


        try:
            assert key_result['bias_type'] == original_set['bias_type']
        except:
            logging.info(f"bias type has changed: , {key_result['bias_type']}, {original_set['bias_type']}")
        
        list_of_compare_contents = [{'item_id': original_set['id'],
                  'target_adapted': key_result['target'], 'target_orignal': original_set['target'],
                  'context_adapted': key_result['context'], 'context_orignal': original_set['context'],
                  'bias_adapted': key_result['bias_type'], 'bias_orignal': original_set['bias_type'],
                  'sentence_1_adapted':key_result['sentence_1'],
                  'sentence_1_orignal': original_set['sentences'][0]['sentence'],
                  'sentence_1_label_adapted': key_result['label_1'],
                  'sentence_1_label_orignal': original_set['sentences'][0]['gold_label'],
                  'sentence_2_adapted':key_result['sentence_2'],
                  'sentence_2_orignal': original_set['sentences'][1]['sentence'],
                  'sentence_2_label_adapted': key_result['label_2'],
                  'sentence_2_label_orignal': original_set['sentences'][1]['gold_label'],
                  'sentence_3_adapted':key_result['sentence_3'],
                  'sentence_3_orignal': original_set['sentences'][2]['sentence'],
                  'sentence_2_label_adapted': key_result['label_2'],
                  'sentence_2_label_orignal': original_set['sentences'][1]['gold_label'],
                  'reason': key_result['reason']
                  }]

        with open(f'{output_root}/test_adapted_sg_comparison_{args.intra_inter}_{model_short_name}_part_{args.part}_{args.bias_type}.csv', 'w', newline='', encoding='utf-8') as csvfile:

            writer = csv.DictWriter(csvfile, fieldnames=list_of_compare_contents[0].keys())
            if i == 0:
                writer.writeheader() #write header
            writer.writerow(list_of_compare_contents[0])
        
        comparison += list_of_compare_contents

        print(list_of_compare_contents)




if __name__ == "__main__":
    main()


