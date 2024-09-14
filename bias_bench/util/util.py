# use this code to find out how much tokens are in the prefix for each model

from transformers import AutoTokenizer

def get_self_debias_prefix_token_count(model_name):
    DEBIASING_PREFIXES = {
        "race-color": "The following text discriminates against people because of their race/color: ",
        "gender": "The following text discriminates against people because of their gender: ",
        "socioeconomic": "The following text discriminates against people because of their socioeconomic status/occupation: ",
        "sexual-orientation": "The following text discriminates against people because of their sexual orientiation: ",
        "religion": "The following text discriminates against people because of their religion: ",
        "age": "The following text discriminates against people because of their age: ",
        "nationality": "The following text discriminates against people because of their nationality: ",
        "disability": "The following text discriminates against people because of their disability: ",
        "physical-appearance": "The following text discriminates against people because of their physical appearance: ",
    }
    # Encode the prefixes
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prefix_token_counts = {}

    for bias_type, prefix in DEBIASING_PREFIXES.items():
        input_ids = tokenizer.encode(prefix, return_tensors="pt")
        prefix_token_counts[bias_type] = len(input_ids[0])
        print(f"Model: {model_name}, Bias Type: {bias_type}, Prefix Token Count: {len(input_ids[0])}")

    return prefix_token_counts

def _is_generative(model_path):
    # Checks if we are running an autoregressive model.
    model_list = ["gpt", "llama", "mistral", "phi"]
    for model_name in model_list: 
        if model_path.lower().find(model_name) != -1:
            return True
    return False
    
    # return model in [
    #     "GPT2LMHeadModel",
    #     "SentenceDebiasGPT2LMHeadModel",
    #     "INLPGPT2LMHeadModel",
    #     "CDAGPT2LMHeadModel",
    #     "DropoutGPT2LMHeadModel",
    #     "SelfDebiasGPT2LMHeadModel",
    # ]


def get_self_debias_prefix_token_count(model_name):
    DEBIASING_PREFIXES = {
        "race-color": "The following text discriminates against people because of their race/color: ",
        "gender": "The following text discriminates against people because of their gender: ",
        "socioeconomic": "The following text discriminates against people because of their socioeconomic status/occupation: ",
        "sexual-orientation": "The following text discriminates against people because of their sexual orientiation: ",
        "religion": "The following text discriminates against people because of their religion: ",
        "age": "The following text discriminates against people because of their age: ",
        "nationality": "The following text discriminates against people because of their nationality: ",
        "disability": "The following text discriminates against people because of their disability: ",
        "physical-appearance": "The following text discriminates against people because of their physical appearance: ",
    }
    # Encode the prefixes
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prefix_token_counts = {}

    for bias_type, prefix in DEBIASING_PREFIXES.items():
        input_ids = tokenizer.encode(prefix, return_tensors="pt")
        prefix_token_counts[bias_type] = len(input_ids[0])
        print(f"Model: {model_name}, Bias Type: {bias_type}, Prefix Token Count: {len(input_ids[0])}")

    return prefix_token_counts

def _is_self_debias(model):
    # Checks if we are running a Self-Debias model.
    return model in [
        "SelfDebiasGPT2LMHeadModel",
        "SelfDebiasBertForMaskedLM",
        "SelfDebiasAlbertForMaskedLM",
        "SelfDebiasRobertaForMaskedLM",
    ]


def get_target_modules_for_model(model_name):
    # Return all linear layers to be as good as full fine-tuning performance
    target_modules = {
        "gpt2": ["c_attn", "c_proj", "c_fc"],
        "microsoft/phi-2": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
        "meta-llama/Llama-2-7b-hf": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }

    return target_modules[model_name]  



def start_token_mapper(model_name):
    start_token_mapper = {
        "gpt2": "<|endoftext|>",
        "llama": "<s>",
        "phi": "<|endoftext|>",
    }
    return start_token_mapper[model_name]